"""Run a paired deterministic feasibility check for the CBF shield."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis_orbit.envs import ContinuousConjunctionEnv
from aegis_orbit.envs.cw_relative_dynamics import relative_drift_acceleration
from aegis_orbit.shields import ControlBarrierShield, SafetySupervisor
from benchmarks.oracle_reference import get, load_config
from benchmarks.run_unshielded_benchmark import step_reward_and_termination
from benchmarks.run_shield_comparison import initial_state_hash

EPISODE_FIELDS = (
    "scenario", "condition", "eval_seed", "d_safe_m", "alpha1_per_s",
    "alpha2_per_s", "initial_safe", "initial_psi_m2_per_s",
    "initial_hocbf_admissible", "initial_state_hash", "collided",
    "min_distance_m", "fuel_mps", "reward", "episode_steps",
    "completed_horizon", "shield_evaluations", "feasible_steps",
    "infeasible_steps", "projection_steps", "backup_steps",
    "max_feasible_constraint_residual", "max_box_violation", "solver_error",
)
STEP_FIELDS = (
    "scenario", "condition", "eval_seed", "step", "mode", "initial_safe",
    "initial_hocbf_admissible", "h_m2", "psi_m2_per_s", "b_m2_per_s2",
    "min_box_a_dot_u_m2_per_s2",
    "nominal_a_dot_u_m2_per_s2", "applied_a_dot_u_m2_per_s2",
    "constraint_residual_m2_per_s2", "u_nom_x_mps2", "u_nom_y_mps2",
    "u_nom_z_mps2", "u_applied_x_mps2", "u_applied_y_mps2",
    "u_applied_z_mps2", "feasible", "projection_applied", "backup_used",
    "distance_m", "collided",
)


def _environment(config: dict[str, Any]) -> ContinuousConjunctionEnv:
    low = float(get(config, "environment", "initial_offset_m", "low"))
    high = float(get(config, "environment", "initial_offset_m", "high"))
    return ContinuousConjunctionEnv(
        n=float(get(config, "environment", "n")),
        dt=float(get(config, "environment", "dt")),
        horizon_steps=int(get(config, "environment", "horizon_steps")),
        u_max=float(get(config, "environment", "u_max")),
        d_safe=0.0,
        initial_offset_m=0.5 * (low + high),
        closing_speed_mps=float(get(config, "environment", "closing_speed_mps")),
        initial_offset_range_m=(low, high),
    )


def _barrier_diagnostics(
    shield: ControlBarrierShield, state: np.ndarray, nominal: np.ndarray, applied: np.ndarray
) -> dict[str, float]:
    relative_position = state[:3] - state[6:9]
    relative_velocity = state[3:6] - state[9:12]
    drift = np.asarray(shield.drift_fn(state), dtype=np.float64)
    h = shield.barrier(relative_position)
    h_dot = 2.0 * float(np.dot(relative_position, relative_velocity))
    psi = h_dot + shield.alpha1 * h
    a = -2.0 * relative_position
    b = (
        2.0 * float(np.dot(relative_velocity, relative_velocity))
        + 2.0 * float(np.dot(relative_position, drift))
        + shield.alpha1 * h_dot
        + shield.alpha2 * psi
    )
    return {
        "h": h,
        "psi": psi,
        "b": b,
        "min_box_a_dot_u": -shield.u_max * float(np.sum(np.abs(a))),
        "nominal_a_dot_u": float(np.dot(a, np.clip(nominal, -shield.u_max, shield.u_max))),
        "applied_a_dot_u": float(np.dot(a, applied)),
        "constraint_residual": float(np.dot(a, applied) - b),
        "box_violation": float(np.max(np.abs(applied) - shield.u_max)),
    }


def run_episode(
    *,
    scenario_name: str,
    condition: str,
    config: dict[str, Any],
    seed: int,
    d_safe: float,
    alpha1: float,
    alpha2: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = _environment(config)
    rng = np.random.default_rng(seed)
    initial_state = environment.reset(rng)
    initial_hash = initial_state_hash(initial_state)
    state = initial_state.copy()
    u_max = float(get(config, "environment", "u_max"))
    mean_motion = float(get(config, "environment", "n"))
    shield = None
    supervisor = None
    if condition == "shield_on":
        shield = ControlBarrierShield(
            d_safe=d_safe,
            alpha1=alpha1,
            alpha2=alpha2,
            u_max=u_max,
            drift_fn=lambda current: relative_drift_acceleration(
                np.asarray(current, dtype=np.float64)[:6], mean_motion
            ),
            allow_unvalidated_d_safe=True,
        )
        supervisor = SafetySupervisor(shield)

    initial_relative_position = initial_state[:3] - initial_state[6:9]
    initial_safe = bool(
        float(np.dot(initial_relative_position, initial_relative_position)) >= d_safe**2
    ) if condition == "shield_on" else ""
    initial_psi = ""
    initial_hocbf_admissible = ""
    if shield is not None:
        initial_relative_velocity = initial_state[3:6] - initial_state[9:12]
        initial_h = shield.barrier(initial_relative_position)
        initial_h_dot = 2.0 * float(
            np.dot(initial_relative_position, initial_relative_velocity)
        )
        initial_psi = initial_h_dot + alpha1 * initial_h
        initial_hocbf_admissible = bool(initial_safe and initial_psi >= 0.0)
    hard_body_radius = float(get(config, "safety_geometry", "hard_body_radius_m"))
    proximity_weight = float(get(config, "reward", "proximity_weight"))
    reward_components = {
        "fuel_term": 0.0,
        "collision_term": 0.0,
        "proximity_term": 0.0,
        "survival_term": 0.0,
    }
    min_distance = math.inf
    fuel = 0.0
    collided = False
    clipped_steps = 0
    feasible_steps = 0
    infeasible_steps = 0
    projection_steps = 0
    backup_steps = 0
    max_constraint_residual = 0.0
    max_box_violation = 0.0
    solver_error = ""
    step_rows: list[dict[str, Any]] = []
    horizon = int(get(config, "environment", "horizon_steps"))

    for step in range(1, horizon + 1):
        nominal = np.array([-u_max, 0.0, 0.0], dtype=np.float64)
        diagnostics: dict[str, float] = {}
        if supervisor is None:
            applied = nominal.copy()
            mode = "nominal"
        else:
            try:
                applied, mode = supervisor.act(state, nominal)
            except Exception as error:  # retained per episode for explicit reporting
                solver_error = f"{type(error).__name__}: {error}"
                break
            diagnostics = _barrier_diagnostics(shield, state, nominal, applied)
            if mode == "backup":
                infeasible_steps += 1
                backup_steps += 1
            else:
                feasible_steps += 1
                projection_steps += int(mode == "shielded")
                max_constraint_residual = max(
                    max_constraint_residual, diagnostics["constraint_residual"]
                )
                max_box_violation = max(
                    max_box_violation, diagnostics["box_violation"]
                )

        prior_collided = collided
        next_state, info = environment.step(
            applied, collision_radius_m=hard_body_radius
        )
        distance = float(info["min_distance_in_step"])
        min_distance = min(min_distance, distance)
        collided = collided or distance < hard_body_radius
        clipped_action = np.clip(applied, -u_max, u_max)
        fuel += float(np.linalg.norm(clipped_action) * environment.dt)
        clipped_steps += int(bool(info["clipped"]))
        current_rewards, done, collided = step_reward_and_termination(
            applied,
            info,
            config,
            prior_collided=prior_collided,
            at_horizon=step == horizon,
            proximity_weight=proximity_weight,
            clipped_action=clipped_action,
        )
        for key in reward_components:
            reward_components[key] += current_rewards[key]

        if condition == "shield_on":
            feasible = mode != "backup"
            residual = diagnostics["constraint_residual"]
            step_rows.append({
                "scenario": scenario_name,
                "condition": condition,
                "eval_seed": seed,
                "step": step,
                "mode": mode,
                "initial_safe": int(initial_safe),
                "initial_hocbf_admissible": int(initial_hocbf_admissible),
                "h_m2": diagnostics["h"],
                "psi_m2_per_s": diagnostics["psi"],
                "b_m2_per_s2": diagnostics["b"],
                "min_box_a_dot_u_m2_per_s2": diagnostics["min_box_a_dot_u"],
                "nominal_a_dot_u_m2_per_s2": diagnostics["nominal_a_dot_u"],
                "applied_a_dot_u_m2_per_s2": diagnostics["applied_a_dot_u"],
                "constraint_residual_m2_per_s2": residual,
                "u_nom_x_mps2": nominal[0], "u_nom_y_mps2": nominal[1],
                "u_nom_z_mps2": nominal[2], "u_applied_x_mps2": applied[0],
                "u_applied_y_mps2": applied[1], "u_applied_z_mps2": applied[2],
                "feasible": int(feasible),
                "projection_applied": int(mode == "shielded"),
                "backup_used": int(mode == "backup"),
                "distance_m": distance,
                "collided": int(distance < hard_body_radius),
            })
        state = next_state
        if done:
            break

    actual_steps = step if "step" in locals() else 0
    row = {
        "scenario": scenario_name,
        "condition": condition,
        "eval_seed": seed,
        "d_safe_m": d_safe if condition == "shield_on" else "",
        "alpha1_per_s": alpha1 if condition == "shield_on" else "",
        "alpha2_per_s": alpha2 if condition == "shield_on" else "",
        "initial_safe": int(initial_safe) if condition == "shield_on" else "",
        "initial_psi_m2_per_s": initial_psi,
        "initial_hocbf_admissible": int(initial_hocbf_admissible) if condition == "shield_on" else "",
        "initial_state_hash": initial_hash,
        "collided": int(collided),
        "min_distance_m": min_distance,
        "fuel_mps": fuel,
        "reward": sum(reward_components.values()),
        "episode_steps": actual_steps,
        "completed_horizon": int(actual_steps == horizon),
        "shield_evaluations": actual_steps if condition == "shield_on" else 0,
        "feasible_steps": feasible_steps,
        "infeasible_steps": infeasible_steps,
        "projection_steps": projection_steps,
        "backup_steps": backup_steps,
        "max_feasible_constraint_residual": max_constraint_residual,
        "max_box_violation": max_box_violation,
        "solver_error": solver_error,
    }
    return row, step_rows


def summarize(episode_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    ci_rows = []
    z = 1.959963984540054
    groups = sorted({(row["scenario"], row["condition"]) for row in episode_rows})
    for scenario, condition in groups:
        group = [row for row in episode_rows if (row["scenario"], row["condition"]) == (scenario, condition)]
        count = len(group)
        collisions = sum(int(row["collided"]) for row in group)
        minimum_distances = np.asarray([float(row["min_distance_m"]) for row in group])
        is_shield = condition == "shield_on"
        projection_episodes = sum(int(row["projection_steps"]) > 0 for row in group) if is_shield else 0
        infeasible_episodes = sum(int(row["infeasible_steps"]) > 0 for row in group) if is_shield else 0
        backup_episodes = sum(int(row["backup_steps"]) > 0 for row in group) if is_shield else 0
        steps = sum(int(row["shield_evaluations"]) for row in group)
        projections = sum(int(row["projection_steps"]) for row in group)
        infeasible = sum(int(row["infeasible_steps"]) for row in group)
        backups = sum(int(row["backup_steps"]) for row in group)
        summary_rows.append({
            "scenario": scenario, "condition": condition, "episodes": count,
            "collisions": collisions, "collision_rate": collisions / count,
            "infeasible_episodes": infeasible_episodes,
            "infeasible_episode_rate": infeasible_episodes / count if is_shield else "",
            "infeasible_shield_steps": infeasible,
            "feasible_shield_steps": steps - infeasible,
            "projection_episodes": projection_episodes,
            "projection_episode_rate": projection_episodes / count if is_shield else "",
            "projection_steps": projections,
            "projection_step_rate": projections / steps if steps else "",
            "backup_episodes": backup_episodes,
            "backup_steps": backups,
            "mean_min_distance_m": float(np.mean(minimum_distances)),
            "minimum_min_distance_m": float(np.min(minimum_distances)),
            "q05_min_distance_m": float(np.quantile(minimum_distances, 0.05)),
            "median_min_distance_m": float(np.median(minimum_distances)),
            "q95_min_distance_m": float(np.quantile(minimum_distances, 0.95)),
            "mean_fuel_mps": float(np.mean([float(row["fuel_mps"]) for row in group])),
            "mean_reward": float(np.mean([float(row["reward"]) for row in group])),
            "completed_horizons": sum(int(row["completed_horizon"]) for row in group),
            "solver_errors": sum(bool(row["solver_error"]) for row in group),
            "max_feasible_constraint_residual": max(float(row["max_feasible_constraint_residual"]) for row in group),
            "max_box_violation": max(float(row["max_box_violation"]) for row in group),
        })
        events = [("collision", collisions)]
        if is_shield:
            events += [
                ("episode_with_infeasibility", infeasible_episodes),
                ("episode_with_projection", projection_episodes),
                ("episode_with_backup", backup_episodes),
            ]
        for event, successes in events:
            p = successes / count
            denominator = 1.0 + z * z / count
            center = (p + z * z / (2.0 * count)) / denominator
            half = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
            ci_rows.append({
                "scenario": scenario, "condition": condition, "event": event,
                "events": successes, "episodes": count, "rate": p,
                "wilson_95_low": center - half, "wilson_95_high": center + half,
            })
    return summary_rows, ci_rows


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path, seed_start: int = 10000, seed_count: int = 20) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path("configs/sac_unshielded.yaml")
    base_config = load_config(config_path)
    u_max = float(get(base_config, "environment", "u_max"))
    scenario_specs = (
        {
            "name": "active_projection",
            "offset_low_m": 60.0, "offset_high_m": 60.1,
            "closing_speed_mps": 0.05,
            "horizon_steps": 5, "d_safe_m": 50.0,
            "alpha1_per_s": 0.01, "alpha2_per_s": 0.008,
        },
        {
            "name": "infeasible_backup",
            "offset_low_m": 67.0, "offset_high_m": 67.01,
            "closing_speed_mps": 0.5,
            "horizon_steps": 5, "d_safe_m": 67.0,
            "alpha1_per_s": 0.5, "alpha2_per_s": 0.5,
        },
    )
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for spec in scenario_specs:
        config = deepcopy(base_config)
        config["environment"]["initial_offset_m"]["low"] = spec["offset_low_m"]
        config["environment"]["initial_offset_m"]["high"] = spec["offset_high_m"]
        config["environment"]["closing_speed_mps"] = spec["closing_speed_mps"]
        config["environment"]["horizon_steps"] = spec["horizon_steps"]
        for seed in range(seed_start, seed_start + seed_count):
            for condition in ("shield_off", "shield_on"):
                row, steps = run_episode(
                    scenario_name=spec["name"], condition=condition,
                    config=config, seed=seed, d_safe=spec["d_safe_m"],
                    alpha1=spec["alpha1_per_s"], alpha2=spec["alpha2_per_s"],
                )
                episode_rows.append(row)
                step_rows.extend(steps)
    summary_rows, ci_rows = summarize(episode_rows)
    write_csv(output_dir / "episodes.csv", EPISODE_FIELDS, episode_rows)
    write_csv(output_dir / "steps.csv", STEP_FIELDS, step_rows)
    write_csv(output_dir / "summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(output_dir / "confidence_intervals.csv", list(ci_rows[0]), ci_rows)
    manifest = {
        "objective": "analytical projection and infeasibility behavior under paired controlled CW scenarios",
        "official_config_path": str(config_path.resolve()),
        "official_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "official_config_modified": False,
        "dataset_used": False,
        "policy": "fixed deterministic inward radial action [-u_max, 0, 0]; no SAC policy or training",
        "truth_model": "exact CW",
        "action_units": "relative acceleration m/s^2, per-axis bounded by u_max",
        "evaluation_seeds": {"start": seed_start, "count": seed_count, "end_inclusive": seed_start + seed_count - 1},
        "episodes": len(episode_rows),
        "steps_per_episode": 5,
        "scenarios": list(scenario_specs),
        "runtime_seconds": time.monotonic() - started,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    for row in summary_rows:
        print(
            f"SUMMARY {row['scenario']} {row['condition']} n={row['episodes']} "
            f"collisions={row['collisions']} feasible={row['feasible_shield_steps']} "
            f"infeasible={row['infeasible_shield_steps']} projections={row['projection_steps']} "
            f"backup={row['backup_steps']} min_distance_m={row['minimum_min_distance_m']:.6f} "
            f"mean_fuel_mps={row['mean_fuel_mps']:.9g} reward={row['mean_reward']:.9g}"
        )


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-count", type=int, default=20)
    args = parser.parse_args()
    run(args.output_dir, args.seed_start, args.seed_count)


if __name__ == "__main__":
    cli()
