"""Run the post-hoc shield comparison for verifier-selected primary policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from aegis_orbit.agents import SACAgent
from aegis_orbit.envs import ContinuousConjunctionEnv, PerturbedConjunctionEnv
from aegis_orbit.envs.cw_relative_dynamics import relative_drift_acceleration
from aegis_orbit.shields import ControlBarrierShield, SafetySupervisor
from aegis_orbit.utils.streaming_csv import StreamingCSVWriter
from .oracle_reference import get, load_config
from .run_unshielded_benchmark import _prepare_output_dir, deterministic_evaluation_policy

LABEL_TEMPLATE = "PARAMETER-SWEEP d_safe = {value:g} m (not a derived safe radius)"
D_SAFE_VALUES = (40.0, 67.0)
EXPLORATORY_D_SAFE_VALUES = (30.0, 40.0, 50.0, 60.0, 67.0)
TRUTH_MODELS = ("cw", "cw_j2")
SHIELD_ALPHA1 = 0.5
SHIELD_ALPHA2 = 0.5
DRAG_BLOCK_REASON = (
    "drag ballistic coefficients and atmosphere settings not sourced; PLACEHOLDER"
)
EPISODE_FIELDS = (
    "condition",
    "d_safe",
    "truth_model",
    "train_seed",
    "eval_seed",
    "collided",
    "min_distance_m",
    "fuel_mps",
    "clipped_fraction",
    "nominal_steps",
    "shielded_steps",
    "backup_steps",
    "infeasible_steps",
    "first_infeasible_step",
    "relative_distance_at_first_infeasible_m",
    "relative_speed_at_first_infeasible_mps",
    "initial_state_hash",
    "policy_sha256",
    "label",
    "shield_off_reused_across_d_safe",
    "solver_error",
    "feasible_step_count",
    "constraint_violation_max",
    "box_violation_max",
)
STEP_FIELDS = (
    "condition",
    "d_safe",
    "truth_model",
    "label",
    "train_seed",
    "eval_seed",
    "step",
    "mode",
    "feasible",
    "a_dot_u",
    "b",
    "u_x",
    "u_y",
    "u_z",
)


def summarize_modes(modes: list[str]) -> dict[str, int]:
    """Count episode mode totals; backup activations are infeasible steps."""
    return {
        "nominal": modes.count("nominal"),
        "shielded": modes.count("shielded"),
        "backup": modes.count("backup"),
        "infeasible": modes.count("backup"),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_final_policy_rows(
    policy_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    index_path = policy_dir / "index.csv"
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["method"], int(row["train_seed"]))
        previous = selected.get(key)
        if previous is None or int(row["step"]) > int(previous["step"]):
            selected[key] = row
    policies = sorted(
        (row for (method, _), row in selected.items() if method == "sac_primary"),
        key=lambda row: int(row["train_seed"]),
    )
    if len(rows) != 19:
        raise ValueError(f"expected 19 policy index rows, found {len(rows)}")
    if len(policies) != 5:
        raise ValueError(f"expected 5 final sac_primary policies, found {len(policies)}")
    return rows, policies


def final_policy_rows(policy_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows, policies = select_final_policy_rows(policy_dir)
    verified = []
    for row in policies:
        path = policy_dir / row["file"]
        if not path.is_file():
            raise FileNotFoundError(f"selected policy missing: {path}")
        digest = sha256_file(path)
        expected = row.get("sha256", "").strip().lower()
        if not expected or digest != expected:
            raise ValueError(
                f"selected policy hash mismatch: {row['file']} "
                f"expected={expected!r} observed={digest}"
            )
        verified.append({**row, "observed_sha256": digest})
    return rows, verified


def validate_d_safe_values(
    values: list[float], *, allow_exploratory: bool = False
) -> None:
    """Keep the default comparison grid gated; opt in to the labelled pilot grid."""
    allowed_values = (
        EXPLORATORY_D_SAFE_VALUES if allow_exploratory else D_SAFE_VALUES
    )
    if not values or any(value not in allowed_values for value in values):
        raise ValueError(f"--d-safe accepts only {allowed_values}")
    if len(set(values)) != len(values):
        raise ValueError("--d-safe values must be unique")


def filter_policy_rows(
    policies: list[dict[str, str]], train_seed: int | None
) -> list[dict[str, str]]:
    """Optionally retain one already hash-verified policy by training seed."""
    if train_seed is None:
        return policies
    filtered = [row for row in policies if int(row["train_seed"]) == train_seed]
    if len(filtered) != 1:
        raise ValueError(f"expected one verified sac_primary policy for seed {train_seed}")
    return filtered


def load_agent(actor_path: Path, config: dict[str, Any]) -> SACAgent:
    import torch

    state_dict = torch.load(actor_path, map_location="cpu", weights_only=True)
    state_dim = int(state_dict["body.0.weight"].shape[1])
    action_dim = int(state_dict["mean.weight"].shape[0])
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        u_max=float(get(config, "environment", "u_max")),
        learning_rate=float(get(config, "training", "actor_lr")),
        gamma=float(get(config, "training", "gamma")),
        tau=float(get(config, "training", "tau")),
        target_entropy=float(get(config, "training", "target_entropy")),
    )
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    return agent


def initial_state_hash(state: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(state, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def assert_initial_state_hashes(rows: list[dict[str, Any]]) -> None:
    """Require paired shield conditions to begin at identical states."""
    pairs: dict[tuple[int, int, str, float], dict[str, str]] = {}
    for row in rows:
        key = (
            int(row["train_seed"]),
            int(row["eval_seed"]),
            str(row["truth_model"]),
            float(row["d_safe"]),
        )
        pairs.setdefault(key, {})[str(row["condition"])] = str(row["initial_state_hash"])
    for key, conditions in pairs.items():
        if set(conditions) != {"shield_off", "shield_on"}:
            raise AssertionError(f"missing paired condition for {key}: {conditions}")
        if conditions["shield_off"] != conditions["shield_on"]:
            raise AssertionError(f"initial-state hash mismatch for {key}: {conditions}")


def _barrier_terms(
    shield: ControlBarrierShield, state: np.ndarray
) -> tuple[np.ndarray, float]:
    relative_position = state[:3] - state[6:9]
    relative_velocity = state[3:6] - state[9:12]
    drift = (
        np.zeros(3, dtype=np.float64)
        if shield.drift_fn is None
        else np.asarray(shield.drift_fn(state), dtype=np.float64)
    )
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
    return a, b


def _new_environment(
    config: dict[str, Any], truth_model: str = "cw"
) -> ContinuousConjunctionEnv | PerturbedConjunctionEnv:
    low = float(get(config, "environment", "initial_offset_m", "low"))
    high = float(get(config, "environment", "initial_offset_m", "high"))
    parameters = {
        "n": float(get(config, "environment", "n")),
        "dt": float(get(config, "environment", "dt")),
        "horizon_steps": int(get(config, "environment", "horizon_steps")),
        "u_max": float(get(config, "environment", "u_max")),
        "d_safe": float(get(config, "safety_geometry", "d_safe_legacy_unused_m")),
        "initial_offset_m": 0.5 * (low + high),
        "closing_speed_mps": float(get(config, "environment", "closing_speed_mps")),
    }
    if truth_model == "cw_j2":
        return PerturbedConjunctionEnv(
            **parameters,
            j2_enabled=True,
            drag_enabled=False,
            truth_model="cw_j2",
            shield_model="cw",
            allow_unvalidated_d_safe=True,
            initial_offset_range_m=(low, high),
            rho0=float(get(config, "perturbations", "rho0")),
            h0=float(get(config, "perturbations", "h0")),
            scale_height=float(get(config, "perturbations", "scale_height")),
        )
    return ContinuousConjunctionEnv(
        **parameters,
        initial_offset_range_m=(low, high),
    )


def _run_episode(
    *,
    config: dict[str, Any],
    policy,
    train_seed: int,
    eval_seed: int,
    policy_sha256: str,
    condition: str,
    d_safe: float,
    truth_model: str = "cw",
    initial_state: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = _new_environment(config, truth_model)
    rng = np.random.default_rng(eval_seed)
    reset_state = environment.reset(rng)
    if initial_state is not None and initial_state_hash(reset_state) != initial_state_hash(initial_state):
        raise AssertionError("provided initial state does not match the reset state")
    state = reset_state
    state_hash = initial_state_hash(state)
    supervisor = None
    shield = None
    if condition == "shield_on":
        mean_motion = float(get(config, "environment", "n"))
        shield = ControlBarrierShield(
            d_safe=d_safe,
            alpha1=SHIELD_ALPHA1,
            alpha2=SHIELD_ALPHA2,
            u_max=float(get(config, "environment", "u_max")),
            drift_fn=lambda current_state: relative_drift_acceleration(
                np.asarray(current_state, dtype=np.float64)[:6], mean_motion
            ),
            allow_unvalidated_d_safe=True,
        )
        supervisor = SafetySupervisor(shield=shield)

    hard_body_radius = float(get(config, "safety_geometry", "hard_body_radius_m"))
    horizon_steps = int(get(config, "environment", "horizon_steps"))
    collided = False
    minimum_distance = math.inf
    fuel = 0.0
    clipped_steps = 0
    first_infeasible_step = -1
    first_infeasible_distance: float | str = ""
    first_infeasible_speed: float | str = ""
    feasible_step_count = 0
    constraint_violation_max = 0.0
    box_violation_max = 0.0
    solver_error = ""
    episode_steps: list[dict[str, Any]] = []
    for step_index in range(1, horizon_steps + 1):
        requested_action = np.asarray(policy(state, rng), dtype=np.float64)
        if supervisor is None:
            applied_action = requested_action
            mode = "nominal"
            feasible = True
            a = np.zeros(3, dtype=np.float64)
            b = 0.0
        else:
            try:
                applied_action, mode = supervisor.act(state, requested_action)
            except Exception as error:
                solver_error = f"{type(error).__name__}: {error}"
                break
            feasible = mode != "backup"
            a, b = _barrier_terms(shield, state)
            if feasible:
                feasible_step_count += 1
                constraint_violation_max = max(
                    constraint_violation_max, float(np.dot(a, applied_action) - b)
                )
                box_violation_max = max(
                    box_violation_max,
                    float(np.max(np.abs(applied_action) - environment.u_max)),
                )
            else:
                if first_infeasible_step < 0:
                    relative_position = state[:3] - state[6:9]
                    relative_velocity = state[3:6] - state[9:12]
                    first_infeasible_step = step_index
                    first_infeasible_distance = float(np.linalg.norm(relative_position))
                    first_infeasible_speed = float(np.linalg.norm(relative_velocity))
        episode_steps.append(
            {
                "condition": condition,
                "d_safe": d_safe,
                "truth_model": truth_model,
                "label": LABEL_TEMPLATE.format(value=d_safe),
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "step": step_index,
                "mode": mode,
                "feasible": int(feasible),
                "a_dot_u": float(np.dot(a, applied_action)) if condition == "shield_on" else "",
                "b": b if condition == "shield_on" else "",
                "u_x": float(applied_action[0]),
                "u_y": float(applied_action[1]),
                "u_z": float(applied_action[2]),
            }
        )
        state, info = environment.step(applied_action)
        distance = float(info["min_distance_in_step"])
        minimum_distance = min(minimum_distance, distance)
        collided = collided or distance < hard_body_radius
        fuel += float(np.linalg.norm(np.clip(applied_action, -environment.u_max, environment.u_max)) * environment.dt)
        clipped_steps += int(bool(info["clipped"]))

    mode_counts = summarize_modes([str(step["mode"]) for step in episode_steps])
    row = {
        "condition": condition,
        "d_safe": d_safe,
        "truth_model": truth_model,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "collided": int(collided),
        "min_distance_m": minimum_distance if math.isfinite(minimum_distance) else "",
        "fuel_mps": fuel,
        "clipped_fraction": clipped_steps / horizon_steps,
        "nominal_steps": mode_counts["nominal"],
        "shielded_steps": mode_counts["shielded"],
        "backup_steps": mode_counts["backup"],
        "infeasible_steps": mode_counts["infeasible"],
        "first_infeasible_step": first_infeasible_step,
        "relative_distance_at_first_infeasible_m": first_infeasible_distance,
        "relative_speed_at_first_infeasible_mps": first_infeasible_speed,
        "initial_state_hash": state_hash,
        "policy_sha256": policy_sha256,
        "label": LABEL_TEMPLATE.format(value=d_safe),
        "shield_off_reused_across_d_safe": int(condition == "shield_off"),
        "solver_error": solver_error,
        "feasible_step_count": feasible_step_count,
        "constraint_violation_max": constraint_violation_max,
        "box_violation_max": box_violation_max,
    }
    return row, episode_steps if mode_counts["infeasible"] > 0 else []


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    output_dir: Path,
    config_path: Path,
    policy_dir: Path,
    selected: list[dict[str, str]],
    d_safe_values: list[float],
    truth_models: list[str],
    eval_count: int,
    config: dict[str, Any],
    train_seed_filter: int | None,
) -> None:
    mismatch_header = (
        "Shield drift model: CW-only. Environment truth: full nonlinear J2 "
        "acceleration. The shield operates under model mismatch in this table."
    )
    manifest = {
        "scope": "post-hoc comparison of verifier-selected sac_primary policies",
        "truth_models": truth_models,
        "d_safe_parameters_m": d_safe_values,
        "d_safe_label_template": LABEL_TEMPLATE,
        "d_safe_parameter_labels": [
            LABEL_TEMPLATE.format(value=value) for value in d_safe_values
        ],
        "evaluation_seeds": {
            "start": 10000,
            "count": eval_count,
            "end_inclusive": 10000 + eval_count - 1,
        },
        "conditions": ["shield_off", "shield_on"],
        "shield_drift_model": "CW-only in both truth-model conditions",
        "shield_gains": {"alpha1_per_s": SHIELD_ALPHA1, "alpha2_per_s": SHIELD_ALPHA2},
        "exploratory_parameter_sweep": any(
            value not in D_SAFE_VALUES for value in d_safe_values
        ),
        "shield_off_reused_across_d_safe": True,
        "protocol": {
            "policy_index": str((policy_dir / "index.csv").resolve()),
            "policy_selection": (
                "largest step per (method, train_seed), method sac_primary"
                + (
                    f", filtered to train_seed={train_seed_filter}"
                    if train_seed_filter is not None
                    else ""
                )
            ),
            "sha256_verified_before_loading": True,
            "evaluation_seeds_inclusive": [10000, 10000 + eval_count - 1],
            "shield_off_computed_once_per_policy_seed_truth_model": True,
            "post_hoc": True,
            "shield_in_the_loop_training": False,
        },
        "expected_csv_rows": (
            len(selected) * eval_count * 2 * len(d_safe_values) * len(truth_models)
        ),
        "expected_rows_per_truth_model": len(selected) * eval_count * 2 * len(d_safe_values),
        "u_max_mps2": float(get(config, "environment", "u_max")),
        "table_headers": {
            model: [
                f"{LABEL_TEMPLATE.format(value=value)}"
                + (f"\n{mismatch_header}" if model == "cw_j2" else "")
                for value in d_safe_values
            ]
            for model in truth_models
        },
        "cw_j2_table_mismatch_header": mismatch_header,
        "training": "post-hoc on selected policies; no shield-in-the-loop training",
        "selected_policies": [
            {
                "method": row["method"],
                "train_seed": int(row["train_seed"]),
                "step": int(row["step"]),
                "file": row["file"],
                "sha256": row["observed_sha256"],
            }
            for row in selected
        ],
        "config": str(config_path.resolve()),
        "policy_dir": str(policy_dir.resolve()),
        "environment_conditions": {
            "cw": {"status": "running", "truth_model": "cw", "j2_enabled": False, "drag_enabled": False},
            "cw_j2": {
                "status": "running",
                "truth_model": "cw_j2",
                "j2_enabled": True,
                "drag_enabled": False,
                "environment_acceleration": "full nonlinear J2",
                "shield_drift_model": "CW-only",
                "table_header": mismatch_header,
            },
            "CW + J2 + drag (blocked)": {
                "status": "blocked",
                "reason": DRAG_BLOCK_REASON,
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def run_comparison(
    *,
    config_path: Path,
    output_dir: Path,
    policy_dir: Path,
    d_safe_values: list[float],
    eval_count: int = 100,
    truth_models: tuple[str, ...] = ("cw",),
    train_seed_filter: int | None = None,
    allow_exploratory_d_safe: bool = False,
) -> None:
    start_time = time.monotonic()
    validate_d_safe_values(
        d_safe_values, allow_exploratory=allow_exploratory_d_safe
    )
    if eval_count < 1 or eval_count > 100:
        raise ValueError("--eval-count must be between 1 and 100")
    if not truth_models or any(model not in TRUTH_MODELS for model in truth_models):
        raise ValueError(f"--truth-model accepts only {TRUTH_MODELS}")
    if len(set(truth_models)) != len(truth_models):
        raise ValueError("--truth-model values must be unique")
    _prepare_output_dir(output_dir)
    config = load_config(config_path)
    _, all_selected = final_policy_rows(policy_dir)
    selected = filter_policy_rows(all_selected, train_seed_filter)
    _write_manifest(
        output_dir,
        config_path,
        policy_dir,
        selected,
        d_safe_values,
        list(truth_models),
        eval_count,
        config,
        train_seed_filter,
    )
    seed_start = int(get(config, "evaluation_seeds", "start"))
    all_rows: list[dict[str, Any]] = []
    completed_physical_episodes = 0
    expected_physical_episodes = (
        len(selected) * eval_count * (1 + len(d_safe_values)) * len(truth_models)
    )
    model_episode_durations: dict[str, list[float]] = {
        model: [] for model in truth_models
    }
    expected_per_model = len(selected) * eval_count * (1 + len(d_safe_values))
    with StreamingCSVWriter(
        output_dir / "shield_comparison_steps.csv", STEP_FIELDS
    ) as step_writer:
        for policy_row in selected:
            policy_path = policy_dir / policy_row["file"]
            agent = load_agent(policy_path, config)
            policy = deterministic_evaluation_policy(agent)
            train_seed = int(policy_row["train_seed"])
            for eval_seed in range(seed_start, seed_start + eval_count):
                for truth_model in truth_models:
                    env = _new_environment(config, truth_model)
                    initial_state = env.reset(np.random.default_rng(eval_seed))
                    episode_started = time.monotonic()
                    off_row, _ = _run_episode(
                        config=config,
                        policy=policy,
                        train_seed=train_seed,
                        eval_seed=eval_seed,
                        policy_sha256=policy_row["observed_sha256"],
                        condition="shield_off",
                        d_safe=d_safe_values[0],
                        truth_model=truth_model,
                        initial_state=initial_state,
                    )
                    episode_duration = time.monotonic() - episode_started
                    if completed_physical_episodes < 10:
                        model_episode_durations[truth_model].append(episode_duration)
                    completed_physical_episodes += 1
                    progress_line(
                        completed_physical_episodes,
                        expected_physical_episodes,
                        start_time,
                        model_episode_durations,
                        expected_per_model,
                    )
                    for d_safe in d_safe_values:
                        reused = dict(off_row)
                        reused["d_safe"] = d_safe
                        reused["label"] = LABEL_TEMPLATE.format(value=d_safe)
                        reused["shield_off_reused_across_d_safe"] = 1
                        all_rows.append(reused)
                        episode_started = time.monotonic()
                        on_row, on_steps = _run_episode(
                            config=config,
                            policy=policy,
                            train_seed=train_seed,
                            eval_seed=eval_seed,
                            policy_sha256=policy_row["observed_sha256"],
                            condition="shield_on",
                            d_safe=d_safe,
                            truth_model=truth_model,
                            initial_state=initial_state,
                        )
                        episode_duration = time.monotonic() - episode_started
                        if completed_physical_episodes < 10:
                            model_episode_durations[truth_model].append(episode_duration)
                        all_rows.append(on_row)
                        for step_row in on_steps:
                            step_writer.write(step_row)
                        step_writer.flush()
                        completed_physical_episodes += 1
                        progress_line(
                            completed_physical_episodes,
                            expected_physical_episodes,
                            start_time,
                            model_episode_durations,
                            expected_per_model,
                        )
    assert_initial_state_hashes(all_rows)
    _write_csv(output_dir / "shield_comparison.csv", EPISODE_FIELDS, all_rows)
    if any(row["solver_error"] for row in all_rows):
        raise RuntimeError("solver exceptions were recorded; comparison gate failed")


def progress_line(
    completed: int,
    expected: int,
    started: float,
    model_episode_durations: dict[str, list[float]],
    expected_per_model: int,
) -> None:
    elapsed = time.monotonic() - started
    print(
        f"[{elapsed:.1f}s] physical_episode={completed}/{expected} completed",
        flush=True,
    )
    if completed == min(10, expected):
        model_averages = {
            model: sum(durations) / len(durations)
            for model, durations in model_episode_durations.items()
            if durations
        }
        estimated_total = sum(
            average * expected_per_model for average in model_averages.values()
        )
        model_measurements = " ".join(
            f"{model}_samples={len(model_episode_durations[model])} "
            f"{model}_per_episode_s={model_averages.get(model, math.nan):.3f}"
            for model in sorted(model_episode_durations)
        )
        print(
            f"MEASURED first_10_physical_episodes={completed} elapsed_s={elapsed:.3f} "
            f"{model_measurements} expected_physical_episodes={expected} "
            f"expected_per_truth_model={expected_per_model} "
            f"estimated_total_s={estimated_total:.1f}",
            flush=True,
        )


def cli(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_unshielded.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--d-safe", nargs="+", type=float, required=True)
    parser.add_argument(
        "--truth-model",
        nargs="+",
        choices=TRUTH_MODELS,
        default=("cw",),
    )
    parser.add_argument("--eval-count", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("--train-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--allow-exploratory-d-safe", action="store_true", help=argparse.SUPPRESS)
    parsed = parser.parse_args(arguments)
    run_comparison(
        config_path=parsed.config,
        output_dir=parsed.output_dir,
        policy_dir=parsed.policy_dir,
        d_safe_values=parsed.d_safe,
        eval_count=parsed.eval_count,
        truth_models=tuple(parsed.truth_model),
        train_seed_filter=parsed.train_seed,
        allow_exploratory_d_safe=parsed.allow_exploratory_d_safe,
    )


if __name__ == "__main__":
    cli()
