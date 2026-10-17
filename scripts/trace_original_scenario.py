"""Record full CBF terms along one verified original SAC encounter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis_orbit.envs.cw_relative_dynamics import relative_drift_acceleration
from aegis_orbit.shields import ControlBarrierShield, SafetySupervisor
from benchmarks import run_shield_comparison as comparison
from benchmarks.oracle_reference import get, load_config

TRACE_FIELDS = (
    "eval_seed", "step", "mode", "relative_r_x_m", "relative_r_y_m",
    "relative_r_z_m", "relative_v_x_mps", "relative_v_y_mps",
    "relative_v_z_mps", "drift_x_mps2", "drift_y_mps2", "drift_z_mps2",
    "d_safe_m", "h_m2", "h_dot_m2_per_s", "psi_m2_per_s",
    "two_v_squared_m2_per_s2", "two_r_dot_f_m2_per_s2",
    "alpha1_h_dot_m2_per_s2", "psi_dot_without_control_m2_per_s2",
    "alpha2_psi_m2_per_s2", "psi_dot_applied_m2_per_s2",
    "hocbf_lhs_applied_m2_per_s2",
    "a_x_m", "a_y_m", "a_z_m", "b_m2_per_s2",
    "min_box_a_dot_u_m2_per_s2", "max_box_a_dot_u_m2_per_s2",
    "nominal_a_dot_u_m2_per_s2", "applied_a_dot_u_m2_per_s2",
    "constraint_residual_m2_per_s2", "u_max_mps2_per_axis",
    "u_nom_x_mps2", "u_nom_y_mps2", "u_nom_z_mps2",
    "u_applied_x_mps2", "u_applied_y_mps2", "u_applied_z_mps2",
    "feasible", "projection_applied", "backup_used", "initial_h_m2",
    "initial_psi_m2_per_s", "initial_hocbf_admissible", "min_distance_in_step_m",
    "collision_this_step", "collision_so_far",
)


def trace_episode(output_dir: Path, eval_seed: int, d_safe: float, train_seed: int) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path("configs/sac_unshielded.yaml")
    config = load_config(config_path)
    policy_dir = Path("results/rerun_primary_20261014/policies")
    _, final_rows = comparison.final_policy_rows(policy_dir)
    policy_row = comparison.filter_policy_rows(final_rows, train_seed)[0]
    agent = comparison.load_agent(policy_dir / policy_row["file"], config)
    policy = comparison.deterministic_evaluation_policy(agent)
    environment = comparison._new_environment(config, "cw")
    state = environment.reset(np.random.default_rng(eval_seed))
    initial_state = state.copy()

    u_max = float(get(config, "environment", "u_max"))
    n = float(get(config, "environment", "n"))
    alpha1 = comparison.SHIELD_ALPHA1
    alpha2 = comparison.SHIELD_ALPHA2
    shield = ControlBarrierShield(
        d_safe=d_safe,
        alpha1=alpha1,
        alpha2=alpha2,
        u_max=u_max,
        drift_fn=lambda current: relative_drift_acceleration(
            np.asarray(current, dtype=np.float64)[:6], n
        ),
        allow_unvalidated_d_safe=True,
    )
    supervisor = SafetySupervisor(shield)
    hard_body_radius = float(get(config, "safety_geometry", "hard_body_radius_m"))

    initial_r = initial_state[:3] - initial_state[6:9]
    initial_v = initial_state[3:6] - initial_state[9:12]
    initial_h = shield.barrier(initial_r)
    initial_h_dot = 2.0 * float(np.dot(initial_r, initial_v))
    initial_psi = initial_h_dot + alpha1 * initial_h
    initial_admissible = bool(initial_h >= 0.0 and initial_psi >= 0.0)

    rows: list[dict[str, Any]] = []
    first_infeasible_snapshot: dict[str, Any] | None = None
    collided = False
    minimum_distance = math.inf
    for step in range(1, int(get(config, "environment", "horizon_steps")) + 1):
        relative_position = state[:3] - state[6:9]
        relative_velocity = state[3:6] - state[9:12]
        drift = relative_drift_acceleration(state[:6], n)
        h = shield.barrier(relative_position)
        h_dot = 2.0 * float(np.dot(relative_position, relative_velocity))
        psi = h_dot + alpha1 * h
        psi_dot_without_control = (
            2.0 * float(np.dot(relative_velocity, relative_velocity))
            + 2.0 * float(np.dot(relative_position, drift)) + alpha1 * h_dot
        )
        two_v_squared = 2.0 * float(np.dot(relative_velocity, relative_velocity))
        two_r_dot_f = 2.0 * float(np.dot(relative_position, drift))
        alpha1_h_dot = alpha1 * h_dot
        alpha2_psi = alpha2 * psi
        b = psi_dot_without_control + alpha2_psi
        a = -2.0 * relative_position
        min_box = -u_max * float(np.sum(np.abs(a)))
        max_box = -min_box
        nominal = np.asarray(policy(state, np.random.default_rng(eval_seed)), dtype=np.float64)
        applied, mode = supervisor.act(state, nominal)
        nominal_dot = float(np.dot(a, np.clip(nominal, -u_max, u_max)))
        applied_dot = float(np.dot(a, applied))
        residual = applied_dot - b
        psi_dot_applied = psi_dot_without_control + 2.0 * float(np.dot(relative_position, applied))
        next_state, info = environment.step(applied, collision_radius_m=hard_body_radius)
        distance = float(info["min_distance_in_step"])
        collision_this_step = distance < hard_body_radius
        collided = collided or collision_this_step
        minimum_distance = min(minimum_distance, distance)
        row = {
            "eval_seed": eval_seed,
            "step": step,
            "mode": mode,
            "relative_r_x_m": relative_position[0],
            "relative_r_y_m": relative_position[1],
            "relative_r_z_m": relative_position[2],
            "relative_v_x_mps": relative_velocity[0],
            "relative_v_y_mps": relative_velocity[1],
            "relative_v_z_mps": relative_velocity[2],
            "drift_x_mps2": drift[0],
            "drift_y_mps2": drift[1],
            "drift_z_mps2": drift[2],
            "d_safe_m": d_safe,
            "h_m2": h,
            "h_dot_m2_per_s": h_dot,
            "psi_m2_per_s": psi,
            "two_v_squared_m2_per_s2": two_v_squared,
            "two_r_dot_f_m2_per_s2": two_r_dot_f,
            "alpha1_h_dot_m2_per_s2": alpha1_h_dot,
            "psi_dot_without_control_m2_per_s2": psi_dot_without_control,
            "alpha2_psi_m2_per_s2": alpha2_psi,
            "psi_dot_applied_m2_per_s2": psi_dot_applied,
            "hocbf_lhs_applied_m2_per_s2": psi_dot_applied + alpha2_psi,
            "a_x_m": a[0], "a_y_m": a[1], "a_z_m": a[2],
            "b_m2_per_s2": b,
            "min_box_a_dot_u_m2_per_s2": min_box,
            "max_box_a_dot_u_m2_per_s2": max_box,
            "nominal_a_dot_u_m2_per_s2": nominal_dot,
            "applied_a_dot_u_m2_per_s2": applied_dot,
            "constraint_residual_m2_per_s2": residual,
            "u_max_mps2_per_axis": u_max,
            "u_nom_x_mps2": nominal[0], "u_nom_y_mps2": nominal[1],
            "u_nom_z_mps2": nominal[2],
            "u_applied_x_mps2": applied[0], "u_applied_y_mps2": applied[1],
            "u_applied_z_mps2": applied[2],
            "feasible": int(mode != "backup"),
            "projection_applied": int(mode == "shielded"),
            "backup_used": int(mode == "backup"),
            "initial_h_m2": initial_h,
            "initial_psi_m2_per_s": initial_psi,
            "initial_hocbf_admissible": int(initial_admissible),
            "min_distance_in_step_m": distance,
            "collision_this_step": int(collision_this_step),
            "collision_so_far": int(collided),
        }
        rows.append(row)
        if mode == "backup" and first_infeasible_snapshot is None:
            first_infeasible_snapshot = dict(row)
        state = next_state

    with (output_dir / "step_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    selected = [rows[0]]
    if first_infeasible_snapshot is not None:
        selected.append(first_infeasible_snapshot)
    with (output_dir / "key_states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "scope": "full numerical replay of one original SAC/CW shield episode",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "policy": policy_row["file"],
        "policy_sha256": policy_row["observed_sha256"],
        "train_seed": train_seed,
        "evaluation_seed": eval_seed,
        "truth_model": "cw",
        "shield_d_safe_parameter_m": d_safe,
        "alpha1_per_s": alpha1,
        "alpha2_per_s": alpha2,
        "initial_state": initial_state.tolist(),
        "initial_h_m2": initial_h,
        "initial_h_dot_m2_per_s": initial_h_dot,
        "initial_psi_m2_per_s": initial_psi,
        "initial_hocbf_admissible": initial_admissible,
        "first_infeasible_step": first_infeasible_snapshot["step"] if first_infeasible_snapshot else None,
        "episode_steps": len(rows),
        "collided": int(collided),
        "minimum_distance_m": minimum_distance,
        "solver_errors": 0,
        "step_count": len(rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if first_infeasible_snapshot:
        print("FIRST_INFEASIBLE", json.dumps(first_infeasible_snapshot, indent=2))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--d-safe", type=float, default=40.0)
    args = parser.parse_args()
    trace_episode(args.output_dir, args.eval_seed, args.d_safe, args.train_seed)


if __name__ == "__main__":
    cli()
