"""Reproduce a small unshielded baseline for the sigma-miss investigation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import run_shield_comparison as shield_comparison
from benchmarks.oracle_reference import get, load_config
from benchmarks.run_unshielded_benchmark import make_environment, step_reward_and_termination


def run_baseline(
    *,
    config_path: Path,
    policy_dir: Path,
    output_dir: Path,
    seed_start: int,
    episode_count: int,
    train_seed: int,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    _, policies = shield_comparison.final_policy_rows(policy_dir)
    selected = shield_comparison.filter_policy_rows(policies, train_seed)[0]
    agent = shield_comparison.load_agent(policy_dir / selected["file"], config)
    policy = shield_comparison.deterministic_evaluation_policy(agent)
    hard_body_radius = float(get(config, "safety_geometry", "hard_body_radius_m"))
    proximity_weight = float(get(config, "reward", "proximity_weight"))
    horizon_steps = int(get(config, "environment", "horizon_steps"))
    rows: list[dict[str, object]] = []
    started = time.monotonic()

    for eval_seed in range(seed_start, seed_start + episode_count):
        rng = np.random.default_rng(eval_seed)
        environment = make_environment(config, eval_seed)
        initial_state = environment.reset(rng)
        state = initial_state.copy()
        collided = False
        minimum_distance = float("inf")
        fuel = 0.0
        clipped_steps = 0
        reward_components = {
            "fuel_term": 0.0,
            "collision_term": 0.0,
            "proximity_term": 0.0,
            "survival_term": 0.0,
        }

        for step in range(1, horizon_steps + 1):
            action = np.asarray(policy(state, rng), dtype=np.float64)
            clipped_action = np.clip(action, -environment.u_max, environment.u_max)
            state, info = environment.step(action)
            minimum_distance = min(
                minimum_distance, float(info["min_distance_in_step"])
            )
            fuel += float(np.linalg.norm(clipped_action) * environment.dt)
            clipped_steps += int(bool(info["clipped"]))
            current, done, collided = step_reward_and_termination(
                action,
                info,
                config,
                prior_collided=collided,
                at_horizon=step == horizon_steps,
                proximity_weight=proximity_weight,
                clipped_action=clipped_action,
            )
            for key in reward_components:
                reward_components[key] += current[key]
            if done:
                break

        rows.append(
            {
                "condition": "shield_off",
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "truth_model": "cw",
                "d_safe_m": "",
                "collided": int(collided),
                "collision_rate_denominator": 1,
                "min_distance_m": minimum_distance,
                "fuel_mps": fuel,
                "reward": sum(reward_components.values()),
                "fuel_reward": reward_components["fuel_term"],
                "collision_reward": reward_components["collision_term"],
                "proximity_reward": reward_components["proximity_term"],
                "survival_reward": reward_components["survival_term"],
                "episode_length_steps": step,
                "horizon_steps": horizon_steps,
                "completed_horizon": int(step == horizon_steps),
                "clipped_fraction": clipped_steps / horizon_steps,
                "initial_state_hash": shield_comparison.initial_state_hash(
                    initial_state
                ),
                "policy_sha256": selected["observed_sha256"],
            }
        )

    with (output_dir / "baseline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    metrics = {
        "collisions": sum(int(row["collided"]) for row in rows),
        "mean_min_distance_m": float(np.mean([float(row["min_distance_m"]) for row in rows])),
        "minimum_min_distance_m": min(float(row["min_distance_m"]) for row in rows),
        "mean_fuel_mps": float(np.mean([float(row["fuel_mps"]) for row in rows])),
        "mean_reward": float(np.mean([float(row["reward"]) for row in rows])),
        "completed_horizon": sum(int(row["completed_horizon"]) for row in rows),
    }
    manifest = {
        "phase": "fresh unshielded baseline reproduction",
        "config": str(config_path.resolve()),
        "config_sha256": config_hash,
        "policy": selected["file"],
        "policy_sha256": selected["observed_sha256"],
        "train_seed": train_seed,
        "evaluation_seeds": list(range(seed_start, seed_start + episode_count)),
        "episodes": episode_count,
        "horizon_steps": horizon_steps,
        "truth_model": "cw",
        "shield": "off",
        "runtime_seconds": time.monotonic() - started,
        "metrics": metrics,
    }
    (output_dir / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_unshielded.yaml"))
    parser.add_argument(
        "--policy-dir", type=Path,
        default=Path("results/rerun_primary_20261014/policies"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--episode-count", type=int, default=20)
    parser.add_argument("--train-seed", type=int, default=0)
    args = parser.parse_args()
    run_baseline(
        config_path=args.config,
        policy_dir=args.policy_dir,
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        episode_count=args.episode_count,
        train_seed=args.train_seed,
    )


if __name__ == "__main__":
    cli()