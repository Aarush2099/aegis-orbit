"""Verify saved policy hashes and replay the final selected policy per seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from aegis_orbit.agents import SACAgent
from .oracle_reference import get, load_config
from .run_unshielded_benchmark import (
    deterministic_evaluation_policy,
    make_environment,
    run_episode,
)


def _collision_flag(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    raise ValueError(f"invalid collided value: {value!r}")


def _load_agent(actor_path: Path, config: dict) -> SACAgent:
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


def verify_policies(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Check every indexed file hash and replay each seed's final selected actor."""
    config = load_config(config_path)
    index_path = output_dir / "policies" / "index.csv"
    benchmark_path = output_dir / "unshielded_benchmark.csv"
    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    with benchmark_path.open(newline="", encoding="utf-8") as handle:
        benchmark_rows = list(csv.DictReader(handle))

    failures: list[str] = []
    policy_dir = index_path.parent
    indexed_files_hashed = 0
    hash_mismatches = 0
    verified_rows: set[tuple[str, int, int, str]] = set()
    latest_rows: dict[tuple[str, int], dict[str, str]] = {}
    for row in index_rows:
        policy_path = policy_dir / row["file"]
        if not policy_path.is_file():
            hash_mismatches += 1
            failures.append(f"missing policy file: {row['file']}")
            continue
        digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        indexed_files_hashed += 1
        expected_digest = row.get("sha256", "").strip()
        if not expected_digest or digest != expected_digest:
            hash_mismatches += 1
            failures.append(
                f"sha256 mismatch: {row['file']} expected={expected_digest!r} observed={digest}"
            )
        else:
            verified_rows.add((
                row["method"],
                int(row["train_seed"]),
                int(row["step"]),
                row["file"],
            ))
        key = (row["method"], int(row["train_seed"]))
        previous = latest_rows.get(key)
        if previous is None or int(row["step"]) > int(previous["step"]):
            latest_rows[key] = row

    if indexed_files_hashed < len(index_rows):
        failures.append(
            f"hashed {indexed_files_hashed} indexed files; expected {len(index_rows)}"
        )

    max_difference = 0.0
    comparisons = 0
    final_policies_reevaluated = 0
    evaluation_start = int(get(config, "evaluation_seeds", "start"))
    evaluation_count = int(get(config, "evaluation_seeds", "count"))
    hard_body_radius = float(get(config, "safety_geometry", "hard_body_radius_m"))
    expected_rows = {
        (row["method"], int(row["train_seed"]), int(row["eval_seed"])): row
        for row in benchmark_rows
        if row["method"] != "oracle_avoidable_subset"
    }

    for (method, train_seed), selected_row in sorted(latest_rows.items()):
        selected_key = (
            method,
            train_seed,
            int(selected_row["step"]),
            selected_row["file"],
        )
        if selected_key not in verified_rows:
            failures.append(
                f"final selected policy failed hash verification: {selected_row['file']}"
            )
            continue
        actor_path = policy_dir / selected_row["file"]
        agent = _load_agent(actor_path, config)
        policy = deterministic_evaluation_policy(agent)
        final_policies_reevaluated += 1
        for eval_seed in range(evaluation_start, evaluation_start + evaluation_count):
            key = (method, train_seed, eval_seed)
            expected = expected_rows.get(key)
            if expected is None:
                failures.append(f"missing benchmark episode row: {key}")
                continue
            observed = run_episode(
                make_environment(config, eval_seed),
                policy,
                np.random.default_rng(eval_seed),
                hard_body_radius,
                config,
                float(get(config, "reward", "proximity_weight"))
                if method == "sac_primary"
                else 0.0,
            )
            comparisons += 1
            if _collision_flag(expected["collided"]) != bool(observed["collided"]):
                failures.append(
                    f"collision mismatch method={method} train_seed={train_seed} "
                    f"eval_seed={eval_seed}"
                )
            for metric in ("min_distance_m", "fuel_mps"):
                difference = abs(float(expected[metric]) - float(observed[metric]))
                max_difference = max(max_difference, difference)
                if not math.isclose(
                    float(expected[metric]),
                    float(observed[metric]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    failures.append(
                        f"{metric} mismatch method={method} train_seed={train_seed} "
                        f"eval_seed={eval_seed} expected={expected[metric]} "
                        f"observed={observed[metric]}"
                    )

    if not index_rows:
        failures.append("policy index contains no rows")
    if hash_mismatches:
        failures.append(f"{hash_mismatches} indexed policy hash checks failed")
    return {
        "passed": not failures,
        "failures": failures,
        "indexed_files_hashed": indexed_files_hashed,
        "hash_mismatches": hash_mismatches,
        "final_policies_reevaluated": final_policies_reevaluated,
        "final_policy_episodes": comparisons,
        "maximum_difference": max_difference,
    }


def cli(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_unshielded.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    result = verify_policies(parsed.config, parsed.output_dir)
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"{status} indexed_files_hashed={result['indexed_files_hashed']} "
        f"hash_mismatches={result['hash_mismatches']} "
        f"final_policies_reevaluated={result['final_policies_reevaluated']} "
        f"final_policy_episodes={result['final_policy_episodes']} "
        f"maximum_difference={result['maximum_difference']:.12g}"
    )
    if result["passed"]:
        return
    for failure in result["failures"]:
        print(f"FAIL {failure}")
    raise SystemExit(1)


if __name__ == "__main__":
    cli()
