"""Validate Step 2C benchmark artifact integrity before reporting metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .oracle_reference import get, load_config
from .run_unshielded_benchmark import config_hash


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if not path.is_file():
        return [], f"missing {path.name}"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), None


def _number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def check_gates(config_path: Path, results_dir: Path) -> list[tuple[str, bool, str]]:
    """Run six artifact-integrity gates; return named outcomes and evidence."""
    config = load_config(config_path)
    benchmark_path = results_dir / "unshielded_benchmark.csv"
    training_path = results_dir / "training_log.csv"
    stability_path = results_dir / "stability_log.csv"
    selection_path = results_dir / "checkpoint_selection.csv"
    manifest_path = results_dir / "run_manifest.json"

    benchmark, benchmark_error = _read_csv(benchmark_path)
    training, training_error = _read_csv(training_path)
    stability, stability_error = _read_csv(stability_path)
    selections, selection_error = _read_csv(selection_path)
    manifest = None
    manifest_error = None
    if not manifest_path.is_file():
        manifest_error = "missing run_manifest.json"
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

    required_errors = [
        error for error in (
            benchmark_error, training_error, stability_error, selection_error, manifest_error
        ) if error is not None
    ]
    results: list[tuple[str, bool, str]] = []
    results.append((
        "required_artifacts",
        not required_errors,
        "all five benchmark artifacts exist" if not required_errors else "; ".join(required_errors),
    ))

    evaluation_start = int(get(config, "evaluation_seeds", "start"))
    evaluation_count = int(get(config, "evaluation_seeds", "count"))
    train_seeds = [int(seed) for seed in get(config, "seeds")]
    primary_weight = float(get(config, "reward", "proximity_weight"))
    configured_methods = {}
    for value in get(config, "ablations", "proximity_weight_values"):
        proximity = float(value)
        name = (
            "sac_primary"
            if proximity == primary_weight
            else f"sac_ablation_proximity_{proximity:.2f}".replace(".00", "")
        )
        configured_methods[name] = proximity
    requested_methods = manifest.get("methods") if manifest is not None else None
    if requested_methods is None:
        selected_methods = list(configured_methods)
    elif isinstance(requested_methods, list):
        selected_methods = requested_methods
    else:
        selected_methods = []
    selected_proximity_values = [
        configured_methods[method]
        for method in selected_methods
        if method in configured_methods
    ]
    expected_methods = {"passive": evaluation_count, "random": evaluation_count}
    for method in selected_methods:
        if method in configured_methods:
            expected_methods[method] = len(train_seeds) * evaluation_count
    observed_methods: dict[str, int] = {}
    observed_keys: set[tuple[str, int, int]] = set()
    oracle_rows = []
    for row in benchmark:
        method = row.get("method", "")
        if method == "oracle_avoidable_subset":
            oracle_rows.append(row)
            continue
        observed_methods[method] = observed_methods.get(method, 0) + 1
        observed_keys.add((method, int(row["train_seed"]), int(row["eval_seed"])))
    expected_keys = {
        (method, train_seed, eval_seed)
        for method in expected_methods
        for train_seed in ([-1] if method in {"passive", "random"} else train_seeds)
        for eval_seed in range(evaluation_start, evaluation_start + evaluation_count)
    }
    oracle_summary_ok = (
        len(oracle_rows) == 1
        and oracle_rows[0].get("method_label")
        == "oracle finite-search ceiling, not a policy result"
    )
    coverage_ok = (
        len(selected_methods) == len(set(selected_methods))
        and set(selected_methods) <= set(configured_methods)
        and observed_methods == expected_methods
        and observed_keys == expected_keys
        and oracle_summary_ok
    )
    results.append((
        "evaluation_coverage",
        coverage_ok,
        f"observed={observed_methods}; expected={expected_methods}; "
        f"oracle_summary_rows={len(oracle_rows)}",
    ))

    expected_hash = config_hash(config)
    hashes_ok = bool(benchmark) and all(row.get("config_hash") == expected_hash for row in benchmark)
    manifest_hash_ok = manifest is not None and manifest.get("config_hash") == expected_hash
    results.append((
        "config_identity",
        hashes_ok and manifest_hash_ok,
        f"expected config_hash={expected_hash}; manifest_match={manifest_hash_ok}",
    ))

    metric_error = None
    try:
        for row in benchmark:
            if row.get("method") == "oracle_avoidable_subset":
                passive_count = int(row["oracle_passive_collision_count"])
                avoidable_count = int(row["oracle_avoidable_count"])
                fraction = _number(row, "oracle_avoidable_fraction")
                mean_best_distance = _number(row, "oracle_mean_best_min_distance_m")
                if (
                    passive_count < 0
                    or not 0 <= avoidable_count <= passive_count
                    or not 0.0 <= fraction <= 1.0
                    or mean_best_distance < 0.0
                ):
                    raise ValueError("oracle subset summary is outside its valid range")
                if passive_count and not math.isclose(
                    fraction,
                    avoidable_count / passive_count,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("oracle avoidable fraction does not match its counts")
                continue
            collision_value = row["collided"].lower()
            if collision_value not in {"true", "false", "0", "1"}:
                raise ValueError("collision metric is not a boolean")
            collided = float(collision_value in {"true", "1"})
            minimum = _number(row, "min_distance_m")
            fuel = _number(row, "fuel_mps")
            clipped = _number(row, "clipped_fraction")
            if minimum < 0.0 or fuel < 0.0 or not 0.0 <= clipped <= 1.0:
                raise ValueError("collision, distance, fuel, or clipping metric is outside its valid range")
    except (KeyError, ValueError) as error:
        metric_error = str(error)
    results.append((
        "physical_metrics",
        bool(benchmark) and metric_error is None,
        metric_error or f"validated {len(benchmark)} benchmark rows",
    ))

    checkpoint_error = None
    try:
        interval = int(get(config, "training", "validation_interval_steps"))
        total_steps = int(get(config, "training", "total_env_steps"))
        expected_rows = (
            (total_steps // interval)
            * len(train_seeds)
            * len(selected_proximity_values)
        )
        if len(selections) != expected_rows or expected_rows == 0:
            raise ValueError(f"found {len(selections)} checkpoint rows; expected {expected_rows}")
        groups: dict[tuple[int, float], list[dict[str, str]]] = {}
        for row in selections:
            key = (int(row["seed"]), float(row["proximity_weight"]))
            groups.setdefault(key, []).append(row)
        expected_groups = {
            (seed, value)
            for seed in train_seeds
            for value in selected_proximity_values
        }
        if set(groups) != expected_groups:
            raise ValueError("checkpoint seed/proximity groups do not match config")
        for group in groups.values():
            best_score = (math.inf, math.inf)
            seen_steps: set[int] = set()
            for row in sorted(group, key=lambda item: int(item["step"])):
                step = int(row["step"])
                if step in seen_steps:
                    raise ValueError("checkpoint group contains duplicate validation steps")
                seen_steps.add(step)
                score = (_number(row, "collision_rate"), _number(row, "mean_fuel"))
                selected = row["selected"].lower() == "true"
                should_select = score < best_score
                if selected != should_select:
                    raise ValueError("checkpoint selected flag does not match best-so-far collision/fuel score")
                if selected:
                    best_score = score
    except (KeyError, ValueError, ZeroDivisionError) as error:
        checkpoint_error = str(error)
    results.append((
        "checkpoint_selection",
        bool(selections) and checkpoint_error is None,
        checkpoint_error or f"validated {len(selections)} checkpoint decisions",
    ))

    log_error = None
    try:
        horizon_steps = int(get(config, "environment", "horizon_steps"))
        total_steps = int(get(config, "training", "total_env_steps"))
        if not training and total_steps >= horizon_steps:
            raise ValueError("training_log.csv contains no completed episodes")
        for row in training:
            for key in ("fuel_term", "collision_term", "proximity_term", "survival_term", "reward"):
                _number(row, key)
            if row.get("collided") not in ("0", "1"):
                raise ValueError("training log contains an invalid collision flag")
        if not stability:
            raise ValueError("stability_log.csv contains no optimizer updates")
        for row in stability:
            for key in ("actor_loss", "critic_loss", "q_min", "q_max", "alpha", "entropy"):
                _number(row, key)
            if row.get("nan") != "0":
                raise ValueError("stability log contains a NaN flag")
    except (KeyError, ValueError) as error:
        log_error = str(error)
    results.append((
        "training_stability",
        log_error is None,
        log_error or f"validated {len(training)} episodes and {len(stability)} updates",
    ))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_unshielded.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    arguments = parser.parse_args()

    outcomes = check_gates(arguments.config, arguments.results_dir)
    print("Artifact-integrity checks (not the original Step 2C gate definitions)")
    failed = False
    for name, passed, evidence in outcomes:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {evidence}")
        failed = failed or not passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
