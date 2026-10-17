"""Rerun seed 4 with opt-in episode-level collision trace logging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from . import run_unshielded_benchmark as benchmark
from .run_unshielded_benchmark import get, load_config

EXPECTED_CONFIG_SHA256 = "098391214ac211765190e2c84b81f60a51ee5c8d9d5ce4cf3aa5eae5c2ec6426"
MATCH_FIELDS = ("step", "episode_length", "collided", "reward")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _consume_output_reservation(output_dir: Path) -> None:
    token = os.environ.get("AEGIS_ORBIT_OUTPUT_RESERVATION")
    reservation = output_dir / ".runner-output-reservation"
    expected_names = {
        ".runner-output-reservation",
        "launch.json",
        "pid.txt",
        "run.log",
        "run.log.err",
    }
    permitted_names = expected_names | {
        "launcher_attempt1.err",
        "launch_attempt1.json",
        "pid_attempt1.txt",
    }
    if not output_dir.is_dir() or token is None or not reservation.is_file():
        raise RuntimeError(f"Missing output reservation: {output_dir}")
    if reservation.read_text(encoding="utf-8").lstrip("\ufeff") != token:
        raise RuntimeError("Output reservation token does not match")
    if {entry.name for entry in output_dir.iterdir()} not in (
        expected_names,
        permitted_names,
    ):
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    if not all((output_dir / name).is_file() for name in expected_names):
        raise RuntimeError("Output reservation is missing launcher files")
    reservation.unlink()


def _read_primary_seed_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["method"] == "sac_primary" and row["seed"] == "4"
        ]


def _window_counts(rows: list[dict[str, str]], step_field: str) -> list[dict]:
    windows = []
    for lower in range(0, 200_000, 10_000):
        selected = [
            row for row in rows
            if lower < int(row[step_field]) <= lower + 10_000
        ]
        windows.append({
            "lower_step_exclusive": lower,
            "upper_step_inclusive": lower + 10_000,
            "episodes": len(selected),
            "collisions": sum(int(row["collided"]) for row in selected),
        })
    return windows


def _mismatch_report(
    original_rows: list[dict[str, str]],
    rerun_rows: list[dict[str, str]],
) -> list[str]:
    problems = []
    if len(original_rows) != len(rerun_rows):
        problems.append(
            f"episode row count differs: original={len(original_rows)} rerun={len(rerun_rows)}"
        )
    for index, (original, rerun) in enumerate(zip(original_rows, rerun_rows), start=1):
        for field in MATCH_FIELDS:
            if original[field] != rerun[field]:
                problems.append(
                    f"row={index} field={field} original={original[field]} rerun={rerun[field]}"
                )
                if len(problems) >= 25:
                    return problems
    return problems


def run(config_path: Path, original_log_path: Path, output_dir: Path) -> None:
    raw_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if raw_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"Config SHA-256 mismatch: expected={EXPECTED_CONFIG_SHA256} actual={raw_hash}"
        )
    config = load_config(config_path)
    if 4 not in [int(seed) for seed in get(config, "seeds")]:
        raise RuntimeError("Seed 4 is not listed in the requested config")
    proximity_weight = float(get(config, "reward", "proximity_weight"))
    if proximity_weight != 0.0:
        raise RuntimeError(f"Expected sac_primary proximity weight 0.0, got {proximity_weight}")

    _consume_output_reservation(output_dir)
    benchmark._RUN_STARTED_AT = time.monotonic()
    logs: list[dict] = []
    stability_logs: list[dict] = []
    reward_logs: list[dict] = []
    selection_logs: list[dict] = []
    episode_diagnostics: list[dict] = []
    benchmark.train(
        4,
        config,
        proximity_weight,
        logs=logs,
        stability_logs=stability_logs,
        reward_logs=reward_logs,
        selection_logs=selection_logs,
        selection_path=output_dir / "checkpoint_selection.csv",
        policy_output_dir=output_dir / "policies",
        method_name="sac_primary",
        episode_diagnostic_logs=episode_diagnostics,
    )

    _write_csv(output_dir / "training_log.csv", reward_logs)
    _write_csv(output_dir / "episode_diagnostics.csv", episode_diagnostics)

    original_rows = _read_primary_seed_rows(original_log_path)
    rerun_path = output_dir / "training_log.csv"
    rerun_rows = _read_primary_seed_rows(rerun_path)
    row_mismatches = _mismatch_report(original_rows, rerun_rows)
    if row_mismatches:
        (output_dir / "validation_failure.json").write_text(
            json.dumps({
                "status": "FAIL",
                "phase": "row_by_row_training_log_comparison",
                "config_sha256": raw_hash,
                "mismatches": row_mismatches,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        for item in row_mismatches:
            print(f"VALIDATION FAIL {item}", flush=True)
        raise RuntimeError("Training log differs from the original; window analysis stopped")

    original_windows = _window_counts(original_rows, "step")
    rerun_windows = _window_counts(rerun_rows, "step")
    window_mismatches = []
    for original, rerun in zip(original_windows, rerun_windows):
        if original != rerun:
            window_mismatches.append({
                "original": original,
                "rerun": rerun,
            })
    seed4_mid_collisions = sum(
        int(row["collided"])
        for row in rerun_rows
        if 30_000 < int(row["step"]) <= 60_000
    )
    if seed4_mid_collisions != 240:
        window_mismatches.append({
            "expected_collisions_in_30000_60000": 240,
            "actual_collisions_in_30000_60000": seed4_mid_collisions,
        })
    if window_mismatches:
        (output_dir / "validation_failure.json").write_text(
            json.dumps({
                "status": "FAIL",
                "phase": "endpoint_binned_window_comparison",
                "config_sha256": raw_hash,
                "mismatches": window_mismatches,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        for item in window_mismatches:
            print(f"VALIDATION FAIL window={item}", flush=True)
        raise RuntimeError("Endpoint-binned window counts differ; analysis stopped")

    _write_csv(output_dir / "window_count_verification.csv", [
        {
            **original,
            "rerun_episodes": rerun["episodes"],
            "rerun_collisions": rerun["collisions"],
            "matches_original": True,
        }
        for original, rerun in zip(original_windows, rerun_windows)
    ])
    episode_by_step = {str(row["step"]): row for row in episode_diagnostics}
    if len(episode_by_step) != len(episode_diagnostics):
        raise RuntimeError("Episode diagnostic rows have duplicate end steps")
    missing_diagnostics = [
        str(row["step"]) for row in rerun_rows
        if str(row["step"]) not in episode_by_step
    ]
    if missing_diagnostics:
        raise RuntimeError(
            f"Missing per-episode diagnostic rows at steps {missing_diagnostics[:10]}"
        )

    focus_rows = [
        row for row in episode_diagnostics
        if 30_000 < int(row["step"]) <= 60_000
    ]
    offsets = [float(row["initial_offset_m"]) for row in focus_rows]
    offset_bands = []
    for band_index in range(5):
        lower = 67.0 + band_index * 0.1
        upper = lower + 0.1
        band_rows = [
            row for row in focus_rows
            if lower <= float(row["initial_offset_m"]) < upper
            or (band_index == 4 and float(row["initial_offset_m"]) == upper)
        ]
        offset_bands.append({
            "offset_lower_m": f"{lower:.1f}",
            "offset_upper_m": f"{upper:.1f}",
            "episodes": len(band_rows),
            "collisions": sum(int(row["collided"]) for row in band_rows),
        })
    _write_csv(output_dir / "seed4_30k_60k_offset_bands.csv", offset_bands)

    collision_rows = [
        row for row in episode_diagnostics if int(row["collided"]) == 1
    ]
    if any(
        row["collision_substep_index_zero_based"] is None
        or row["collision_substep_time_s"] is None
        for row in collision_rows
    ):
        raise RuntimeError("A collided episode is missing sampled collision timing")
    episode_times = [float(row["episode_to_terminal_s"]) for row in episode_diagnostics]
    validation = {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": raw_hash,
        "seed": 4,
        "training_log_row_count_original": len(original_rows),
        "training_log_row_count_rerun": len(rerun_rows),
        "matched_fields": list(MATCH_FIELDS),
        "row_by_row_match": True,
        "all_endpoint_binned_10k_windows_match": True,
        "collisions_in_30000_60000_endpoint_bins": seed4_mid_collisions,
        "episode_diagnostics_row_count": len(episode_diagnostics),
        "collided_episode_count": len(collision_rows),
        "mean_episode_to_terminal_seconds": statistics.mean(episode_times),
        "median_episode_to_terminal_seconds": statistics.median(episode_times),
        "timing_note": "Wall time from episode reset to terminal detection; it includes updates on preceding steps but stops before the optimizer update for the terminal transition.",
        "collision_sampling_note": "Sub-step index is zero-based among the 21 existing equally spaced distance samples per control step; timestamp is the first sampled time with distance strictly below the hard-body radius, not a continuous-time root.",
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"VALIDATION PASS rows={len(rerun_rows)} "
        f"mid_window_collisions={seed4_mid_collisions} "
        f"mean_episode_to_terminal_s={statistics.mean(episode_times):.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-training-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.original_training_log, args.output_dir)


if __name__ == "__main__":
    main()
