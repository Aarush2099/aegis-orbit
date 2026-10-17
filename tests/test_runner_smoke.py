import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.oracle_reference import (
    evaluate_encounter,
    get,
    summarize_oracle_avoidable_subset,
)
import benchmarks.run_unshielded_benchmark as benchmark
from benchmarks.check_gates import check_gates


def _smoke_config(tmp_path: Path) -> Path:
    source = Path("configs/sac_unshielded.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["training"]["total_env_steps"] = 500
    config["training"]["validation_interval_steps"] = 250
    config["training"]["warmup_steps"] = 64
    config["training"]["batch_size"] = 64
    config["seeds"] = [0]
    config["evaluation_seeds"]["count"] = 2
    config["validation_seeds"]["count"] = 2
    path = tmp_path / "smoke_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.slow
def test_runner_writes_all_artifacts_only_to_configured_output(monkeypatch, tmp_path, capsys):
    config_path = _smoke_config(tmp_path)
    output_dir = tmp_path / "artifacts"
    repository_results = Path("results")
    benchmark_csv = repository_results / "unshielded_benchmark.csv"
    benchmark_csv_before = benchmark_csv.exists()
    results_before = {
        path.relative_to(repository_results): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in repository_results.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        benchmark,
        "oracle_preflight",
        lambda config: {
            "passive_collision_count": 69,
            "oracle_avoidable_count": 69,
            "minimum_margin_m": 0.426314,
        },
    )

    benchmark.cli([
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--methods",
        "sac_primary",
    ])

    assert benchmark_csv.exists() == benchmark_csv_before
    results_after = {
        path.relative_to(repository_results): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in repository_results.rglob("*")
        if path.is_file()
    }
    assert results_after == results_before
    expected = {
        "unshielded_benchmark.csv",
        "training_log.csv",
        "stability_log.csv",
        "checkpoint_selection.csv",
        "run_manifest.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    assert (output_dir / "policies" / "index.csv").is_file()
    manifest = yaml.safe_load((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["methods"] == ["sac_primary"]
    with (output_dir / "policies" / "index.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert manifest["saved_policy_count"] == len(list(csv.DictReader(handle)))
    checks = check_gates(config_path, output_dir)
    assert len(checks) == 6
    assert all(passed for _, passed, _ in checks)
    output = capsys.readouterr().out
    assert "Artifact-integrity checks (not the original Step 2C gate definitions)" in output
    assert "oracle_avoidable_subset,label=oracle finite-search ceiling, not a policy result" in output
    with (output_dir / "unshielded_benchmark.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        oracle_rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == "oracle_avoidable_subset"
        ]
    assert len(oracle_rows) == 1
    assert oracle_rows[0]["method_label"] == "oracle finite-search ceiling, not a policy result"
    with (output_dir / "unshielded_benchmark.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        passive_rows = [
            row for row in csv.DictReader(handle) if row["method"] == "passive"
        ]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hard_radius = float(get(config, "safety_geometry", "hard_body_radius_m"))
    oracle_results = []
    for row in passive_rows:
        if row["collided"].lower() in {"true", "1"}:
            seed = int(row["eval_seed"])
            initial_state = benchmark.make_environment(config, seed).reset(
                np.random.default_rng(seed)
            )
            oracle_results.append(
                evaluate_encounter(seed, config, hard_radius, initial_state=initial_state)
            )
        else:
            oracle_results.append((False, 0.0))
    expected_summary = summarize_oracle_avoidable_subset(
        [row["collided"].lower() in {"true", "1"} for row in passive_rows],
        oracle_results,
    )
    assert int(oracle_rows[0]["oracle_passive_collision_count"]) == expected_summary[
        "passive_collision_count"
    ]
    assert int(oracle_rows[0]["oracle_avoidable_count"]) == expected_summary[
        "oracle_avoidable_count"
    ]
    assert float(oracle_rows[0]["oracle_avoidable_fraction"]) == expected_summary[
        "avoidable_fraction"
    ]
    assert float(oracle_rows[0]["oracle_mean_best_min_distance_m"]) == expected_summary[
        "mean_best_min_distance_m"
    ]


def test_gate_checker_exits_nonzero_for_missing_artifacts(tmp_path):
    config_path = Path("configs/sac_unshielded.yaml").resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.check_gates",
            "--config",
            str(config_path),
            "--results-dir",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Artifact-integrity checks (not the original Step 2C gate definitions)" in result.stdout
