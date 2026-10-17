import csv
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.check_shield_gates import check_gates
from benchmarks.run_shield_comparison import (
    EXPLORATORY_D_SAFE_VALUES,
    EPISODE_FIELDS,
    assert_initial_state_hashes,
    filter_policy_rows,
    run_comparison,
    summarize_modes,
    validate_d_safe_values,
)


def _paired_rows(off_hash="same", on_hash="same"):
    common = {
        "train_seed": 1,
        "eval_seed": 10000,
        "truth_model": "cw",
        "d_safe": 40.0,
    }
    return [
        {**common, "condition": "shield_off", "initial_state_hash": off_hash},
        {**common, "condition": "shield_on", "initial_state_hash": on_hash},
    ]


def test_episode_mode_counts_include_one_feasible_and_one_infeasible_step():
    episodes = [
        summarize_modes(["nominal", "shielded"]),
        summarize_modes(["nominal", "backup"]),
    ]

    assert episodes[0]["nominal"] == 1
    assert episodes[0]["shielded"] == 1
    assert episodes[0]["backup"] == episodes[0]["infeasible"] == 0
    assert episodes[1]["nominal"] == 1
    assert episodes[1]["backup"] == episodes[1]["infeasible"] == 1


def test_initial_state_hash_pair_matches_and_mismatch_fails():
    assert_initial_state_hashes(_paired_rows())

    with pytest.raises(AssertionError, match="initial-state hash mismatch"):
        assert_initial_state_hashes(_paired_rows(on_hash="different"))


def test_cli_rejects_unsupported_d_safe_before_touching_output(tmp_path):
    output_dir = tmp_path / "comparison"
    with pytest.raises(ValueError, match="accepts only"):
        run_comparison(
            config_path=Path("unused.yaml"),
            output_dir=output_dir,
            policy_dir=tmp_path / "policies",
            d_safe_values=[50.0],
            eval_count=2,
        )
    assert not output_dir.exists()


def test_exploratory_grid_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="accepts only"):
        validate_d_safe_values([30.0])

    validate_d_safe_values(
        list(EXPLORATORY_D_SAFE_VALUES), allow_exploratory=True
    )
    with pytest.raises(ValueError, match="accepts only"):
        validate_d_safe_values([31.0], allow_exploratory=True)
    with pytest.raises(ValueError, match="must be unique"):
        validate_d_safe_values([30.0, 30.0], allow_exploratory=True)


def test_policy_seed_filter_keeps_single_verified_policy():
    policies = [
        {"train_seed": "0", "file": "seed0.pt"},
        {"train_seed": "1", "file": "seed1.pt"},
    ]

    assert filter_policy_rows(policies, 1) == [policies[1]]
    assert filter_policy_rows(policies, None) == policies
    with pytest.raises(ValueError, match="expected one verified"):
        filter_policy_rows(policies, 2)


def test_runner_rejects_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    (output_dir / "preserve.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        run_comparison(
            config_path=Path("unused.yaml"),
            output_dir=output_dir,
            policy_dir=tmp_path / "policies",
            d_safe_values=[40.0],
            eval_count=2,
        )

    assert (output_dir / "preserve.txt").read_text(encoding="utf-8") == "keep"


def test_runner_rejects_unsupported_truth_model_before_touching_output(tmp_path):
    output_dir = tmp_path / "comparison"
    with pytest.raises(ValueError, match="--truth-model accepts only"):
        run_comparison(
            config_path=Path("unused.yaml"),
            output_dir=output_dir,
            policy_dir=tmp_path / "policies",
            d_safe_values=[40.0],
            eval_count=2,
            truth_models=("cw_j2_drag",),
        )
    assert not output_dir.exists()


def _write_gate_fixture(
    output_dir: Path,
    *,
    bad_hash_pair: bool = False,
    bad_off_reuse: bool = False,
    truth_models: tuple[str, ...] = ("cw",),
    selected_train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> None:
    policy_dir = output_dir / "policies"
    policy_dir.mkdir(parents=True)
    index_rows = []
    selected = []
    for train_seed in range(5):
        filename = f"primary_{train_seed}.pt"
        content = f"policy-{train_seed}".encode()
        (policy_dir / filename).write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        row = {
            "method": "sac_primary",
            "train_seed": str(train_seed),
            "step": "1",
            "file": filename,
            "sha256": digest,
        }
        index_rows.append(row)
        if train_seed in selected_train_seeds:
            selected.append(
                {
                    "method": "sac_primary",
                    "train_seed": train_seed,
                    "step": 1,
                    "file": filename,
                    "sha256": digest,
                }
            )
    for index in range(14):
        filename = f"other_{index}.pt"
        content = f"other-{index}".encode()
        (policy_dir / filename).write_bytes(content)
        index_rows.append(
            {
                "method": "ablation",
                "train_seed": str(index + 10),
                "step": "1",
                "file": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    with (policy_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)

    manifest = {
        "policy_dir": str(policy_dir),
        "selected_policies": selected,
        "d_safe_parameters_m": [40.0, 67.0],
        "truth_models": list(truth_models),
        "u_max_mps2": 1e-4,
        "evaluation_seeds": {"start": 10000, "count": 2},
        "expected_csv_rows": 8 * len(selected_train_seeds) * len(truth_models),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    episode_rows = []
    for truth_model in truth_models:
        for train_seed in selected_train_seeds:
            for eval_seed in (10000, 10001):
                for d_safe in (40.0, 67.0):
                    initial_hash = f"{truth_model}-{train_seed}-{eval_seed}"
                    common = {
                        "d_safe": d_safe,
                        "truth_model": truth_model,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "collided": 0,
                        "min_distance_m": 20.0,
                        "fuel_mps": 0.01,
                        "initial_state_hash": initial_hash,
                        "solver_error": "",
                        "backup_steps": 0,
                        "infeasible_steps": 0,
                        "feasible_step_count": 2,
                        "constraint_violation_max": 0.0,
                        "box_violation_max": 0.0,
                        "shield_off_reused_across_d_safe": 1,
                    }
                    episode_rows.append(
                        {
                            **common,
                            "condition": "shield_off",
                            "fuel_mps": (
                                0.02
                                if bad_off_reuse
                                and train_seed == 0
                                and eval_seed == 10000
                                and d_safe == 67.0
                                else 0.01
                            ),
                        }
                    )
                    on_hash = initial_hash
                    if (
                        bad_hash_pair
                        and truth_model == "cw"
                        and train_seed == 0
                        and eval_seed == 10000
                        and d_safe == 40.0
                    ):
                        on_hash = "bad"
                    episode_rows.append(
                        {
                            **common,
                            "condition": "shield_on",
                            "initial_state_hash": on_hash,
                        }
                    )
    with (output_dir / "shield_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    with (output_dir / "shield_comparison_steps.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerow(["condition", "d_safe", "mode"])


def test_gate_checker_passes_fixture_and_fails_on_hash_mismatch(tmp_path):
    passing_dir = tmp_path / "passing"
    passing_dir.mkdir()
    _write_gate_fixture(passing_dir)
    assert all(passed for _, passed, _ in check_gates(passing_dir))

    failing_dir = tmp_path / "failing"
    failing_dir.mkdir()
    _write_gate_fixture(failing_dir, bad_hash_pair=True)
    results = dict((name, passed) for name, passed, _ in check_gates(failing_dir))
    assert not results["identical_initial_state_hashes"]
    assert results["selected_policy_hashes_match_index"]

    reuse_failure_dir = tmp_path / "reuse_failure"
    reuse_failure_dir.mkdir()
    _write_gate_fixture(reuse_failure_dir, bad_off_reuse=True)
    reuse_results = dict(
        (name, passed) for name, passed, _ in check_gates(reuse_failure_dir)
    )
    assert not reuse_results["expected_row_count_and_shield_off_reuse"]


def test_gate_checker_counts_both_truth_models_and_reused_shield_off(tmp_path):
    output_dir = tmp_path / "two_models"
    output_dir.mkdir()
    _write_gate_fixture(output_dir, truth_models=("cw", "cw_j2"))
    gates = check_gates(output_dir)
    assert all(passed for _, passed, _ in gates)
    row_gate = next(evidence for name, _, evidence in gates if name == "expected_row_count_and_shield_off_reuse")
    assert "episode_rows=80 expected_rows=80" in row_gate
    assert "shield_off_d40_cw_j2=10" in row_gate
    assert "shield_on_d67_cw_j2=10" in row_gate


def test_gate_checker_validates_only_manifest_selected_policy_hashes(tmp_path):
    output_dir = tmp_path / "single_policy"
    output_dir.mkdir()
    _write_gate_fixture(output_dir, selected_train_seeds=(0,))

    gates = check_gates(output_dir)
    assert all(passed for _, passed, _ in gates)
    hash_evidence = next(
        evidence
        for name, _, evidence in gates
        if name == "selected_policy_hashes_match_index"
    )
    assert "policies_checked=1" in hash_evidence

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_policies"][0]["sha256"] = "incorrect"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    results = dict((name, passed) for name, passed, _ in check_gates(output_dir))
    assert not results["selected_policy_hashes_match_index"]
