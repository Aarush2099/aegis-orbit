import csv
import hashlib
from pathlib import Path

from benchmarks.verify_policies import verify_policies


def test_tampered_policy_file_fails_index_hash_check(tmp_path):
    output_dir = tmp_path / "run"
    policy_dir = output_dir / "policies"
    policy_dir.mkdir(parents=True)
    policy_path = policy_dir / "sac_primary_seed0_step100.pt"
    policy_path.write_bytes(b"original actor bytes")
    original_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    with (policy_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "train_seed",
                "step",
                "file",
                "sha256",
                "validation_collision_rate",
                "validation_mean_fuel",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "method": "sac_primary",
            "train_seed": 0,
            "step": 100,
            "file": policy_path.name,
            "sha256": original_hash,
            "validation_collision_rate": 0.0,
            "validation_mean_fuel": 0.0,
        })
    with (output_dir / "unshielded_benchmark.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "train_seed",
                "eval_seed",
                "collided",
                "min_distance_m",
                "fuel_mps",
            ],
        )
        writer.writeheader()

    policy_path.write_bytes(b"tampered actor bytes")
    result = verify_policies(
        Path("configs/sac_unshielded.yaml"),
        output_dir,
    )

    assert result["passed"] is False
    assert any("sha256 mismatch" in failure for failure in result["failures"])


def test_missing_indexed_policy_file_fails_hash_coverage_check(tmp_path):
    output_dir = tmp_path / "run"
    policy_dir = output_dir / "policies"
    policy_dir.mkdir(parents=True)
    with (policy_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "train_seed",
                "step",
                "file",
                "sha256",
                "validation_collision_rate",
                "validation_mean_fuel",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "method": "sac_primary",
            "train_seed": 0,
            "step": 100,
            "file": "missing.pt",
            "sha256": "configured-digest",
            "validation_collision_rate": 0.0,
            "validation_mean_fuel": 0.0,
        })
    with (output_dir / "unshielded_benchmark.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "train_seed",
                "eval_seed",
                "collided",
                "min_distance_m",
                "fuel_mps",
            ],
        )
        writer.writeheader()

    result = verify_policies(
        Path("configs/sac_unshielded.yaml"),
        output_dir,
    )

    assert result["passed"] is False
    assert result["indexed_files_hashed"] == 0
    assert result["hash_mismatches"] == 1
    assert any(
        "hashed 0 indexed files; expected 1" in failure
        for failure in result["failures"]
    )
