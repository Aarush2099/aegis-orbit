"""Check integrity and action constraints for a shield comparison output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .oracle_reference import get, load_config
from .run_shield_comparison import select_final_policy_rows, sha256_file


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_gates(output_dir: Path) -> list[tuple[str, bool, str]]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    episode_rows = _read_csv(output_dir / "shield_comparison.csv")
    step_rows = _read_csv(output_dir / "shield_comparison_steps.csv")
    truth_models = set(manifest.get("truth_models", ["cw"]))
    policy_dir = Path(manifest["policy_dir"])
    index_rows, final_selected = select_final_policy_rows(policy_dir)
    indexed_by_seed = {
        int(row["train_seed"]): row
        for row in index_rows
        if row["method"] == "sac_primary"
    }
    final_selected_by_seed = {
        int(row["train_seed"]): row for row in final_selected
    }
    selected = manifest.get("selected_policies", [])
    pair_hashes: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in episode_rows:
        key = (row["train_seed"], row["eval_seed"], row["truth_model"], row["d_safe"])
        pair_hashes.setdefault(key, {})[row["condition"]] = row["initial_state_hash"]
    paired = all(
        set(conditions) == {"shield_off", "shield_on"}
        and conditions["shield_off"] == conditions["shield_on"]
        for conditions in pair_hashes.values()
    )
    expected_pairs = (
        len(manifest["selected_policies"])
        * int(manifest["evaluation_seeds"]["count"])
        * len(manifest["d_safe_parameters_m"])
        * len(truth_models)
    )
    paired = paired and len(pair_hashes) == expected_pairs
    gates: list[tuple[str, bool, str]] = [
        (
            "identical_initial_state_hashes",
            paired,
            f"paired_keys={len(pair_hashes)} expected_pairs={expected_pairs} mismatches="
            f"{sum(not (set(c)=={'shield_off','shield_on'} and c.get('shield_off')==c.get('shield_on')) for c in pair_hashes.values())}",
        )
    ]
    hash_mismatches = 0
    selected_seeds: set[int] = set()
    for row in selected:
        train_seed = int(row["train_seed"])
        indexed = indexed_by_seed.get(train_seed)
        final = final_selected_by_seed.get(train_seed)
        path = policy_dir / row["file"]
        if (
            train_seed in selected_seeds
            or indexed is None
            or final is None
            or row["method"] != "sac_primary"
            or row["file"] != final["file"]
            or int(row["step"]) != int(final["step"])
            or row["sha256"].strip().lower() != indexed.get("sha256", "").strip().lower()
            or not path.is_file()
            or sha256_file(path) != indexed.get("sha256", "").strip().lower()
        ):
            hash_mismatches += 1
        selected_seeds.add(train_seed)
    gates.append(
        (
            "selected_policy_hashes_match_index",
            bool(selected) and hash_mismatches == 0,
            f"policies_checked={len(selected)} hash_mismatches={hash_mismatches}",
        )
    )

    feasible_rows = [
        row
        for row in episode_rows
        if row["condition"] == "shield_on" and int(row["feasible_step_count"]) > 0
    ]
    summary_action_ok = all(
        float(row["constraint_violation_max"]) <= 1e-9
        and float(row["box_violation_max"]) <= 0.0
        for row in feasible_rows
    )
    recorded_feasible_steps = [
        row
        for row in step_rows
        if row["condition"] == "shield_on" and row["feasible"] == "1"
    ]
    if "u_max_mps2" in manifest:
        u_max = float(manifest["u_max_mps2"])
    else:
        config = load_config(Path(manifest["config"]))
        u_max = float(get(config, "environment", "u_max"))
    recorded_action_ok = all(
        float(row["a_dot_u"]) <= float(row["b"]) + 1e-9
        and max(abs(float(row[f"u_{axis}"])) for axis in "xyz")
        <= u_max + 1e-12
        for row in recorded_feasible_steps
    )
    action_ok = summary_action_ok and recorded_action_ok
    gates.append(
        (
            "feasible_step_constraint_and_box",
            action_ok,
            f"shield_episodes_with_feasible_steps={len(feasible_rows)} "
            f"recorded_feasible_steps={len(recorded_feasible_steps)} "
            f"max_constraint_violation={max((float(r['constraint_violation_max']) for r in feasible_rows), default=0.0):.12g} "
            f"max_box_violation={max((float(r['box_violation_max']) for r in feasible_rows), default=0.0):.12g} "
            f"recorded_step_checks={'PASS' if recorded_action_ok else 'FAIL'}",
        )
    )
    backup_ok = all(int(row["backup_steps"]) == int(row["infeasible_steps"]) for row in episode_rows)
    gates.append(
        (
            "backup_steps_equal_infeasible_steps",
            backup_ok,
            f"episode_rows={len(episode_rows)} mismatches="
            f"{sum(int(r['backup_steps']) != int(r['infeasible_steps']) for r in episode_rows)}",
        )
    )
    solver_errors = [row for row in episode_rows if row.get("solver_error", "").strip()]
    gates.append(
        (
            "no_solver_errors",
            not solver_errors,
            f"solver_error_rows={len(solver_errors)}",
        )
    )

    expected_rows = int(manifest["expected_csv_rows"])
    expected_per_group = (
        len(manifest["selected_policies"])
        * int(manifest["evaluation_seeds"]["count"])
    )
    d_safe_values = {float(value) for value in manifest["d_safe_parameters_m"]}
    group_counts: dict[tuple[str, float, str], int] = {}
    for row in episode_rows:
        key = (row["condition"], float(row["d_safe"]), row["truth_model"])
        group_counts[key] = group_counts.get(key, 0) + 1
    required_groups = {
        (condition, d_safe, truth_model)
        for condition in ("shield_off", "shield_on")
        for d_safe in d_safe_values
        for truth_model in truth_models
    }
    group_counts_ok = (
        set(group_counts) == required_groups
        and all(group_counts[key] == expected_per_group for key in required_groups)
    )

    off_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in episode_rows:
        if row["condition"] == "shield_off":
            off_groups.setdefault(
                (row["train_seed"], row["eval_seed"], row["truth_model"]), []
            ).append(row)
    off_reused_ok = all(
        {float(row["d_safe"]) for row in rows} == d_safe_values
        and len(
            {
                (
                    row["collided"],
                    row["min_distance_m"],
                    row["fuel_mps"],
                    row["initial_state_hash"],
                )
                for row in rows
            }
        )
        == 1
        and all(row["shield_off_reused_across_d_safe"] == "1" for row in rows)
        for rows in off_groups.values()
    )
    row_count_ok = (
        len(episode_rows) == expected_rows
        and len(episode_rows) == expected_per_group * len(required_groups)
        and group_counts_ok
        and off_reused_ok
        and len(off_groups) == expected_per_group * len(truth_models)
    )
    reuse_mismatches = sum(
        not (
            {float(row["d_safe"]) for row in rows} == d_safe_values
            and len(
                {
                    (
                        row["collided"],
                        row["min_distance_m"],
                        row["fuel_mps"],
                        row["initial_state_hash"],
                    )
                    for row in rows
                }
            )
            == 1
            and all(row["shield_off_reused_across_d_safe"] == "1" for row in rows)
        )
        for rows in off_groups.values()
    )
    gates.append(
        (
            "expected_row_count_and_shield_off_reuse",
            row_count_ok,
            f"episode_rows={len(episode_rows)} expected_rows={expected_rows} "
            f"models={','.join(sorted(truth_models))} "
            f"groups={{{', '.join(f'{key[0]}_d{key[1]:g}_{key[2]}={group_counts.get(key, 0)}' for key in sorted(required_groups))}}} "
            f"expected_per_group={expected_per_group} "
            f"off_groups_reused={len(off_groups)} off_reuse_mismatches={reuse_mismatches} "
            f"infeasible_step_rows={len(step_rows)}",
        )
    )
    return gates


def cli(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(arguments)
    checks = check_gates(args.output_dir)
    for name, passed, evidence in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {evidence}")
    if not all(passed for _, passed, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
