import numpy as np
import pytest

from benchmarks.oracle_reference import (
    check_vectorized_matches_loop,
    evaluate_encounter,
    summarize_oracle_avoidable_subset,
)


def test_oracle_subset_summary_counts_only_passive_collisions():
    summary = summarize_oracle_avoidable_subset(
        [True, True, False],
        [(True, 12.0), (False, 9.0), (True, 100.0)],
    )

    assert summary == {
        "passive_collision_count": 2,
        "oracle_avoidable_count": 1,
        "avoidable_fraction": 0.5,
        "mean_best_min_distance_m": 10.5,
    }


def test_oracle_finds_clearance_for_constructed_avoidable_case():
    config = {"u_max": 1e-4, "dt": 1.0, "horizon_steps": 20, "d_safe_legacy_unused_m": 106.7, "initial_offset_m": 67.5, "closing_speed_mps": 0.5}
    avoidable, minimum = evaluate_encounter(0, config, 1.0)
    assert avoidable and minimum > 1.0


def test_oracle_reports_no_clearance_for_unreachable_case():
    config = {"u_max": 1e-4, "dt": 1.0, "horizon_steps": 600, "d_safe_legacy_unused_m": 106.7, "initial_offset_m": 25.0, "closing_speed_mps": 0.5}
    avoidable, minimum = evaluate_encounter(0, config, 100.0)
    assert not avoidable and minimum < 100.0


@pytest.mark.slow
def test_vectorized_oracle_matches_loop_reference():
    config = {"u_max": 1e-4, "dt": 1.0, "horizon_steps": 20, "d_safe_legacy_unused_m": 106.7, "initial_offset_m": 67.5, "closing_speed_mps": 0.5}
    assert check_vectorized_matches_loop([0, 1, 2, 3, 4], config, 1.0) < 1e-9