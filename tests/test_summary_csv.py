import csv

from benchmarks.run_unshielded_benchmark import (
    ORACLE_SUMMARY_FIELDS,
    SUMMARY_FIELDS,
    write_summary_files,
)


def _assert_csv_rows_match_header(content: str) -> None:
    rows = list(csv.DictReader(content.splitlines()))
    header = next(csv.reader(content.splitlines()))
    assert all(list(row.keys()) == header for row in rows)


def test_summary_csv_rows_have_exactly_the_declared_header_fields(tmp_path):
    benchmark_rows = [
        {
            "method": "passive",
            "collided": True,
            "min_distance_m": 9.5,
            "fuel_mps": 0.0,
        },
        {
            "method": "random",
            "collided": False,
            "min_distance_m": 11.0,
            "fuel_mps": 0.25,
        },
    ]
    oracle_summary = {
        "passive_collision_count": 1,
        "oracle_avoidable_count": 1,
        "avoidable_fraction": 1.0,
        "mean_best_min_distance_m": 10.5,
    }

    summary_content, oracle_content = write_summary_files(
        tmp_path,
        benchmark_rows,
        oracle_summary,
        "oracle finite-search ceiling, not a policy result",
    )

    _assert_csv_rows_match_header(summary_content)
    _assert_csv_rows_match_header(oracle_content)
    assert next(csv.reader(summary_content.splitlines())) == list(SUMMARY_FIELDS)
    assert next(csv.reader(oracle_content.splitlines())) == list(ORACLE_SUMMARY_FIELDS)
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        assert all(set(row) == set(SUMMARY_FIELDS) for row in csv.DictReader(handle))
    with (tmp_path / "oracle_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert set(rows[0]) == set(ORACLE_SUMMARY_FIELDS)
    assert rows[0]["label"] == "oracle finite-search ceiling, not a policy result"
