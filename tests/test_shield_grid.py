import csv
from pathlib import Path

from benchmarks.oracle_reference import load_config
from benchmarks.shield_grid import (
    HEADER_TEXT,
    barrier_at_offset,
    compute_grid,
    write_grid,
)


def test_67_m_parameter_is_on_barrier_at_67_m_start_and_output_is_labelled(tmp_path):
    assert 67.0**2 - 67**2 == 0.0
    h_at_endpoint = barrier_at_offset(offset_m=67.0, d_safe_m=67.0)
    assert h_at_endpoint == 0.0
    assert h_at_endpoint >= 0.0

    config = load_config(Path("configs/sac_unshielded.yaml"))
    rows = compute_grid(config)
    assert all(row["sampled_start_count"] == 202 for row in rows)
    assert all(row[HEADER_TEXT] == HEADER_TEXT for row in rows)

    content = write_grid(tmp_path / "grid", rows)
    with (tmp_path / "grid" / "shield_grid.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[0] == HEADER_TEXT
        parsed_rows = list(reader)
    assert len(parsed_rows) == len(rows)
    assert all(row[HEADER_TEXT] == HEADER_TEXT for row in parsed_rows)
    assert HEADER_TEXT in content
    assert parsed_rows[-1]["d_safe_m"] == "67.0"
    assert float(parsed_rows[-1]["min_h_m2"]) == 0.0
    assert parsed_rows[-1]["verdict"] == "PASS"
