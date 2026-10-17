"""Evaluate the requested labelled d_safe parameter grid at training starts."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np

from .oracle_reference import get, load_config

HEADER_TEXT = "PARAMETER-SWEEP, NOT A DERIVED SAFE RADIUS"
GRID_D_SAFE_M = (30.0, 40.0, 50.0, 60.0, 67.0)
INTERIOR_OFFSET_COUNT = 200
CSV_FIELDS = (
    HEADER_TEXT,
    "d_safe_m",
    "min_h_m2",
    "fraction_start_h_nonnegative",
    "sampled_start_count",
    "verdict",
)


def barrier_at_offset(offset_m: float, d_safe_m: float) -> float:
    """Calculate h for the deterministic radial start vector [offset, 0, 0]."""
    relative_position = np.array([offset_m, 0.0, 0.0], dtype=np.float64)
    return float(np.dot(relative_position, relative_position) - d_safe_m**2)


def compute_grid(config: dict) -> list[dict[str, object]]:
    """Compute parameter-only start-state barrier rows across the configured band."""
    low = float(get(config, "environment", "initial_offset_m", "low"))
    high = float(get(config, "environment", "initial_offset_m", "high"))
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("configured initial-offset band must be finite and ordered")

    offsets = np.linspace(
        low,
        high,
        INTERIOR_OFFSET_COUNT + 2,
        dtype=np.float64,
    )
    rows = []
    for d_safe_m in GRID_D_SAFE_M:
        barrier_values = np.asarray(
            [barrier_at_offset(float(offset), d_safe_m) for offset in offsets],
            dtype=np.float64,
        )
        fraction_nonnegative = float(np.mean(barrier_values >= 0.0))
        rows.append({
            HEADER_TEXT: HEADER_TEXT,
            "d_safe_m": d_safe_m,
            "min_h_m2": float(np.min(barrier_values)),
            "fraction_start_h_nonnegative": fraction_nonnegative,
            "sampled_start_count": len(offsets),
            "verdict": "PASS" if bool(np.all(barrier_values >= 0.0)) else "FAIL",
        })
    return rows


def write_grid(output_dir: Path, rows: list[dict[str, object]]) -> str:
    """Write the labelled grid without overwriting existing output."""
    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries:
            raise FileExistsError(
                f"Refusing to write into non-empty output directory: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)
    output_path = output_dir / "shield_grid.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing grid: {output_path}")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    output_path.write_text(content, encoding="utf-8", newline="")
    return content


def cli(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_unshielded.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/shield_grid_20261014"),
    )
    parsed = parser.parse_args(arguments)
    config = load_config(parsed.config)
    content = write_grid(parsed.output_dir, compute_grid(config))
    print(content, end="", flush=True)


if __name__ == "__main__":
    cli()
