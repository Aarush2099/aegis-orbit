"""Generate the Step 2B passive perturbation sensitivity artifacts."""

import argparse
from pathlib import Path
import struct
import zlib

import numpy as np

from aegis_orbit.envs import PerturbedConjunctionEnv
from aegis_orbit.utils import mean_motion, orbit_radius


def _write_png(path: Path, values: list[float]) -> None:
    """Write a small dependency-free bar chart PNG."""
    width, height = 900, 500
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    maximum = max(values) if values else 1.0
    bar_width = max(1, width // (len(values) * 2))
    for index, value in enumerate(values):
        x0 = (2 * index + 1) * width // (2 * len(values)) - bar_width // 2
        x1 = min(width, x0 + bar_width)
        y0 = height - 40
        y1 = int(y0 - 400.0 * value / maximum) if maximum else y0
        pixels[max(0, y1):y0, max(0, x0):x1, :] = np.array([40, 100, 180], dtype=np.uint8)
    baseline_y = int(height - 40.0 - 400.0 * values[0] / maximum) if maximum else height - 40
    pixels[max(0, baseline_y - 2):min(height, baseline_y + 3), :, :] = np.array(
        [190, 40, 40], dtype=np.uint8
    )
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    content = b"\x89PNG\r\n\x1a\n"
    content += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    content += chunk(b"IDAT", zlib.compress(raw, 9))
    content += chunk(b"IEND", b"")
    path.write_bytes(content)


def main(results_dir: Path) -> None:
    """Run five explicit seeds for each passive perturbation configuration."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "perturbation_sensitivity.csv"
    png_path = results_dir / "perturbation_sensitivity.png"
    existing = [path for path in (csv_path, png_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {existing}")
    n = mean_motion(orbit_radius(550e3))
    configs = {
        "CW only": {},
        "CW + J2": {"j2_enabled": True, "truth_model": "cw_j2"},
        "CW + drag, nominal density": {"drag_enabled": True},
        "CW + drag, storm peak": {
            "drag_enabled": True,
            "storm": {"onset_s": 0.0, "rise_s": 1.0, "peak_factor": 2.0, "decay_s": 3600.0},
        },
        "CW + J2 + drag + storm": {
            "j2_enabled": True,
            "truth_model": "cw_j2",
            "drag_enabled": True,
            "storm": {"onset_s": 0.0, "rise_s": 1.0, "peak_factor": 2.0, "decay_s": 3600.0},
        },
    }
    rows = []
    for config, options in configs.items():
        for seed in range(5):
            environment = PerturbedConjunctionEnv(
                n=n,
                dt=1.0,
                horizon_steps=600,
                u_max=1e-4,
                d_safe=106.7,
                initial_offset_m=200.0,
                closing_speed_mps=0.5,
                allow_unvalidated_d_safe=True,
                ballistic_coeff_chief=2.0e-2,
                ballistic_coeff_deputy=3.0e-2,
                **options,
            )
            environment.reset(np.random.default_rng(seed))
            distances = []
            for _ in range(600):
                distances.append(float(np.linalg.norm(environment.relative_state()[:3])))
                environment.step(np.zeros(3, dtype=np.float64))
            rows.append((config, seed, min(distances), distances[-1]))
    csv_path.write_text(
        "config,seed,min_distance_m,final_distance_m\n"
        + "\n".join(
            f"{config},{seed},{minimum:.12g},{final:.12g}"
            for config, seed, minimum, final in rows
        )
        + "\n",
        encoding="utf-8",
    )
    grouped = {config: [row[2] for row in rows if row[0] == config] for config in configs}
    _write_png(png_path, [float(np.mean(grouped[c])) for c in configs])
    print("config | mean minimum distance (m)")
    for config in configs:
        print(f"{config} | {np.mean(grouped[config]):.6f}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the CSV and PNG sensitivity outputs.",
    )
    args = parser.parse_args()
    main(args.output_dir)


if __name__ == "__main__":
    cli()
