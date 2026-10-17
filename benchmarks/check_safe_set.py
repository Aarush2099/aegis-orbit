"""Check the initial-state barrier condition across the configured training band."""

from __future__ import annotations

import numpy as np

from aegis_orbit.envs import ContinuousConjunctionEnv
from .oracle_reference import get, load_config


def check_safe_set() -> int:
    """Print the minimum initial barrier value, or stop on unresolved inputs."""
    config = load_config()
    geometry = config["safety_geometry"]
    sigma_miss = geometry["sigma_miss_m"]
    k_sigma = geometry["k_sigma"]
    if sigma_miss is None:
        print(
            "BLOCKED: d_safe_m cannot be derived because sigma_miss_m is "
            "PLACEHOLDER (null); no start-state barrier result is available."
        )
        return 2

    hard_body_radius = float(get(config, "safety_geometry", "hard_body_radius_m"))
    derived_d_safe = hard_body_radius + float(k_sigma) * float(sigma_miss)
    low = float(get(config, "environment", "initial_offset_m", "low"))
    high = float(get(config, "environment", "initial_offset_m", "high"))
    offsets = np.concatenate((np.linspace(low, high, 200), np.array([low, high])))
    rng = np.random.default_rng(0)
    barriers = []
    for offset in offsets:
        environment = ContinuousConjunctionEnv(
            n=float(get(config, "environment", "n")),
            dt=float(get(config, "environment", "dt")),
            horizon_steps=int(get(config, "environment", "horizon_steps")),
            u_max=float(get(config, "environment", "u_max")),
            d_safe=derived_d_safe,
            initial_offset_m=float(offset),
            closing_speed_mps=float(get(config, "environment", "closing_speed_mps")),
            initial_offset_range_m=None,
        )
        initial_state = environment.reset(rng)
        relative_position = initial_state[:3]
        barriers.append(float(np.dot(relative_position, relative_position) - derived_d_safe**2))

    minimum_barrier = min(barriers)
    print(f"minimum_h={minimum_barrier:.12g} m^2")
    if minimum_barrier < 0.0:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_safe_set())
