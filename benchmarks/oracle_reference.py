"""Finite-candidate oracle feasibility reference for Step 2C."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import yaml

from aegis_orbit.envs import ContinuousConjunctionEnv
from aegis_orbit.envs.cw_relative_dynamics import cw_control_input_matrix, cw_state_transition

DEFAULT_CW_MEAN_MOTION_RAD_PER_S = 1.1e-3  # rad/s; Step 2C calibration from configs/sac_unshielded.yaml.


def load_config(path: Path = Path("configs/sac_unshielded.yaml")) -> dict:
    """Load the nested YAML configuration and reject tab indentation."""
    text = path.read_text(encoding="utf-8")
    if "\t" in text:
        raise ValueError(f"YAML contains tab indentation: {path}")
    config = yaml.safe_load(text)
    if config is None:
        raise ValueError(f"YAML file is empty: {path}")
    return config


def get(config: dict, *path: str):
    """Read a required nested config key and fail loudly when absent."""
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError("missing config key: " + ".".join(path))
        current = current[key]
    return current


def config_value(config: dict, *path: str):
    """Return a config value from nested or flat dictionaries."""
    try:
        return get(config, *path)
    except KeyError:
        if len(path) >= 2 and path[0] == "environment":
            if len(path) == 2 and path[1] in config:
                return config[path[1]]
            if len(path) == 2 and path[1] == "initial_offset_m" and "initial_offset_m" in config:
                return config["initial_offset_m"]
            if len(path) == 3 and path[1] == "initial_offset_m":
                if isinstance(config.get("initial_offset_m"), dict):
                    return config["initial_offset_m"][path[2]]
                if "initial_offset_m" in config and path[2] in {"low", "high"}:
                    return config["initial_offset_m"]
        if len(path) >= 2 and path[0] == "safety_geometry":
            if len(path) == 2 and path[1] in config:
                return config[path[1]]
            if len(path) == 2 and path[1] in {"d_safe_legacy_unused_m", "hard_body_radius_m", "proximity_reference_m"}:
                direct = path[1].replace("_m", "")
                if direct in config:
                    return config[direct]
        if len(path) == 2 and path[0] in config and isinstance(config[path[0]], dict) and path[1] in config[path[0]]:
            return config[path[0]][path[1]]
        if len(path) == 3 and path[0] in config and isinstance(config[path[0]], dict):
            nested = config[path[0]]
            if path[1] in nested and isinstance(nested[path[1]], dict) and path[2] in nested[path[1]]:
                return nested[path[1]][path[2]]
        if len(path) == 1 and path[0] in config:
            return config[path[0]]
        raise


def check_vectorized_matches_loop(seed_values: list[int], config: dict[str, float], hard_body_radius_m: float) -> float:
    """Check the vectorized-equivalent constant oracle against the loop oracle."""
    differences = []
    for seed in seed_values:
        loop = _evaluate_encounter_loop(seed, config, hard_body_radius_m)
        vectorized = evaluate_encounter(seed, config, hard_body_radius_m)
        if loop[0] != vectorized[0]:
            raise AssertionError("oracle avoidability mismatch")
        differences.append(abs(loop[1] - vectorized[1]))
    return float(max(differences, default=0.0))


def candidate_controls(u_max: float) -> list[np.ndarray]:
    """Return the finite control candidates used by the reference.

    The set contains all constant controls on the three-level Cartesian grid,
    zero control, and radial sign bang-bang controls. It is deliberately coarse.
    """
    candidates = [
        np.asarray(levels, dtype=np.float64) * u_max
        for levels in product((-1.0, 0.0, 1.0), repeat=3)
    ]
    candidates.extend(
        np.asarray(direction, dtype=np.float64) * u_max
        for direction in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    )
    return candidates


def summarize_oracle_avoidable_subset(
    passive_collision_flags: list[bool],
    oracle_results: list[tuple[bool, float]],
) -> dict[str, float | int]:
    """Summarize oracle clearance only for encounters that passively collided."""
    if len(passive_collision_flags) != len(oracle_results):
        raise ValueError("passive collision flags and oracle results must have equal lengths")
    subset = [
        (avoidable, best_distance)
        for passive_collision, (avoidable, best_distance) in zip(
            passive_collision_flags, oracle_results
        )
        if passive_collision
    ]
    if not subset:
        return {
            "passive_collision_count": 0,
            "oracle_avoidable_count": 0,
            "avoidable_fraction": 0.0,
            "mean_best_min_distance_m": 0.0,
        }
    avoidable_count = sum(int(avoidable) for avoidable, _ in subset)
    return {
        "passive_collision_count": len(subset),
        "oracle_avoidable_count": avoidable_count,
        "avoidable_fraction": avoidable_count / len(subset),
        "mean_best_min_distance_m": float(np.mean([distance for _, distance in subset])),
    }


def _evaluate_encounter_loop(
    seed: int,
    config: dict[str, float],
    hard_body_radius_m: float,
) -> tuple[bool, float]:
    """Evaluate one encounter against the finite candidate reference."""
    u_max = config_value(config, "environment", "u_max")
    candidates = candidate_controls(float(u_max))
    best_minimum = -np.inf
    avoidable = False
    for candidate in candidates:
        n_value = config_value(config, "environment", "n") if "environment" in config or "n" in config else DEFAULT_CW_MEAN_MOTION_RAD_PER_S
        environment = ContinuousConjunctionEnv(
            n=float(n_value),
            dt=float(config_value(config, "environment", "dt")),
            horizon_steps=int(config_value(config, "environment", "horizon_steps")),
            u_max=float(config_value(config, "environment", "u_max")),
            d_safe=float(config_value(config, "safety_geometry", "d_safe_legacy_unused_m")),
            initial_offset_m=float(np.mean([
                config_value(config, "environment", "initial_offset_m", "low"),
                config_value(config, "environment", "initial_offset_m", "high"),
            ])),
            closing_speed_mps=float(config_value(config, "environment", "closing_speed_mps")),
        )
        environment.reset(np.random.default_rng(seed))
        minimum = np.inf
        for _ in range(int(config_value(config, "environment", "horizon_steps"))):
            _, info = environment.step(candidate)
            minimum = min(minimum, float(info["min_distance_in_step"]))
        best_minimum = max(best_minimum, minimum)
        if minimum >= hard_body_radius_m:
            avoidable = True
    return avoidable, float(best_minimum)


def evaluate_encounter(
    seed: int,
    config: dict[str, float],
    hard_body_radius_m: float,
    initial_state: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Evaluate the finite constant-thrust oracle with a batched CW update.

    When supplied, ``initial_state`` is the exact 12-state used by another
    evaluation path for this encounter.
    """
    n = float(config_value(config, "environment", "n")) if "environment" in config or "n" in config else DEFAULT_CW_MEAN_MOTION_RAD_PER_S
    dt = float(config_value(config, "environment", "dt"))
    horizon = int(config_value(config, "environment", "horizon_steps"))
    u_max = float(config_value(config, "environment", "u_max"))
    offset = float(np.mean([
        config_value(config, "environment", "initial_offset_m", "low"),
        config_value(config, "environment", "initial_offset_m", "high"),
    ]))
    environment = ContinuousConjunctionEnv(
        n=n,
        dt=dt,
        horizon_steps=horizon,
        u_max=u_max,
        d_safe=float(config_value(config, "safety_geometry", "d_safe_legacy_unused_m")),
        initial_offset_m=offset,
        closing_speed_mps=float(config_value(config, "environment", "closing_speed_mps")),
    )
    if initial_state is None:
        initial = environment.reset(np.random.default_rng(seed))[:6]
    else:
        encounter_state = np.asarray(initial_state, dtype=np.float64)
        if encounter_state.shape != (12,):
            raise ValueError("initial_state must be a 12-element encounter state")
        initial = encounter_state[:6].copy()
    candidates = np.asarray(candidate_controls(u_max), dtype=np.float64)
    sub_times = np.linspace(0.0, dt, 21, dtype=np.float64)
    phi = np.asarray([cw_state_transition(n, time) for time in sub_times])
    gamma = np.asarray([cw_control_input_matrix(n, time) for time in sub_times])
    states = np.repeat(initial[None, :], len(candidates), axis=0)
    minima = np.full(len(candidates), np.inf, dtype=np.float64)
    for _ in range(horizon):
        samples = np.einsum("tij,cj->tci", phi, states) + np.einsum("tij,cj->tci", gamma, candidates)
        minima = np.minimum(minima, np.linalg.norm(samples[:, :, :3], axis=2).min(axis=0))
        states = samples[-1]
    best_minimum = float(np.max(minima))
    return bool(best_minimum >= hard_body_radius_m), best_minimum


def main() -> None:
    """Print the oracle fraction on the configured training-distribution seeds."""
    config = load_config()
    hard_body_radius = float(config_value(config, "safety_geometry", "hard_body_radius_m"))
    seeds = range(5)
    results = [evaluate_encounter(seed, config, hard_body_radius) for seed in seeds]
    fraction = sum(avoidable for avoidable, _ in results) / len(results)
    print("oracle_method=finite_constant_and_bang_bang_candidates")
    print("oracle_resolution=29 candidate sequences per encounter")
    print(f"training_parameters={config}")
    print(f"avoidable_fraction={fraction:.3f}")
    for seed, (avoidable, minimum) in zip(seeds, results):
        print(f"seed={seed}, avoidable={avoidable}, best_min_distance_m={minimum:.6f}")


if __name__ == "__main__":
    main()