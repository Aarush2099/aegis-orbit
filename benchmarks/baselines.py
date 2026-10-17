"""Passive and random unshielded baselines."""

import numpy as np

from aegis_orbit.envs import ContinuousConjunctionEnv


def run_episode(environment: ContinuousConjunctionEnv, policy, rng: np.random.Generator, hard_body_radius_m: float) -> dict[str, float]:
    """Run one plain environment episode and return unshaped metrics."""
    state = environment.reset(rng)
    minimum = np.inf
    fuel = 0.0
    clipped = 0
    for _ in range(environment.horizon_steps):
        action = np.asarray(policy(state, rng), dtype=np.float64)
        state, info = environment.step(action)
        minimum = min(minimum, float(info["min_distance_in_step"]))
        fuel += float(np.linalg.norm(np.clip(action, -environment.u_max, environment.u_max)) * environment.dt)
        clipped += int(bool(info["clipped"]))
    return {"collided": float(minimum < hard_body_radius_m), "min_distance_m": float(minimum), "final_distance_m": float(info["distance"]), "fuel_mps": fuel, "clipped_fraction": clipped / environment.horizon_steps}


def passive_policy(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return zero acceleration."""
    return np.zeros(3, dtype=np.float64)


def random_policy(u_max: float):
    """Return a seeded uniform random policy factory."""
    def policy(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-u_max, u_max, size=3).astype(np.float64)
    return policy