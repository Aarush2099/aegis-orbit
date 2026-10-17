import numpy as np
import torch
from torch import nn

import benchmarks.run_unshielded_benchmark as benchmark


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))


class _Agent:
    def __init__(self, *args, **kwargs):
        self.actor = _Actor()

    def action(self, state, deterministic=False):
        return np.zeros(3)

    def update(self, batch):
        return {
            "actor_loss": 0.5,
            "critic_loss": 0.75,
            "q_min": -1.0,
            "q_max": 1.0,
            "alpha": 0.2,
            "entropy": 2.0,
            "nan": 0,
        }


class _Environment:
    u_max = 1.0
    horizon_steps = 1

    def __init__(self):
        self.steps = 0

    def reset(self, rng):
        self.steps = 0
        return np.zeros(12)

    def step(self, action):
        self.steps += 1
        return np.zeros(12), {
            "min_distance_in_step": 20.0,
            "distance": 20.0,
            "clipped": False,
        }

    @property
    def done(self):
        return self.steps >= self.horizon_steps


def test_training_writes_complete_stability_and_episode_logs(monkeypatch):
    environment = _Environment()
    monkeypatch.setattr(benchmark, "SACAgent", _Agent)
    monkeypatch.setattr(benchmark, "make_environment", lambda config, seed: environment)
    monkeypatch.setattr(
        benchmark,
        "run_episode",
        lambda *args, **kwargs: {"collided": False, "fuel_mps": 0.0},
    )
    config = {
        "environment": {"u_max": 1.0},
        "safety_geometry": {
            "hard_body_radius_m": 10.0,
            "proximity_reference_m": 67.0,
        },
        "training": {
            "actor_lr": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "target_entropy": -3.0,
            "replay_capacity": 10,
            "total_env_steps": 2,
            "warmup_steps": 0,
            "batch_size": 1,
            "validation_interval_steps": 2,
        },
        "validation_seeds": {"start": 10, "count": 1},
        "reward": {
            "fuel_weight": 1.0,
            "collision_penalty": 100.0,
            "proximity_weight": 0.0,
            "survival_bonus": 10.0,
        },
    }
    stability_logs = []
    reward_logs = []
    benchmark.train(0, config, 0.0, stability_logs=stability_logs, reward_logs=reward_logs)

    assert len(stability_logs) == 2
    assert {
        "seed", "step", "actor_loss", "critic_loss", "q_min", "q_max",
        "alpha", "entropy", "nan",
    } <= stability_logs[0].keys()
    assert all(row["nan"] == 0 for row in stability_logs)
    assert len(reward_logs) == 2
    assert all("reward" in row and "collision_term" in row for row in reward_logs)
