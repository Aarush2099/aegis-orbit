import csv

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
        if not deterministic:
            with torch.no_grad():
                self.actor.weight.add_(1.0)
        return np.array([float(self.actor.weight.detach()), 0.0, 0.0])


class _Environment:
    u_max = 1.0
    horizon_steps = 10

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


def test_train_restores_best_validation_actor_and_persists_selection(monkeypatch, tmp_path):
    environment = _Environment()
    monkeypatch.setattr(benchmark, "SACAgent", _Agent)
    monkeypatch.setattr(benchmark, "make_environment", lambda config, seed: environment)

    def validation_episode(env, policy, rng, hard_body_radius_m, config, proximity_weight):
        action = policy(np.zeros(12), rng)
        return {
            "collided": int(action[0] > 1.5),
            "fuel_mps": float(action[0]),
        }

    monkeypatch.setattr(benchmark, "run_episode", validation_episode)
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
            "batch_size": 10,
            "validation_interval_steps": 1,
        },
        "validation_seeds": {"start": 10, "count": 1},
        "reward": {
            "fuel_weight": 1.0,
            "collision_penalty": 100.0,
            "proximity_weight": 0.0,
            "survival_bonus": 10.0,
        },
    }
    selection_path = tmp_path / "checkpoint_selection.csv"
    agent, best_step, best_score = benchmark.train(
        0,
        config,
        0.0,
        selection_logs=[],
        selection_path=selection_path,
    )

    assert best_step == 1
    assert best_score == (0.0, 1.0)
    assert float(agent.actor.weight.detach()) == 1.0
    with selection_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["selected"] == "True"
    assert rows[1]["selected"] == "False"
