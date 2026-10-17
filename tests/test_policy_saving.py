import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import benchmarks.run_unshielded_benchmark as benchmark


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))


class _Agent:
    def __init__(self, **kwargs):
        self.actor = _Actor()

    def action(self, state, deterministic=False):
        if not deterministic:
            with torch.no_grad():
                self.actor.weight.add_(1.0)
        return np.array([float(self.actor.weight.detach()), 0.0, 0.0])


class _Environment:
    u_max = 1.0
    horizon_steps = 10
    done = False

    def reset(self, rng):
        return np.zeros(12)

    def step(self, action):
        return np.zeros(12), {
            "min_distance_in_step": 20.0,
            "distance": 20.0,
            "clipped": False,
        }


def _tiny_config():
    return {
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


def test_train_saves_only_selected_actor_with_index_and_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "SACAgent", _Agent)
    monkeypatch.setattr(benchmark, "make_environment", lambda config, seed: _Environment())

    def validation_episode(env, policy, rng, hard_body_radius_m, config, proximity_weight):
        action = policy(np.zeros(12), rng)
        return {
            "collided": int(action[0] > 1.5),
            "fuel_mps": float(action[0]),
        }

    monkeypatch.setattr(benchmark, "run_episode", validation_episode)
    policy_dir = tmp_path / "policies"
    _, best_step, _ = benchmark.train(
        3,
        _tiny_config(),
        0.0,
        policy_output_dir=policy_dir,
        method_name="sac_primary",
    )

    assert best_step == 1
    policy_files = list(policy_dir.glob("*.pt"))
    assert len(policy_files) == 1
    policy_path = policy_files[0]
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    with (policy_dir / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["file"] == policy_path.name
    assert rows[0]["sha256"] == digest
    sidecar = json.loads(policy_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert {
        "method",
        "train_seed",
        "step",
        "config_hash",
        "validation_collision_rate",
        "validation_mean_fuel",
        "sha256",
    } <= sidecar.keys()
    assert sidecar["sha256"] == digest
    assert sidecar["method"] == "sac_primary"
    assert sidecar["train_seed"] == 3
    assert sidecar["step"] == 1


def test_runner_refuses_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "nonempty"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        benchmark.main(Path("missing-config.yaml"), output_dir)

    assert (output_dir / "existing.txt").read_text(encoding="utf-8") == "preserve"
