import numpy as np
import torch

from aegis_orbit.agents.networks import Actor


def test_actor_shapes_bounds_and_log_probability():
    actor = Actor(12, 3, 1e-4)
    action, log_probability = actor.sample(torch.zeros((8, 12)))
    assert action.shape == (8, 3)
    assert log_probability.shape == (8, 1)
    assert torch.isfinite(log_probability).all()
    assert torch.max(torch.abs(action)) <= 1e-4
    assert np.isfinite(action.detach().numpy()).all()