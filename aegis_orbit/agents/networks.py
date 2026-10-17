"""Torch networks for the unshielded SAC baseline."""

import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Tanh-squashed Gaussian actor with per-axis acceleration scaling."""

    def __init__(self, state_dim: int, action_dim: int, u_max: float) -> None:
        super().__init__()
        self.u_max = float(u_max)
        self.body = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(state)
        return self.mean(hidden), self.log_std(hidden).clamp(-20.0, 2.0)

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample bounded actions and finite log probabilities."""
        mean, log_std = self(state)
        distribution = Normal(mean, log_std.exp())
        latent = distribution.rsample()
        squashed = torch.tanh(latent)
        action = self.u_max * squashed
        log_probability = distribution.log_prob(latent) - torch.log(1.0 - squashed.square() + 1e-6)
        return action, log_probability.sum(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    """Two-hidden-layer state-action value network."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, action), dim=-1))