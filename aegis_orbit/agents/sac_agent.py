"""Minimal CPU SAC implementation without environment-specific shield logic."""

from collections import deque
import copy
import random

import numpy as np
import torch
from torch import nn, optim

from .networks import Actor, QNetwork


class ReplayBuffer:
    """Uniform replay storage for float32 Torch training batches."""

    def __init__(self, capacity: int = 100_000, rng: np.random.Generator | None = None) -> None:
        self.capacity = capacity
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.storage: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.storage.append((np.asarray(state, np.float32), np.asarray(action, np.float32), float(reward), np.asarray(next_state, np.float32), float(done)))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        batch = [self.storage[int(index)] for index in indices]
        values = list(zip(*batch))
        return tuple(torch.as_tensor(np.asarray(value)) for value in values)

    def __len__(self) -> int:
        return len(self.storage)


class SACAgent:
    """Twin-critic SAC agent with automatic entropy temperature tuning."""

    def __init__(self, state_dim: int, action_dim: int, u_max: float, learning_rate: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, target_entropy: float = -3.0) -> None:
        self.actor = Actor(state_dim, action_dim, u_max)
        self.critic1 = QNetwork(state_dim, action_dim)
        self.critic2 = QNetwork(state_dim, action_dim)
        self.target1 = copy.deepcopy(self.critic1)
        self.target2 = copy.deepcopy(self.critic2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=learning_rate)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.gamma, self.tau, self.target_entropy = gamma, tau, target_entropy

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, _ = self.actor(tensor)
            value = self.actor.u_max * torch.tanh(mean) if deterministic else self.actor.sample(tensor)[0]
        return value.squeeze(0).numpy()

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        state, action, reward, next_state, done = batch
        reward, done = reward.float().unsqueeze(-1), done.float().unsqueeze(-1)
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state)
            target_q = torch.min(self.target1(next_state, next_action), self.target2(next_state, next_action)) - self.alpha.detach() * next_logp
            target = reward + self.gamma * (1.0 - done) * target_q
        q1, q2 = self.critic1(state, action), self.critic2(state, action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        new_action, logp = self.actor.sample(state)
        actor_loss = (self.alpha.detach() * logp - torch.min(self.critic1(state, new_action), self.critic2(state, new_action))).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
        for online, target_network in zip(self.critic1.parameters(), self.target1.parameters()): target_network.data.mul_(1-self.tau).add_(self.tau * online.data)
        for online, target_network in zip(self.critic2.parameters(), self.target2.parameters()): target_network.data.mul_(1-self.tau).add_(self.tau * online.data)
        q_values = torch.cat((q1.detach(), q2.detach()))
        values = [
            float(critic_loss.detach()),
            float(actor_loss.detach()),
            float(alpha_loss.detach()),
            float(self.alpha.detach()),
            float(q_values.min()),
            float(q_values.max()),
            float((-logp.detach()).mean()),
        ]
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "q_min": float(q_values.min()),
            "q_max": float(q_values.max()),
            "entropy": float((-logp.detach()).mean()),
            "nan": int(bool(not np.isfinite(values).all())),
        }