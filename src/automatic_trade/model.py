"""PyTorch Q-network and one DQN optimization step."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .replay import Transition


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int = 2, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def optimize_dqn(
    batch: Sequence[Transition],
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    device: torch.device,
) -> float:
    states = torch.as_tensor([x.state for x in batch], dtype=torch.float32, device=device)
    actions = torch.as_tensor([x.action for x in batch], dtype=torch.long, device=device)
    rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=device)
    next_states = torch.as_tensor([x.next_state for x in batch], dtype=torch.float32, device=device)
    done = torch.as_tensor([x.done for x in batch], dtype=torch.float32, device=device)

    predicted = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        expected = rewards + gamma * target(next_states).max(dim=1).values * (1.0 - done)
    loss = F.smooth_l1_loss(predicted, expected)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())

