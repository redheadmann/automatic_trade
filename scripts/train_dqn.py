#!/usr/bin/env python3
"""Train the starter DQN against historical OHLCV data."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from automatic_trade.data import chronological_split, load_ohlcv, prepare_features
from automatic_trade.environment import TradingEnvironment
from automatic_trade.model import QNetwork, optimize_dqn
from automatic_trade.replay import ReplayBuffer, Transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = prepare_features(load_ohlcv(args.csv))
    train, test = chronological_split(frame)
    environment = TradingEnvironment(train, transaction_cost=args.transaction_cost)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online = QNetwork(environment.state_size).to(device)
    target = QNetwork(environment.state_size).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(seed=args.seed)
    epsilon = 1.0

    for episode in range(args.episodes):
        state = environment.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randrange(2)
            else:
                with torch.no_grad():
                    values = online(torch.as_tensor(state, dtype=torch.float32, device=device))
                    action = int(values.argmax().item())

            result = environment.step(action)
            replay.append(Transition(state, action, result.reward, result.state, result.done))
            state, done = result.state, result.done

            if len(replay) >= args.batch_size:
                optimize_dqn(replay.sample(args.batch_size), online, target, optimizer, args.gamma, device)

        epsilon = max(0.05, epsilon * 0.98)
        if (episode + 1) % 10 == 0:
            target.load_state_dict(online.state_dict())
        print(f"episode={episode + 1} value={environment.portfolio_value:.2f} epsilon={epsilon:.3f}")

    print(f"held-out observations reserved for evaluation: {len(test)}")


if __name__ == "__main__":
    main()

