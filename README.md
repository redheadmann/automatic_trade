# automatic_trade

Research code for a deep-Q-learning cryptocurrency trading agent.

The repository currently provides a reproducible starting point rather than a live trading system:

- CoinAPI-style OHLCV CSV loading and chronological train/test splitting
- relative-price and standardized-volume features
- a deterministic long/short trading environment with transaction costs
- a PyTorch Q-network and replay buffer
- a command-line DQN training script
- unit tests for data preparation and portfolio accounting

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data

Provide a CSV containing these columns (case-insensitive):

```text
time_period_start, price_open, price_high, price_low, price_close, volume_traded
```

CoinAPI exports use this schema. API keys and raw private datasets should not be committed.

## Train

```bash
python scripts/train_dqn.py path/to/bitcoin.csv --episodes 100
```

The script reports final portfolio value for each training episode. A serious evaluation should compare an untouched test period against buy-and-hold and always-short baselines.

## Test

```bash
python -m unittest discover -s tests -v
```

## Status and safety

This is experimental backtesting code. It does not connect to an exchange or place real orders. Backtest performance is not evidence of future returns.
