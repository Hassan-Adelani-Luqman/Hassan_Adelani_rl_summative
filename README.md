# Afya Sentinel

Reinforcement-learning dispatch for a mobile malaria early-detection van serving 8 villages during the rainy season. The agent sees only noisy, delayed fever reports from community health workers. It has to decide where to drive, when to screen, when to hand out bed nets, and when to go back to the hub to restock. Four TorchRL agents (DQN, REINFORCE, A2C, PPO) are trained and compared on the same environment.

> Work in progress. This README will be completed before submission.

## Quick start

```bash
uv sync
uv run play.py      # watch the best trained agent (3D viewer + terminal log)
uv run main.py      # same as play.py; `uv run main.py --help` lists the other commands
uv run pytest       # run the test suite
```

Requires [uv](https://docs.astral.sh/uv/). Linux and Windows install CPU-only PyTorch; no GPU is needed.
