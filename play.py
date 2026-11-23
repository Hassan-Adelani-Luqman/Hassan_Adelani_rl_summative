"""Afya Sentinel demo: watch a trained agent dispatch the malaria early-detection van.

    uv run play.py                     # best model, default scenario, opens the 3D viewer
    uv run play.py --algo dqn --seed 7 --scenario heavy_rain
    uv run play.py --no-browser        # verbose terminal log only

The terminal prints a step-by-step log while frames stream to the Three.js viewer
over a WebSocket. Press H in the viewer to switch between Agent View and Reality View.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

ALGOS = ("best", "dqn", "reinforce", "a2c", "ppo")
SCENARIOS = ("default", "heavy_rain", "shuffled_hotspots")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="play.py",
        description="Watch a trained Afya Sentinel agent in the 3D viewer, with a verbose terminal log.",
    )
    parser.add_argument(
        "--algo",
        choices=ALGOS,
        default="best",
        help="which trained model to load (default: the best of the four)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="episode seed (default: the first held-out evaluation seed)",
    )
    parser.add_argument("--scenario", choices=SCENARIOS, default="default", help="district scenario")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to play")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer playback speed multiplier")
    parser.add_argument(
        "--compare-greedy",
        action="store_true",
        help="also run the greedy baseline on the same seed and show the score difference",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="don't start the viewer; print the terminal log only",
    )
    parser.add_argument("--port", type=int, default=8000, help="local port for the viewer server")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise SystemExit("play.py: the simulation and viewer are not built yet (Phase 7).")


if __name__ == "__main__":
    sys.exit(main())
