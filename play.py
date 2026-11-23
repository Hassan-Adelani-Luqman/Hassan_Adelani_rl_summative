"""Afya Sentinel demo: watch a trained agent dispatch the malaria early-detection van in Bugesera.

    uv run play.py                     # best model, opens the 3D viewer, step log in this terminal
    uv run play.py --algo dqn --seed 1000007 --scenario heavy_rain
    uv run play.py --no-browser        # terminal log only

The terminal prints one line per decision (with the agent's rationale) while frames stream
to the Three.js viewer over a WebSocket. In the viewer: H switches Agent View / Reality
View, Space pauses, +/- change speed, N moves on to the next district.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlencode

from environment.custom_env import SCENARIOS

ALGOS = ("best", "dqn", "reinforce", "a2c", "ppo")


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
        help="district seed (default: the first held-out evaluation seed)",
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="default", help="district scenario")
    parser.add_argument("--episodes", type=int, default=1, help="episodes to play with --no-browser")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer playback speed multiplier")
    parser.add_argument(
        "--compare-greedy",
        action="store_true",
        help="print the greedy and do-nothing baselines on the same district after each episode",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="don't start the viewer; print the terminal log only",
    )
    parser.add_argument("--port", type=int, default=8000, help="local port for the viewer server")
    parser.add_argument("--model-dir", default=None, help="load this model directory instead of the committed one")
    return parser


def _policy(args: argparse.Namespace):
    from api.app import resolve_model_dir
    from training.common import load_policy

    model_dir = Path(args.model_dir) if args.model_dir else resolve_model_dir(args.algo)
    return load_policy(model_dir)


def run_terminal(args: argparse.Namespace) -> int:
    """Play episodes straight in the terminal (no server, no browser)."""
    from environment.rendering import EpisodeRunner, reference_runs, terminal_header, terminal_line, terminal_summary
    from training.seeds import HELD_OUT_SEEDS

    policy = _policy(args)
    colour = sys.stdout.isatty()
    seed = args.seed if args.seed is not None else HELD_OUT_SEEDS[0]
    for episode in range(args.episodes):
        runner = EpisodeRunner(policy, seed + episode, args.scenario)
        runner.reset()
        print(terminal_header(policy.name, seed + episode, args.scenario), flush=True)
        while not runner.done:
            print(terminal_line(runner.step(), colour), flush=True)
        reference = reference_runs(seed + episode, args.scenario) if args.compare_greedy else None
        print(terminal_summary(runner.summary(), reference), flush=True)
    return 0


def run_viewer(args: argparse.Namespace) -> int:
    """Serve the viewer, open the browser, and print the step log as the episode plays."""
    import uvicorn

    from api.app import Settings, create_app
    from training.seeds import HELD_OUT_SEEDS

    policy_dir = Path(args.model_dir) if args.model_dir else None
    _policy(args)  # fail fast (with a clear message) if no model is available
    settings = Settings(default_algo=args.algo, verbose=True, colour=sys.stdout.isatty(),
                        compare_baselines=True, default_speed=args.speed)
    if policy_dir:
        settings.model_dirs[args.algo] = policy_dir
    seed = args.seed if args.seed is not None else HELD_OUT_SEEDS[0]
    query = urlencode({"algo": args.algo, "seed": seed, "scenario": args.scenario, "speed": args.speed})
    url = f"http://127.0.0.1:{args.port}/?{query}"
    print(f"Afya Sentinel viewer: {url}\n"
          "  H = Agent / Reality view   Space = pause   +/- = speed   N = next district   Ctrl+C = quit", flush=True)
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_terminal(args) if args.no_browser else run_viewer(args)
    except FileNotFoundError as err:
        raise SystemExit(f"play.py: {err}") from err


if __name__ == "__main__":
    sys.exit(main())
