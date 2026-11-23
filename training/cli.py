"""Command-line entry points behind ``main.py train`` and ``main.py evaluate``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from training.common import ALGOS, RunConfig, default_config, evaluate_policy, load_policy
from training.seeds import EVAL_SETS


def train(cfg: RunConfig) -> dict[str, Any]:
    """Train one run with the right algorithm module."""
    if cfg.algo == "dqn":
        from training.dqn_training import train as train_dqn

        return train_dqn(cfg)
    from training.pg_training import train as train_pg

    return train_pg(cfg)


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``["lr=1e-3", "double_dqn=true", "hidden=[64,64]"]`` -> typed overrides."""
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key] = tuple(value) if isinstance(value, list) else value
    return overrides


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="main.py train", description="Train one algorithm with one configuration.")
    parser.add_argument("--algo", choices=ALGOS, required=True)
    parser.add_argument("--run-id", default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="total frames (default: the configured budget)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config field, e.g. --set lr=1e-3 (repeatable)")
    args = parser.parse_args(argv)
    overrides = {"run_id": args.run_id, "seed": args.seed, **parse_overrides(args.overrides)}
    if args.frames:
        overrides["total_frames"] = args.frames
    result = train(default_config(args.algo, **overrides))
    held_out = result["eval"]["held_out"]
    print(f"{args.algo}:{args.run_id} frames {result['stats']['frames']} in {result['stats']['train_seconds']}s "
          f"({result['stats']['fps']} fps) | held-out return {held_out['return_mean']:.1f} ± {held_out['return_std']:.1f}, "
          f"infections averted {held_out['infections_averted_mean']:.1f}")
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="main.py evaluate", description="Evaluate a saved model on the seed sets.")
    parser.add_argument("model_dir", help="e.g. models/pg/ppo/baseline")
    parser.add_argument("--sets", nargs="+", default=list(EVAL_SETS), choices=list(EVAL_SETS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N seeds of each set")
    args = parser.parse_args(argv)
    results = evaluate_policy(load_policy(args.model_dir), args.sets, args.limit)
    print(f"{'set':<18}{'return':>18}{'infections':>12}{'averted':>9}{'outbreak':>10}{'stranded':>10}")
    for name, r in results.items():
        print(f"{name:<18}{r['return_mean']:>10.1f} ± {r['return_std']:<6.1f}{r['infections_mean']:>11.1f}"
              f"{r['infections_averted_mean']:>9.1f}{r['outbreak_rate']:>10.0%}{r['stranded_rate']:>10.0%}")
    return 0
