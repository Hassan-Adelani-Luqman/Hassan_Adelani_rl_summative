"""Paired comparison of a trained run against a baseline on the same held-out seeds.

Both are evaluated on identical districts (same seeds), so the per-seed differences are
compared directly; the 95% interval comes from a bootstrap over seeds.

    uv run python -m analysis.compare logs/ppo/baseline --baseline greedy
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.common import ROOT
from training.seeds import EVAL_SETS

BASELINE_EPISODES = ROOT / "assets" / "tables" / "baselines_episodes.csv"


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Mean of (a - b) over paired samples, with a 95% bootstrap interval and P(mean diff > 0)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
            "share_of_seeds_better": float((diff > 0).mean()), "n": int(len(diff))}


def baseline_episodes(policy: str, scenario: str = "default", path: Path = BASELINE_EPISODES) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[(df["policy"] == policy) & (df["scenario"] == scenario)].set_index("seed")


def compare_run(log_dir: str | Path, baseline: str = "greedy", set_name: str = "held_out") -> dict[str, Any]:
    """Paired return and infection differences of a run vs a baseline on one seed set."""
    ev = json.loads((Path(log_dir) / "eval.json").read_text())["eval"][set_name]
    seeds, scenario = EVAL_SETS[set_name]
    base = baseline_episodes(baseline, scenario).loc[list(seeds)]
    if len(ev["returns"]) != len(base):
        raise ValueError(f"run has {len(ev['returns'])} episodes, baseline {len(base)}")
    return {
        "set": set_name,
        "baseline": baseline,
        "run_return_mean": float(np.mean(ev["returns"])),
        "baseline_return_mean": float(base["return"].mean()),
        "return": paired_bootstrap(ev["returns"], base["return"]),
        # fewer infections is better, so compare baseline - run
        "infections_saved": paired_bootstrap(base["infections"], ev["infections"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired comparison of a run against a baseline.")
    parser.add_argument("log_dir")
    parser.add_argument("--baseline", default="greedy", choices=["greedy", "random", "do_nothing"])
    parser.add_argument("--set", default="held_out", choices=list(EVAL_SETS))
    args = parser.parse_args(argv)
    result = compare_run(args.log_dir, args.baseline, args.set)
    r, i = result["return"], result["infections_saved"]
    print(f"{args.log_dir} vs {args.baseline} on {args.set} ({r['n']} paired seeds)")
    print(f"  return: {result['run_return_mean']:.1f} vs {result['baseline_return_mean']:.1f} | diff {r['mean_diff']:+.1f} "
          f"[95% CI {r['ci_low']:+.1f}, {r['ci_high']:+.1f}] | better on {r['share_of_seeds_better']:.0%} of seeds")
    print(f"  infections saved vs {args.baseline}: {i['mean_diff']:+.1f} [95% CI {i['ci_low']:+.1f}, {i['ci_high']:+.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
