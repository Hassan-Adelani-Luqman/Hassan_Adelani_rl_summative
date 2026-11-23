"""Pick each algorithm's best sweep configuration for the final multi-seed runs.

    uv run python -m analysis.best_configs          # writes logs/best_configs.json

Best = highest mean held-out return; ties go to the lower standard deviation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from analysis.load_runs import load_runs
from training.common import ALGOS, ROOT
from training.sweeps import BEST_CONFIGS_FILE, SWEEPS


def sweep_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Only the runs that belong to an algorithm's sweep (not test, smoke or final runs)."""
    if runs.empty:
        return runs
    ids = {(algo, run_id) for algo in ALGOS for run_id, _ in SWEEPS[algo]}
    return runs[[(a, r) in ids for a, r in zip(runs["algo"], runs["run_id"], strict=True)]]


def select_best(runs: pd.DataFrame) -> dict[str, str]:
    """{algo: run_id} of the best sweep run per algorithm (by held-out return, then lower std)."""
    best = {}
    sweeps = sweep_runs(runs)
    for algo in ALGOS:
        group = sweeps[sweeps["algo"] == algo]
        if group.empty:
            continue
        ranked = group.sort_values(["held_out_return_mean", "held_out_return_std"], ascending=[False, True])
        best[algo] = str(ranked.iloc[0]["run_id"])
    return best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick the best sweep configuration per algorithm.")
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    parser.add_argument("--models-root", default=str(ROOT / "models"))
    parser.add_argument("--out", default=str(BEST_CONFIGS_FILE))
    args = parser.parse_args(argv)
    runs = load_runs(args.logs_root, args.models_root)
    best = select_best(runs)
    if not best:
        print("no finished sweep runs found")
        return 1
    for algo, run_id in best.items():
        row = runs[(runs["algo"] == algo) & (runs["run_id"] == run_id)].iloc[0]
        print(f"{algo:<10} {run_id:<22} held-out return {row['held_out_return_mean']:.1f} ± {row['held_out_return_std']:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
