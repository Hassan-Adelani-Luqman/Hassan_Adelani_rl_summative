"""Check that the sweep results are complete and comparable before analysing them.

    uv run python -m analysis.validate_runs [--min-runs 10]

For every algorithm: at least ``--min-runs`` finished sweep runs, each with train.csv,
updates.csv and eval.json; one shared frame budget; the same held-out seed set; and the
current environment version. Exits 1 if anything is off.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from analysis.load_runs import load_runs, log_file
from environment.custom_env import ENV_VERSION
from training.common import ALGOS, ROOT
from training.seeds import HELD_OUT_SEEDS, seed_hash
from training.sweeps import SWEEPS


def validate(runs: pd.DataFrame, min_runs: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """Per-algorithm summary table and a list of problems (empty when everything is consistent)."""
    problems: list[str] = []
    rows = []
    expected_hash = seed_hash(HELD_OUT_SEEDS)
    sweep_ids = {algo: {run_id for run_id, _ in SWEEPS[algo]} for algo in ALGOS}
    for algo in ALGOS:
        group = runs[(runs["algo"] == algo) & runs["run_id"].isin(sweep_ids[algo])] if not runs.empty else runs
        missing_files = [r.run_id for r in group.itertuples()
                         if not (log_file(r.log_dir, "train.csv").exists() and log_file(r.log_dir, "updates.csv").exists()
                                 and (Path(r.log_dir) / "eval.json").exists())]
        budgets = sorted(group["cfg_total_frames"].dropna().unique().tolist()) if "cfg_total_frames" in group else []
        hashes = sorted(group["held_out_seed_hash"].dropna().unique().tolist()) if "held_out_seed_hash" in group else []
        versions = sorted(group["env_version"].dropna().unique().tolist()) if "env_version" in group else []
        rows.append({"algo": algo, "runs": len(group), "expected": len(sweep_ids[algo]), "frame_budgets": budgets,
                     "seed_hashes": hashes, "env_versions": versions, "missing_files": missing_files})
        if len(group) < min_runs:
            problems.append(f"{algo}: only {len(group)} finished sweep runs (need >= {min_runs})")
        if missing_files:
            problems.append(f"{algo}: runs missing log files: {missing_files}")
        if len(budgets) > 1:
            problems.append(f"{algo}: mixed frame budgets {budgets}")
        if hashes and hashes != [expected_hash]:
            problems.append(f"{algo}: held-out seed hash {hashes} != {expected_hash}")
        if versions and versions != [ENV_VERSION]:
            problems.append(f"{algo}: env versions {versions} != {ENV_VERSION}")
    budgets_all = sorted({b for row in rows for b in row["frame_budgets"]})
    if len(budgets_all) > 1:
        problems.append(f"algorithms use different frame budgets: {budgets_all}")
    return pd.DataFrame(rows), problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate sweep results before analysis.")
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    parser.add_argument("--models-root", default=str(ROOT / "models"))
    parser.add_argument("--min-runs", type=int, default=10)
    args = parser.parse_args(argv)
    table, problems = validate(load_runs(args.logs_root, args.models_root), args.min_runs)
    print(table.drop(columns="missing_files").to_string(index=False))
    for problem in problems:
        print("PROBLEM:", problem)
    print("OK: sweep results are complete and comparable" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
