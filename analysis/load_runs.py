"""Load finished training runs into DataFrames (used for validation, tables and figures).

A finished run has ``logs/<algo>/<run_id>/eval.json`` plus its model's ``config.json``
(``models/dqn/<run_id>/`` or ``models/pg/<algo>/<run_id>/``).

    uv run python -m analysis.load_runs            # one-line summary per run
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from training.common import ALGOS, ROOT

SUMMARY_METRICS = ("return_mean", "return_std", "infections_mean", "infections_averted_mean", "outbreak_rate",
                   "stranded_rate", "seed_hash", "policy_mode", "invalid_actions")


def model_dir(models_root: Path, algo: str, run_id: str) -> Path:
    return models_root / ("dqn" if algo == "dqn" else Path("pg") / algo) / run_id


def find_runs(logs_root: str | Path = ROOT / "logs") -> list[Path]:
    """Run log directories that contain an eval.json (only logs/<algo>/<run_id>, not smoke or shards)."""
    logs_root = Path(logs_root)
    return sorted(p.parent for algo in ALGOS for p in (logs_root / algo).glob("*/eval.json"))


def load_run(log_dir: Path, models_root: str | Path = ROOT / "models") -> dict[str, Any]:
    """One flat record: identity, stats, per-set metrics (``<set>_<metric>``) and config fields."""
    ev = json.loads((log_dir / "eval.json").read_text())
    record: dict[str, Any] = {k: ev[k] for k in ("algo", "run_id", "seed", "env_version")}
    for key in ("frames", "updates", "episodes", "train_seconds", "fps"):
        record[key] = ev.get(key)
    argmax = ev.get("argmax_held_out")
    if argmax:
        record["argmax_return_mean"] = argmax["return_mean"]
        record["argmax_infections_averted_mean"] = argmax["infections_averted_mean"]
    for set_name, metrics in ev["eval"].items():
        for metric in SUMMARY_METRICS:
            if metric in metrics:
                record[f"{set_name}_{metric}"] = metrics[metric]
    cfg_path = model_dir(Path(models_root), ev["algo"], ev["run_id"]) / "config.json"
    if cfg_path.exists():
        for key, value in json.loads(cfg_path.read_text())["config"].items():
            if key not in ("algo", "run_id", "seed", "logs_root", "models_root"):
                record[f"cfg_{key}"] = tuple(value) if isinstance(value, list) else value
    record["log_dir"] = str(log_dir)
    return record


def load_runs(logs_root: str | Path = ROOT / "logs", models_root: str | Path = ROOT / "models") -> pd.DataFrame:
    """All finished runs, one row each."""
    return pd.DataFrame([load_run(d, models_root) for d in find_runs(logs_root)])


def log_file(log_dir: str | Path, name: str) -> Path:
    """``name`` (e.g. train.csv) or its compacted ``name.gz`` form, whichever exists."""
    plain = Path(log_dir) / name
    return plain if plain.exists() else plain.with_name(name + ".gz")


def load_train(log_dir: str | Path) -> pd.DataFrame:
    """Per-episode training log: frames, episode, return, decisions."""
    return pd.read_csv(log_file(log_dir, "train.csv"))


def load_updates(log_dir: str | Path) -> pd.DataFrame:
    """Per-update training log: losses, entropy or epsilon, gradient norm."""
    return pd.read_csv(log_file(log_dir, "updates.csv"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise finished training runs.")
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    parser.add_argument("--models-root", default=str(ROOT / "models"))
    args = parser.parse_args(argv)
    runs = load_runs(args.logs_root, args.models_root)
    if runs.empty:
        print("no finished runs found")
        return 0
    cols = ["algo", "run_id", "seed", "frames", "fps", "held_out_return_mean", "held_out_infections_averted_mean"]
    print(runs[[c for c in cols if c in runs]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
