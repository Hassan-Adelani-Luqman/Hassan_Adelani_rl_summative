"""Publish the final models that play.py and the API load by default.

    uv run python -m analysis.publish_best

For each algorithm, the final run (best config × seeds) with the highest return on the
training-seed evaluation set is copied to models/dqn/best/ or models/pg/<algo>/best/
(model.pt, config.json and its eval.json). models/best.json names the overall best
algorithm, chosen by the mean of that return across its seeds (more robust than one lucky
seed). The held-out seeds are reported but never used to choose between seeds, so the
published model's held-out score is not inflated by the choice.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from analysis.load_runs import load_runs, model_dir
from training.common import ALGOS, ROOT

SELECT_BY = "train_return_mean"  # evaluation on training seeds; held-out seeds stay out of model selection


def publish(logs_root: Path = ROOT / "logs", models_root: Path = ROOT / "models") -> dict[str, object]:
    runs = load_runs(logs_root, models_root)
    final = runs[runs["run_id"].astype(str).str.startswith("final_")] if not runs.empty else runs
    if final.empty:
        raise SystemExit("no final runs found; pull the final shard first")
    published: dict[str, dict[str, object]] = {}
    for algo in ALGOS:
        group = final[final["algo"] == algo]
        if group.empty:
            continue
        best = group.sort_values(SELECT_BY, ascending=False).iloc[0]
        source = model_dir(models_root, algo, best["run_id"])
        target = model_dir(models_root, algo, "best")
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        for name in ("model.pt", "config.json"):
            shutil.copy2(source / name, target / name)
        shutil.copy2(Path(best["log_dir"]) / "eval.json", target / "eval.json")
        published[algo] = {"run_id": best["run_id"], "seed": int(best["seed"]),
                           "train_return": round(float(best[SELECT_BY]), 1),
                           "train_return_mean_over_seeds": round(float(group[SELECT_BY].mean()), 1),
                           "held_out_return": round(float(best["held_out_return_mean"]), 1),
                           "held_out_return_mean_over_seeds": round(float(group["held_out_return_mean"].mean()), 1),
                           "seeds": len(group)}
    overall = max(published, key=lambda a: published[a]["train_return_mean_over_seeds"])
    summary = {"algo": overall,
               "rule": "per algorithm, the final seed with the highest return on the training-seed evaluation set; "
                       "overall, the algorithm with the highest mean of that return across its seeds",
               "models": published}
    (models_root / "best.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy the best final model of each algorithm to <family>/best.")
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    parser.add_argument("--models-root", default=str(ROOT / "models"))
    args = parser.parse_args(argv)
    summary = publish(Path(args.logs_root), Path(args.models_root))
    for algo, info in summary["models"].items():
        print(f"{algo:<10} best = {info['run_id']:<26} train {info['train_return']:7.1f} "
              f"(mean over {info['seeds']} seeds {info['train_return_mean_over_seeds']:7.1f})  "
              f"held-out {info['held_out_return']:7.1f} (mean {info['held_out_return_mean_over_seeds']:7.1f})")
    print(f"overall best (play.py default): {summary['algo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
