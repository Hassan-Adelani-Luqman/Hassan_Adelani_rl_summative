"""Run one shard of training jobs in parallel worker processes.

    uv run python -m training.run_sweep --shard ppo --workers 4
    uv run python -m training.run_sweep --shard test --workers 3 --logs-root logs/smoke --models-root models/smoke

Each worker runs single-threaded torch. The runner is resumable: a run whose
``eval.json`` already exists is skipped, so an interrupted shard (or a Kaggle session
that timed out) picks up where it stopped. Progress is recorded in
``<logs-root>/shards/<shard>.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from training.common import RunConfig, run_dirs
from training.sweeps import SHARDS, shard_configs


def is_done(cfg: RunConfig) -> bool:
    return (run_dirs(cfg)[0] / "eval.json").exists()


def run_job(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Train one run in a worker process; returns a small summary (never raises)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from training.cli import train

    cfg = RunConfig.from_dict(cfg_dict)
    started = time.time()
    try:
        result = train(cfg)
        held_out = result["eval"].get("held_out", next(iter(result["eval"].values())))
        return {"run": f"{cfg.algo}/{cfg.run_id}", "status": "done", "seconds": round(time.time() - started, 1),
                "fps": result["stats"]["fps"], "return_mean": held_out["return_mean"]}
    except Exception:  # keep the shard going; the failure is recorded in the manifest
        return {"run": f"{cfg.algo}/{cfg.run_id}", "status": "failed", "seconds": round(time.time() - started, 1),
                "error": traceback.format_exc()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="main.py sweep", description="Run a shard of training jobs (resumable).")
    parser.add_argument("--shard", choices=SHARDS, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=None, help="override the frame budget of every run")
    parser.add_argument("--limit", type=int, default=None, help="run at most N pending jobs")
    parser.add_argument("--logs-root", default=None)
    parser.add_argument("--models-root", default=None)
    args = parser.parse_args(argv)

    common: dict[str, Any] = {}
    if args.frames:
        common["total_frames"] = args.frames
    if args.logs_root:
        common["logs_root"] = str(Path(args.logs_root).resolve())
    if args.models_root:
        common["models_root"] = str(Path(args.models_root).resolve())
    configs = shard_configs(args.shard, **common)
    pending = [cfg for cfg in configs if not is_done(cfg)]
    skipped = len(configs) - len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]

    logs_root = Path(configs[0].logs_root)
    manifest_path = logs_root / "shards" / f"{args.shard}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": {}}
    manifest.update(shard=args.shard, total=len(configs), last_started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"shard {args.shard}: {len(configs)} runs, {skipped} already done, running {len(pending)} "
          f"with {args.workers} workers", flush=True)

    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_job, cfg.to_dict()): cfg for cfg in pending}
        for future in as_completed(futures):
            summary = future.result()
            manifest["runs"][summary["run"]] = summary
            manifest_path.write_text(json.dumps(manifest, indent=2))
            failures += summary["status"] != "done"
            detail = (f"return {summary['return_mean']:.1f}, {summary['fps']} fps" if summary["status"] == "done"
                      else summary["error"].strip().splitlines()[-1])
            print(f"  [{summary['status']}] {summary['run']} ({summary['seconds']}s): {detail}", flush=True)
    done = sum(is_done(cfg) for cfg in configs)
    print(f"shard {args.shard}: {done}/{len(configs)} runs complete, {failures} failed this time", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
