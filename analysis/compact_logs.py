"""Compact finished run logs for the repository: 5 significant figures, gzip.

    uv run python -m analysis.compact_logs            # every run under logs/<algo>/<run_id>

train.csv and updates.csv become train.csv.gz and updates.csv.gz (about a fifth of the
size); the loaders read either form. eval.json is left as it is.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from training.common import ALGOS, ROOT

LOG_FILES = ("train.csv", "updates.csv")
SIGNIFICANT = 5


def compact_run(log_dir: Path) -> int:
    """Replace a run's CSV logs with rounded, gzipped copies; returns how many files changed."""
    changed = 0
    for name in LOG_FILES:
        path = log_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].map(lambda v: float(f"{v:.{SIGNIFICANT}g}"))
        df.to_csv(log_dir / f"{name}.gz", index=False)
        path.unlink()
        changed += 1
    return changed


def compact_all(logs_root: Path) -> int:
    return sum(compact_run(d) for algo in ALGOS for d in sorted((logs_root / algo).glob("*")) if d.is_dir())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round and gzip run logs.")
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    args = parser.parse_args(argv)
    print(f"compacted {compact_all(Path(args.logs_root))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
