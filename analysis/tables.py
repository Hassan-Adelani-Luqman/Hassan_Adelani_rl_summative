"""Write result tables as CSV (for analysis) and Markdown (for the README and report)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def readable_float(value: float) -> str:
    """Report-friendly numbers: no scientific notation for returns, '—' for missing values."""
    if math.isnan(value):
        return "—"
    magnitude = abs(value)
    if magnitude >= 1 and float(value).is_integer():
        return str(int(value))
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return "0" if value == 0 else f"{value:.3g}"


def to_markdown(df: pd.DataFrame, float_format: str | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependencies).

    ``float_format`` is a format string; by default numbers use ``readable_float``.
    """

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return readable_float(value) if float_format is None else float_format.format(value)
        return str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows]) + "\n"


def write_table(df: pd.DataFrame, stem: str | Path, float_format: str | None = None) -> tuple[Path, Path]:
    """Write ``<stem>.csv`` and ``<stem>.md``; returns both paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = stem.with_suffix(".csv"), stem.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df, float_format))
    return csv_path, md_path
