"""Write result tables as CSV (for analysis) and Markdown (for the README and report)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame, float_format: str = "{:.2f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependencies)."""

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return float_format.format(value)
        return str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows]) + "\n"


def write_table(df: pd.DataFrame, stem: str | Path, float_format: str = "{:.2f}") -> tuple[Path, Path]:
    """Write ``<stem>.csv`` and ``<stem>.md``; returns both paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = stem.with_suffix(".csv"), stem.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df, float_format))
    return csv_path, md_path
