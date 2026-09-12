"""Loaders that turn labelled CSV datasets into rating arrays.

Kept separate from the metrics so the core library has no I/O concerns. The
loader is generic: point it at any CSV with one column per grader.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np


def load_rater_csv(
    path: str | Path,
    *,
    rater_prefix: str = "human_score",
    min_raters: int = 1,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Load a CSV into an (n_items x n_raters) float array plus the raw rows.

    Columns whose name starts with ``rater_prefix`` are treated as grader labels.
    A leading comment line beginning with ``#`` (as PyRIT's scorer_evals CSVs use)
    is skipped. Missing grader cells become NaN. Rows with fewer than
    ``min_raters`` present labels are dropped from the array but still returned in
    the row list, so callers can align other columns if they wish.

    Returns the ratings array and the list of kept rows.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 1 if lines and lines[0].startswith("#") else 0
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    fieldnames = reader.fieldnames or []
    rater_cols = [c for c in fieldnames if c.startswith(rater_prefix)]
    if not rater_cols:
        raise ValueError(f"No columns starting with '{rater_prefix}' in {path.name}.")

    ratings: list[list[float]] = []
    kept_rows: list[dict[str, str]] = []
    for row in reader:
        values: list[float] = []
        for col in rater_cols:
            cell = row.get(col)
            values.append(float(cell) if cell not in (None, "") else float("nan"))
        present = sum(1 for v in values if not np.isnan(v))
        if present >= min_raters:
            ratings.append(values)
            kept_rows.append(row)

    return np.array(ratings, dtype=float), kept_rows
