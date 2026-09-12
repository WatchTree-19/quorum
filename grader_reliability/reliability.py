"""Grader-reliability metrics: how much do human graders agree, and does a
model only look accurate where the graders happened to agree?

Two ideas, both dataset-agnostic:

1. Inter-rater reliability. Given several graders' labels per item, report
   exact agreement, Krippendorff's alpha (nominal or ordinal), and the share of
   items whose binary verdict rests on a single grader flipping their vote.

2. Accuracy conditioned on agreement. Given a model's predictions on the same
   items, report the model's accuracy separately for items the graders were
   unanimous on, split (a tie), and everything in between. A model that scores
   well overall but poorly on contested items is not as good as its headline
   number: it is riding the easy, unanimous cases.

Input is a 2D array of ratings, shape (n_items, n_raters), with NaN for a rater
who did not label an item. Nothing here is specific to harm scoring; it works
for any labelled dataset with more than one grader per item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AgreementLevel = Literal["unanimous", "majority", "tie"]
LevelOfMeasurement = Literal["nominal", "ordinal"]

DEFAULT_BINARY_THRESHOLD = 0.5


@dataclass
class ReliabilityReport:
    n_items: int
    max_raters: int
    krippendorff_alpha: float | None
    exact_agreement_rate: float
    binary_contested_rate: float
    single_vote_rate: float
    # Present only when predictions are supplied.
    accuracy_overall: float | None
    accuracy_by_agreement: dict[AgreementLevel, dict[str, float]] | None


def _row_values(ratings: np.ndarray) -> list[np.ndarray]:
    """Non-NaN values per item, as a list of 1D arrays."""
    return [row[~np.isnan(row)] for row in np.asarray(ratings, dtype=float)]


def binarize(ratings: np.ndarray, threshold: float = DEFAULT_BINARY_THRESHOLD) -> np.ndarray:
    """Map ratings to 0/1 at a threshold, preserving NaN for missing entries."""
    ratings = np.asarray(ratings, dtype=float)
    out = np.where(ratings >= threshold, 1.0, 0.0)
    out[np.isnan(ratings)] = np.nan
    return out


def exact_agreement_rate(ratings: np.ndarray) -> float:
    """Fraction of items on which every present grader gave the identical value."""
    rows = [row for row in _row_values(ratings) if row.size >= 2]
    if not rows:
        return 0.0
    return float(np.mean([np.unique(row).size == 1 for row in rows]))


def agreement_level(binary_row: np.ndarray) -> AgreementLevel:
    """Classify one item's binary grader votes as unanimous, tie, or majority."""
    votes = binary_row[~np.isnan(binary_row)]
    n = votes.size
    positives = int(votes.sum())
    if positives == 0 or positives == n:
        return "unanimous"
    if n % 2 == 0 and positives == n // 2:
        return "tie"
    return "majority"


def _majority_label(binary_row: np.ndarray) -> float | None:
    """The label a majority of graders gave, or None on a tie."""
    votes = binary_row[~np.isnan(binary_row)]
    n = votes.size
    positives = int(votes.sum())
    if n % 2 == 0 and positives == n // 2:
        return None
    return 1.0 if positives * 2 > n else 0.0


def binary_contested_rate(ratings: np.ndarray, threshold: float = DEFAULT_BINARY_THRESHOLD) -> float:
    """Fraction of items whose binary verdict is not unanimous."""
    binary = binarize(ratings, threshold)
    rows = [row[~np.isnan(row)] for row in binary]
    rows = [row for row in rows if row.size >= 2]
    if not rows:
        return 0.0
    return float(np.mean([agreement_level(row) != "unanimous" for row in rows]))


def single_vote_rate(ratings: np.ndarray, threshold: float = DEFAULT_BINARY_THRESHOLD) -> float:
    """Fraction of items where the binary majority is decided by exactly one flip."""
    binary = binarize(ratings, threshold)
    rows = [row[~np.isnan(row)] for row in binary]
    rows = [row for row in rows if row.size >= 2]
    if not rows:
        return 0.0
    flags = []
    for row in rows:
        n = row.size
        positives = int(row.sum())
        flags.append(positives == 1 or positives == n - 1)
    return float(np.mean(flags))


def _delta_squared(level: LevelOfMeasurement, values: np.ndarray, marginals: dict[float, float]):
    """Return a metric(c, k) -> squared distance function for the given scale."""
    if level == "nominal":
        def metric(c: float, k: float) -> float:
            return 0.0 if c == k else 1.0

        return metric

    # Ordinal: distance uses the marginal counts of the ranks between c and k.
    ordered = sorted(values)
    rank = {v: i for i, v in enumerate(ordered)}
    counts = np.array([marginals[v] for v in ordered], dtype=float)

    def metric(c: float, k: float) -> float:
        lo, hi = sorted((rank[c], rank[k]))
        between = counts[lo:hi + 1].sum()
        return (between - (counts[lo] + counts[hi]) / 2.0) ** 2

    return metric


def krippendorff_alpha(ratings: np.ndarray, level: LevelOfMeasurement = "ordinal") -> float | None:
    """Krippendorff's alpha over an (n_items x n_raters) array with NaN for missing.

    Returns None if fewer than two paired ratings exist. Implemented with the
    coincidence-matrix method so it handles missing data and any number of
    raters; supports nominal and ordinal scales.
    """
    rows = [row for row in _row_values(ratings) if row.size >= 2]
    if not rows:
        return None

    values = sorted({float(v) for row in rows for v in row})
    if len(values) < 2:
        return 1.0  # all graders gave the same single value everywhere

    # Coincidence matrix o[c][k]: pairable observations within items.
    index = {v: i for i, v in enumerate(values)}
    size = len(values)
    o = np.zeros((size, size), dtype=float)
    for row in rows:
        m = row.size
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                o[index[float(row[a])]][index[float(row[b])]] += 1.0 / (m - 1)

    marginal_arr = o.sum(axis=1)
    marginals = {v: marginal_arr[index[v]] for v in values}
    n = marginal_arr.sum()
    if n <= 1:
        return None

    metric = _delta_squared(level, np.array(values), marginals)

    observed = 0.0
    expected = 0.0
    for ci in range(size):
        for ki in range(ci + 1, size):
            d2 = metric(values[ci], values[ki])
            observed += o[ci][ki] * d2
            expected += marginal_arr[ci] * marginal_arr[ki] * d2
    if expected == 0:
        return 1.0
    return float(1.0 - (n - 1) * observed / expected)


def accuracy_by_agreement(
    ratings: np.ndarray,
    predictions: np.ndarray,
    threshold: float = DEFAULT_BINARY_THRESHOLD,
) -> dict[AgreementLevel, dict[str, float]]:
    """Model accuracy split by how much the graders agreed.

    The reference label is the graders' binary majority. Tie items have no
    majority label, so they are counted but excluded from the accuracy figure.
    Predictions are binarized at the same threshold.
    """
    binary = binarize(ratings, threshold)
    pred_binary = (np.asarray(predictions, dtype=float) >= threshold).astype(int)

    buckets: dict[AgreementLevel, list[int]] = {"unanimous": [], "majority": [], "tie": []}
    for i, row in enumerate(binary):
        present = row[~np.isnan(row)]
        if present.size < 2:
            continue
        level = agreement_level(present)
        reference = _majority_label(present)
        if reference is None:
            buckets[level].append(-1)  # tie: counted, not scored
        else:
            buckets[level].append(int(pred_binary[i] == int(reference)))

    report: dict[AgreementLevel, dict[str, float]] = {}
    for level, hits in buckets.items():
        scored = [h for h in hits if h >= 0]
        report[level] = {
            "n": float(len(hits)),
            "accuracy": float(np.mean(scored)) if scored else float("nan"),
        }
    return report


def reliability_report(
    ratings: np.ndarray,
    predictions: np.ndarray | None = None,
    *,
    threshold: float = DEFAULT_BINARY_THRESHOLD,
    level: LevelOfMeasurement = "ordinal",
) -> ReliabilityReport:
    """Compute the full reliability picture for one labelled dataset."""
    ratings = np.asarray(ratings, dtype=float)
    accuracy_overall: float | None = None
    by_level: dict[AgreementLevel, dict[str, float]] | None = None

    if predictions is not None:
        by_level = accuracy_by_agreement(ratings, predictions, threshold)
        binary = binarize(ratings, threshold)
        pred_binary = (np.asarray(predictions, dtype=float) >= threshold).astype(int)
        scored = []
        for i, row in enumerate(binary):
            reference = _majority_label(row[~np.isnan(row)]) if row[~np.isnan(row)].size >= 2 else None
            if reference is not None:
                scored.append(int(pred_binary[i] == int(reference)))
        accuracy_overall = float(np.mean(scored)) if scored else None

    return ReliabilityReport(
        n_items=sum(1 for row in _row_values(ratings) if row.size >= 2),
        max_raters=int(np.max([row.size for row in _row_values(ratings)], initial=0)),
        krippendorff_alpha=krippendorff_alpha(ratings, level),
        exact_agreement_rate=exact_agreement_rate(ratings),
        binary_contested_rate=binary_contested_rate(ratings, threshold),
        single_vote_rate=single_vote_rate(ratings, threshold),
        accuracy_overall=accuracy_overall,
        accuracy_by_agreement=by_level,
    )
