"""The reliability-conditioned scorecard.

The thesis of Quorum is that a single accuracy number hides the thing that
matters: whether a model only gets the cases the experts already agreed on. So
the scorecard reports, side by side,

  * how reliable the ground truth itself is (Krippendorff's alpha across the
    expert labels, and the share of items that are contested),
  * accuracy on unanimous items versus accuracy on majority (contested) items,
    and the gap between them,
  * how the model behaves on split items, where there is no right answer to be
    had: does it abstain, or does it commit confidently anyway,
  * calibration, when confidences are supplied.

The suggested headline for a leaderboard is accuracy on contested (majority)
items, because that is where a financial decision is actually hard. Overall
accuracy is reported too, precisely so the reader can see how much it flatters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grader_reliability import krippendorff_alpha

from quorum.schema import Item, Prediction

STRATA = ("unanimous", "majority", "split")


@dataclass
class StratumResult:
    n: int = 0
    scored: int = 0            # items with a majority label AND a non-abstained prediction
    correct: int = 0
    abstained: int = 0
    confident_commits: int = 0  # non-abstained predictions with |conf-0.5| >= confident_margin
    mean_confidence_in_label: float | None = None  # mean P(predicted label) where available

    @property
    def accuracy(self) -> float | None:
        return (self.correct / self.scored) if self.scored else None

    @property
    def abstain_rate(self) -> float | None:
        return (self.abstained / self.n) if self.n else None

    @property
    def confident_commit_rate(self) -> float | None:
        return (self.confident_commits / self.n) if self.n else None


@dataclass
class Scorecard:
    task: str
    n_items: int
    max_raters: int
    # Ground-truth reliability
    label_alpha: float | None
    contested_share: float          # share of items not unanimous (majority + split)
    split_share: float              # share of items that are even ties
    # Model behaviour by stratum
    strata: dict[str, StratumResult] = field(default_factory=dict)
    # Aggregates
    overall_accuracy: float | None = None   # on all items with a majority label
    reliability_gap: float | None = None    # acc(unanimous) - acc(majority)
    expected_calibration_error: float | None = None
    missing_predictions: int = 0

    @property
    def headline(self) -> float | None:
        """Leaderboard number: accuracy on contested (majority) items."""
        return self.strata["majority"].accuracy if "majority" in self.strata else None


def _ece(confidences: list[float], correct: list[int], bins: int = 10) -> float | None:
    if not confidences:
        return None
    c = np.asarray(confidences); y = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (c >= lo) & (c < hi) if hi < 1.0 else (c >= lo) & (c <= hi)
        if mask.any():
            ece += mask.mean() * abs(c[mask].mean() - y[mask].mean())
    return float(ece)


def compute_scorecard(
    items: list[Item],
    predictions: list[Prediction],
    *,
    confident_margin: float = 0.25,
) -> Scorecard:
    """Score a set of predictions against a set of Quorum items.

    Args:
        items: The evaluation items, each with a quorum of expert labels.
        predictions: The model's predictions, keyed by item_id. Missing items
            are counted as missing, not as wrong.
        confident_margin: A non-abstained prediction with confidence at least
            this far from 0.5 counts as a "confident commit". On split items,
            confident commits are the behaviour the benchmark is designed to
            surface.
    """
    if not items:
        raise ValueError("No items to score.")
    task = items[0].task
    pred_by_id = {p.item_id: p for p in predictions}

    # Ground-truth reliability from the labels alone.
    max_r = max(len(it.labels) for it in items)
    matrix = np.full((len(items), max_r), np.nan)
    for i, it in enumerate(items):
        for j, l in enumerate(it.labels):
            if l is not None:
                matrix[i, j] = l
    alpha = krippendorff_alpha(matrix, level="nominal")

    strata = {s: StratumResult() for s in STRATA}
    conf_all: list[float] = []; corr_all: list[int] = []
    conf_in_label: dict[str, list[float]] = {s: [] for s in STRATA}
    missing = 0
    total_scored = total_correct = 0

    for it in items:
        s = it.agreement
        st = strata[s]
        st.n += 1
        p = pred_by_id.get(it.item_id)
        if p is None:
            missing += 1
            continue
        if p.label is None:
            st.abstained += 1
            continue
        if p.confidence is not None:
            p_label = p.confidence if p.label == 1 else 1.0 - p.confidence
            conf_in_label[s].append(p_label)
            if abs(p.confidence - 0.5) >= confident_margin:
                st.confident_commits += 1
        ref = it.majority_label
        if ref is None:
            continue  # split: no right answer; behaviour recorded above, not scored
        st.scored += 1
        hit = int(p.label == ref)
        st.correct += hit
        total_scored += 1; total_correct += hit
        if p.confidence is not None:
            conf_all.append(p.confidence); corr_all.append(int(ref == 1))

    for s in STRATA:
        vals = conf_in_label[s]
        strata[s].mean_confidence_in_label = float(np.mean(vals)) if vals else None

    n = len(items)
    contested = sum(1 for it in items if it.agreement != "unanimous")
    split = sum(1 for it in items if it.agreement == "split")
    acc_u = strata["unanimous"].accuracy
    acc_m = strata["majority"].accuracy

    return Scorecard(
        task=task,
        n_items=n,
        max_raters=max_r,
        label_alpha=alpha,
        contested_share=contested / n,
        split_share=split / n,
        strata=strata,
        overall_accuracy=(total_correct / total_scored) if total_scored else None,
        reliability_gap=(acc_u - acc_m) if (acc_u is not None and acc_m is not None) else None,
        expected_calibration_error=_ece(conf_all, corr_all),
        missing_predictions=missing,
    )
