"""Out-of-fold baseline classifiers for the Sharia compliance task.

Two baselines, deliberately different in what they are allowed to know:

  informed   - sees the same four leverage ratios the standards use. A model that
               knows the rulebooks. Should be near-perfect except at boundaries.
  generalist - sees only debt/assets and firm size. A competent "financial AI"
               that understands leverage but does not know that three of the
               four rulebooks divide by an averaged market cap. This is the
               realistic case, and it is where the thesis bites.

Predictions are out-of-fold with folds grouped by ticker, so no firm's quarters
appear in both train and test. Confidence is the predicted probability of the
positive (non-compliant) class.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quorum.schema import Item, Prediction


def _safe_log_ratio(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b <= 0 or a <= 0:
        return np.nan
    return float(np.log(a / b))


def features(item: Item, mode: str) -> list[float]:
    x = item.inputs
    if mode == "informed":
        return [
            _safe_log_ratio(x["debt"], x["assets"]),
            _safe_log_ratio(x["debt"], x["mcap"]),
            _safe_log_ratio(x["debt"], x["mcap_avg24"]),
            _safe_log_ratio(x["debt"], x["mcap_avg36"]),
        ]
    if mode == "generalist":
        return [
            _safe_log_ratio(x["debt"], x["assets"]),
            float(np.log(x["assets"])) if x["assets"] and np.isfinite(x["assets"]) and x["assets"] > 0 else np.nan,
            float(np.log(x["mcap"])) if x["mcap"] and np.isfinite(x["mcap"]) and x["mcap"] > 0 else np.nan,
        ]
    raise ValueError(mode)


def out_of_fold_predictions(
    items: list[Item],
    *,
    mode: str,
    abstain_band: tuple[float, float] | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> list[Prediction]:
    """Train/predict out-of-fold; target is the quorum's majority label.

    Split items (no majority) are excluded from training but still predicted, so
    the scorecard can observe how the model behaves where there is no answer.
    """
    X = np.array([features(it, mode) for it in items], dtype=float)
    y = np.array([it.majority_label if it.majority_label is not None else -1 for it in items])
    groups = np.array([it.metadata["group"] for it in items])

    # Impute NaNs with column medians (computed on all rows; features only).
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)

    probs = np.full(len(items), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        tr = tr[y[tr] >= 0]  # train only on items with a majority label
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        model.fit(X[tr], y[tr])
        probs[te] = model.predict_proba(X[te])[:, 1]

    preds: list[Prediction] = []
    for it, p in zip(items, probs):
        if np.isnan(p):
            continue
        if abstain_band is not None and abstain_band[0] <= p <= abstain_band[1]:
            preds.append(Prediction(item_id=it.item_id, label=None, confidence=float(p)))
        else:
            preds.append(Prediction(item_id=it.item_id, label=int(p >= 0.5), confidence=float(p)))
    return preds
