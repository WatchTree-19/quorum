"""Out-of-fold baselines for the sovereign investment-grade task.

Same philosophy as the Sharia baselines: two models that differ in what they
are allowed to know.

  informed   - GDP per capita, inflation, and the current-account balance. A
               model with the standard sovereign-risk inputs.
  generalist - GDP per capita alone. Rich countries are investment grade, poor
               countries are not; this is the prior a generic model falls back
               on, and it is wrong exactly at the boundary the agencies
               dispute.

Each country is its own group, so grouped folds reduce to plain K-fold; the
grouping is kept so a future multi-year panel (country-years) stays leak-free
without touching this code.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quorum.schema import Item, Prediction


def features(item: Item, mode: str) -> list[float]:
    x = item.inputs

    def val(key):
        v = x.get(key)
        return float(v) if v is not None and np.isfinite(v) else np.nan

    gdp = val("gdp_pc_usd")
    log_gdp = float(np.log(gdp)) if np.isfinite(gdp) and gdp > 0 else np.nan
    if mode == "informed":
        return [log_gdp, val("inflation_pct"), val("current_account_pct_gdp")]
    if mode == "generalist":
        return [log_gdp]
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

    Split items (no majority) are excluded from training but still predicted,
    so the scorecard can observe how the model behaves where there is no
    answer to be had.
    """
    X = np.array([features(it, mode) for it in items], dtype=float)
    y = np.array([it.majority_label if it.majority_label is not None else -1 for it in items])
    groups = np.array([it.metadata["group"] for it in items])

    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)

    probs = np.full(len(items), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        tr = tr[y[tr] >= 0]
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
