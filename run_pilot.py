"""Quorum pilot: the Sharia compliance task, two baselines, out-of-fold.

Usage:
    PYTHONPATH=. python run_pilot.py /path/to/panel_wide.csv
"""

from __future__ import annotations

import sys

from quorum import compute_scorecard, render_report
from quorum.adapters.sharia_panel import load_items
from baselines.ratio_classifier import out_of_fold_predictions
from quorum.affinity import rater_affinity, render_affinity


def main() -> None:
    panel = sys.argv[1]
    items = load_items(panel)
    print(f"Loaded {len(items):,} Quorum items from {panel}\n")

    for mode in ("informed", "generalist"):
        preds = out_of_fold_predictions(items, mode=mode)
        sc = compute_scorecard(items, preds)
        print(render_report(sc, model_name=f"{mode} logistic baseline (out-of-fold)"))
        print(render_affinity(rater_affinity(items, preds), model_name=mode))
        print()

    # An abstaining generalist: refuses to answer when its probability is near 0.5.
    preds = out_of_fold_predictions(items, mode="generalist", abstain_band=(0.35, 0.65))
    sc = compute_scorecard(items, preds)
    print(render_report(sc, model_name="generalist baseline WITH abstention band [0.35, 0.65]"))


if __name__ == "__main__":
    main()
