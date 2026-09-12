"""Quorum pilot two: sovereign investment grade, three agencies as the quorum.

Usage:
    PYTHONPATH=. python run_pilot_sovereign.py data/sovereign_panel.csv
"""

from __future__ import annotations

import sys

from quorum import compute_scorecard, render_report
from quorum.adapters.sovereign_ratings import load_items
from baselines.macro_classifier import out_of_fold_predictions
from quorum.affinity import rater_affinity, render_affinity


def main() -> None:
    panel = sys.argv[1] if len(sys.argv) > 1 else "data/sovereign_panel.csv"
    items = load_items(panel)
    n3 = sum(1 for it in items if it.n_raters == 3)
    print(f"Loaded {len(items)} sovereigns ({n3} rated by all three agencies)\n")

    contested = [it for it in items if it.agreement != "unanimous"]
    print("Contested sovereigns (agencies disagree on investment grade):")
    for it in sorted(contested, key=lambda i: i.item_id):
        r = it.metadata["ratings"]
        print(f"  {it.metadata['country']:22} S&P={r['SP'] or '-':5} Moody's={r['MOODYS'] or '-':5} "
              f"Fitch={r['FITCH'] or '-':5} -> {it.agreement}")
    print()

    for mode in ("informed", "generalist"):
        preds = out_of_fold_predictions(items, mode=mode)
        sc = compute_scorecard(items, preds)
        print(render_report(sc, model_name=f"{mode} logistic baseline (out-of-fold)"))
        print(render_affinity(rater_affinity(items, preds), model_name=mode))
        print()

    preds = out_of_fold_predictions(items, mode="generalist", abstain_band=(0.35, 0.65))
    sc = compute_scorecard(items, preds)
    print(render_report(sc, model_name="generalist baseline WITH abstention band [0.35, 0.65]"))


if __name__ == "__main__":
    main()
