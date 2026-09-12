"""Quorum: evaluating financial AI under contested ground truth.

Every item carries a quorum of independent expert labels rather than a single
gold answer. A model is scored not by one accuracy number but by how it behaves
where the experts agree, where they lean, and where they split.

Public API:
    Item, Prediction            - data model (schema.py)
    compute_scorecard, Scorecard - the reliability-conditioned scorecard (scorecard.py)
    render_report                - human-readable report (report.py)
"""

from quorum.schema import Item, Prediction
from quorum.scorecard import Scorecard, compute_scorecard
from quorum.report import render_report

__all__ = ["Item", "Prediction", "Scorecard", "compute_scorecard", "render_report"]
