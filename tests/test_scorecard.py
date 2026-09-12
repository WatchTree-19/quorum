"""Tests for the Quorum schema and scorecard. Offline, deterministic."""

import math

import pytest

from quorum.schema import Item, Prediction
from quorum.scorecard import compute_scorecard


def mk(item_id, labels, task="t"):
    return Item(item_id=item_id, task=task, inputs={}, labels=tuple(labels),
                rater_ids=tuple(f"r{i}" for i in range(len(labels))))


def test_single_gold_items_are_refused():
    with pytest.raises(ValueError):
        mk("x", [1])
    with pytest.raises(ValueError):
        mk("x", [1, None, None])


def test_agreement_classification_and_majority():
    assert mk("a", [1, 1, 1]).agreement == "unanimous"
    assert mk("a", [1, 1, 1]).majority_label == 1
    assert mk("b", [0, 0, 1]).agreement == "majority"
    assert mk("b", [0, 0, 1]).majority_label == 0
    assert mk("c", [1, 0]).agreement == "split"
    assert mk("c", [1, 0]).majority_label is None
    assert mk("d", [1, None, 0, 1]).agreement == "majority"  # None ignored


def test_scorecard_separates_easy_from_contested():
    items = [
        mk("u1", [1, 1, 1]), mk("u2", [0, 0, 0]), mk("u3", [1, 1, 1]),
        mk("m1", [1, 1, 0]), mk("m2", [0, 0, 1]), mk("m3", [1, 1, 0]),
        mk("s1", [1, 0]),
    ]
    preds = [
        Prediction("u1", 1, 0.95), Prediction("u2", 0, 0.05), Prediction("u3", 1, 0.9),
        Prediction("m1", 0, 0.3), Prediction("m2", 1, 0.7), Prediction("m3", 0, 0.2),   # all wrong
        Prediction("s1", 1, 0.9),  # confident commit on a split
    ]
    sc = compute_scorecard(items, preds)
    assert sc.strata["unanimous"].accuracy == 1.0
    assert sc.strata["majority"].accuracy == 0.0
    assert sc.reliability_gap == 1.0
    assert sc.headline == 0.0
    assert sc.strata["split"].n == 1
    assert sc.strata["split"].confident_commit_rate == 1.0
    # overall accuracy hides the gap: 3 right of 6 scored
    assert math.isclose(sc.overall_accuracy, 0.5)


def test_abstention_is_recorded_not_penalised_as_wrong():
    items = [mk("m1", [1, 1, 0]), mk("s1", [1, 0])]
    preds = [Prediction("m1", None, 0.5), Prediction("s1", None, 0.5)]
    sc = compute_scorecard(items, preds)
    assert sc.strata["majority"].abstained == 1
    assert sc.strata["majority"].scored == 0
    assert sc.strata["split"].abstain_rate == 1.0


def test_missing_predictions_counted_not_scored():
    items = [mk("a", [1, 1, 1]), mk("b", [0, 0, 0])]
    sc = compute_scorecard(items, [Prediction("a", 1)])
    assert sc.missing_predictions == 1
    assert sc.strata["unanimous"].scored == 1


def test_label_alpha_reported():
    items = [mk("a", [1, 1, 1]), mk("b", [0, 0, 0]), mk("c", [1, 1, 1])]
    sc = compute_scorecard(items, [Prediction("a", 1), Prediction("b", 0), Prediction("c", 1)])
    assert sc.label_alpha == 1.0
    assert sc.contested_share == 0.0
