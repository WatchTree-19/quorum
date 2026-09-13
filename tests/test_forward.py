"""Tests for forward scoring and the NAMED_UNDATED style. Offline, deterministic.

The spell logic is the load-bearing part: it decides which contested items get
a realised outcome and what that outcome is. A silent off-by-one here would
produce a plausible-looking forecasting metric scored against the wrong verdict,
so the synthetic cases below pin the boundaries by hand rather than trusting the
real panel to exercise them.
"""

import pytest

from quorum.adapters.sovereign_history import load_items
from quorum.forward import find_spells, forward_score, settled_verdicts
from quorum.llm.prompt import PromptStyle, build_prompt
from quorum.schema import Item, Prediction

RAW = "data/history_raw"


def q(group, quarter, labels):
    return Item(
        item_id=f"{group}:{quarter}", task="t", inputs={},
        labels=tuple(labels), rater_ids=("A", "B", "C"),
        metadata={"group": group, "quarter": quarter, "country": group},
    )


UNAN_IG = [0, 0, 0]
UNAN_SPEC = [1, 1, 1]
SPLIT_IG = [0, 0, 1]      # majority says investment grade
SPLIT_SPEC = [1, 1, 0]    # majority says speculative


# --- spells --------------------------------------------------------------

def test_a_spell_takes_the_verdict_of_the_period_that_ends_it():
    items = [q("X", "2000Q1", UNAN_IG), q("X", "2000Q2", SPLIT_IG),
             q("X", "2000Q3", SPLIT_SPEC), q("X", "2000Q4", UNAN_SPEC)]
    spells = find_spells(items)
    assert len(spells) == 1
    assert spells[0].length == 2
    assert spells[0].settled == 1          # the verdict AFTER the spell
    v = settled_verdicts(items)
    # Both contested quarters inherit it, including the one whose own
    # contemporaneous majority disagreed with where it ended up.
    assert v == {"X:2000Q2": 1, "X:2000Q3": 1}


def test_a_spell_still_running_at_the_end_of_the_panel_is_never_scored():
    items = [q("X", "2000Q1", UNAN_IG), q("X", "2000Q2", SPLIT_IG)]
    spells = find_spells(items)
    assert len(spells) == 1
    assert spells[0].resolved is False
    assert settled_verdicts(items) == {}


def test_two_spells_separated_by_agreement_are_not_merged():
    items = [q("X", "2000Q1", SPLIT_IG), q("X", "2000Q2", UNAN_IG),
             q("X", "2000Q3", SPLIT_SPEC), q("X", "2000Q4", UNAN_SPEC)]
    spells = find_spells(items)
    assert [s.length for s in spells] == [1, 1]
    assert [s.settled for s in spells] == [0, 1]


def test_spells_do_not_run_across_entities():
    items = [q("X", "2000Q1", SPLIT_IG), q("Y", "2000Q2", SPLIT_SPEC),
             q("X", "2000Q2", UNAN_SPEC), q("Y", "2000Q3", UNAN_IG)]
    spells = find_spells(items)
    assert {s.group for s in spells} == {"X", "Y"}
    assert settled_verdicts(items) == {"X:2000Q1": 1, "Y:2000Q2": 0}


def test_periods_are_ordered_even_when_the_input_is_shuffled():
    ordered = [q("X", "2000Q1", UNAN_IG), q("X", "2000Q2", SPLIT_IG),
               q("X", "2000Q3", UNAN_SPEC)]
    assert settled_verdicts(list(reversed(ordered))) == settled_verdicts(ordered)


# --- scoring -------------------------------------------------------------

def test_forward_score_counts_the_model_and_the_majority_separately():
    items = [q("X", "2000Q1", UNAN_IG), q("X", "2000Q2", SPLIT_IG),
             q("X", "2000Q3", SPLIT_IG), q("X", "2000Q4", UNAN_SPEC)]
    # Settled verdict is 1. Both contested majorities say 0, so the majority
    # scores 0/2 whilst a model that called it scores 2/2.
    preds = [Prediction("X:2000Q2", 1), Prediction("X:2000Q3", 1)]
    res = forward_score(items, preds)
    assert res.n_items == 2
    assert res.scored == 2 and res.correct == 2
    assert res.accuracy == 1.0
    assert res.majority_scored == 2 and res.majority_correct == 0
    assert res.majority_accuracy == 0.0
    assert res.edge_over_majority == 1.0


def test_abstention_is_not_scored_as_wrong():
    items = [q("X", "2000Q1", UNAN_IG), q("X", "2000Q2", SPLIT_IG),
             q("X", "2000Q3", UNAN_SPEC)]
    res = forward_score(items, [Prediction("X:2000Q2", None)])
    assert res.abstained == 1
    assert res.scored == 0
    assert res.accuracy is None


def test_a_missing_prediction_counts_as_an_abstention_not_a_miss():
    items = [q("X", "2000Q1", UNAN_IG), q("X", "2000Q2", SPLIT_IG),
             q("X", "2000Q3", UNAN_SPEC)]
    res = forward_score(items, [])
    assert res.abstained == 1 and res.scored == 0


def test_unanimous_items_are_never_forward_scored():
    items = [q("X", "2000Q1", UNAN_IG), q("X", "2000Q2", SPLIT_IG),
             q("X", "2000Q3", UNAN_SPEC)]
    res = forward_score(items, [Prediction("X:2000Q1", 0), Prediction("X:2000Q2", 1)])
    assert res.n_items == 1
    assert res.item_ids == ["X:2000Q2"]


# --- the real panel ------------------------------------------------------

def test_real_panel_shape():
    items = load_items(RAW)
    spells = find_spells(items)
    assert len(spells) == 47
    assert sum(1 for s in spells if s.resolved) == 39
    assert len(settled_verdicts(items)) == 424


# --- NAMED_UNDATED -------------------------------------------------------

def test_named_undated_gives_the_country_and_withholds_the_period():
    it = next(i for i in load_items(RAW) if i.agreement != "unanimous")
    named = build_prompt(it, PromptStyle.NAMED)
    undated = build_prompt(it, PromptStyle.NAMED_UNDATED)
    quarter = it.metadata["quarter"]
    # Capable of failing: the period is found when it is genuinely there.
    assert quarter in named
    assert quarter not in undated
    assert it.metadata["iso3"] in undated


def test_no_named_undated_prompt_in_the_panel_leaks_a_period():
    for it in load_items(RAW):
        text = build_prompt(it, PromptStyle.NAMED_UNDATED)
        assert it.metadata["quarter"] not in text


def test_blind_still_withholds_both_now_that_a_third_style_exists():
    it = next(i for i in load_items(RAW) if i.agreement != "unanimous")
    blind = build_prompt(it, PromptStyle.BLIND)
    assert it.metadata["iso3"] not in blind
    assert it.metadata["quarter"] not in blind


@pytest.mark.parametrize("style", list(PromptStyle))
def test_every_style_still_offers_abstention(style):
    it = next(i for i in load_items(RAW) if i.agreement != "unanimous")
    assert "abstain" in build_prompt(it, style)
