"""Tests for the history-panel harness. Offline, deterministic.

Two properties here are load-bearing and easy to break silently.

BLINDNESS ON THIS PANEL IS HARDER than on the snapshot, because a country-
quarter can be identified by its period as well as its name: a reader who knows
the era can name a country from a quarter plus a distinctive inflation print.
So the blind style withholds the quarter too, and the test asserts both halves
against a NAMED prompt that does contain them.

STRATIFICATION MUST NOT TOUCH THE CONTESTED STRATUM. The whole reason for
sampling is to spend the calls where the headline metric lives, so every
contested item has to survive it. A sampler that quietly dropped some would
still produce a plausible-looking scorecard.
"""

import pytest

from quorum.adapters.sovereign_history import load_items
from quorum.llm.prompt import PromptStyle, build_prompt
from run_llm_history import stratified

RAW = "data/history_raw"


@pytest.fixture(scope="module")
def items():
    return load_items(RAW)


def test_panel_shape(items):
    assert len(items) == 2697
    assert sum(1 for it in items if it.agreement != "unanimous") == 522


def test_every_item_carries_fundamentals_so_blind_is_possible(items):
    """Without WDI figures a blind prompt would carry no information at all."""
    keys = {"gdp_pc_usd", "inflation_pct", "current_account_pct_gdp"}
    for it in items:
        assert set(it.inputs) == keys
    with_gdp = sum(1 for it in items if it.inputs["gdp_pc_usd"] is not None)
    assert with_gdp / len(items) > 0.97


def test_blind_prompt_withholds_both_the_country_and_the_quarter(items):
    it = next(i for i in items if i.agreement != "unanimous")
    named = build_prompt(it, PromptStyle.NAMED)
    blind = build_prompt(it, PromptStyle.BLIND)
    # Capable of failing: it finds both when both are there.
    assert it.metadata["iso3"] in named
    assert it.metadata["quarter"] in named
    assert it.metadata["iso3"] not in blind
    assert it.metadata["quarter"] not in blind
    assert "identity withheld" in blind


def test_no_blind_prompt_in_the_whole_panel_leaks_an_identifier(items):
    for it in items:
        blind = build_prompt(it, PromptStyle.BLIND)
        assert it.metadata["iso3"] not in blind
        assert it.metadata["quarter"] not in blind
        assert it.item_id not in blind


# --- stratification ------------------------------------------------------

def test_stratified_keeps_every_contested_item(items):
    work, n_con, n_kept, n_all = stratified(items, 500, seed=0)
    assert n_con == 522
    assert sum(1 for it in work if it.agreement != "unanimous") == 522
    assert n_kept == 500
    assert len(work) == 1022


def test_stratified_with_zero_unanimous_scores_contested_only(items):
    work, n_con, n_kept, _ = stratified(items, 0, seed=0)
    assert n_kept == 0
    assert len(work) == n_con == 522
    assert all(it.agreement != "unanimous" for it in work)


def test_asking_for_more_unanimous_than_exist_takes_all_of_them(items):
    work, n_con, n_kept, n_all = stratified(items, 10**6, seed=0)
    assert n_kept == n_all
    assert len(work) == len(items)


def test_sampling_is_reproducible_from_the_seed(items):
    a, *_ = stratified(items, 200, seed=7)
    b, *_ = stratified(items, 200, seed=7)
    c, *_ = stratified(items, 200, seed=8)
    assert [i.item_id for i in a] == [i.item_id for i in b]
    assert [i.item_id for i in a] != [i.item_id for i in c]


def test_no_duplicate_items_in_a_stratified_set(items):
    work, *_ = stratified(items, 500, seed=0)
    ids = [it.item_id for it in work]
    assert len(ids) == len(set(ids))
