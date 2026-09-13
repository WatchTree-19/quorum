"""Tests for the LLM harness. Offline and deterministic: no network, no keys.

The blindness tests are the ones that matter. If the country name leaks into a
BLIND prompt, every number the harness produces measures recall of published
ratings instead of judgement from fundamentals, and nothing downstream would
show it. So each blindness assertion is written to be capable of failing: the
NAMED counterpart is asserted in the same test, which proves the check can see
the identifier when it is genuinely there.
"""

import json

import pytest

from quorum.adapters.sovereign_ratings import load_items
from quorum.llm.prompt import SYSTEM, PromptStyle, build_prompt, parse_response
from quorum.llm.providers import FakeProvider, ProviderError, get_provider
from quorum.llm.runner import run_model
from quorum.schema import Item

PANEL = "data/sovereign_panel.csv"


def an_item(country="Ruritania", iso="RUR", gdp=9_000.0):
    return Item(
        item_id=iso, task="sovereign_investment_grade",
        inputs={"gdp_pc_usd": gdp, "inflation_pct": 4.0, "current_account_pct_gdp": -2.0},
        labels=(1, 0, 1), rater_ids=("SP", "MOODYS", "FITCH"),
        metadata={"country": country, "ratings": {"SP": "BB+", "MOODYS": "Baa3", "FITCH": "BB+"}},
    )


# --- the prompt does not leak the answer ---------------------------------

def test_blind_prompt_hides_the_country_and_named_prompt_shows_it():
    item = an_item()
    blind = build_prompt(item, PromptStyle.BLIND)
    named = build_prompt(item, PromptStyle.NAMED)
    # The check is capable of failing: it finds the name when the name is there.
    assert "Ruritania" in named
    assert "Ruritania" not in blind
    assert "identity withheld" in blind


def test_no_prompt_ever_contains_the_agency_ratings():
    item = an_item()
    for style in PromptStyle:
        text = build_prompt(item, style)
        for rating in ("BB+", "Baa3"):
            assert rating not in text.replace("BBB- / Baa3", "").replace("BB+ / Ba1", "")


def test_real_panel_blind_prompts_contain_no_country_name_or_iso_code():
    items = load_items(PANEL)
    for item in items:
        text = build_prompt(item, PromptStyle.BLIND)
        assert item.metadata["country"] not in text
        assert item.item_id not in text


def test_prompt_offers_abstention_because_the_scorecard_measures_it():
    text = build_prompt(an_item(), PromptStyle.BLIND)
    assert "abstain" in text
    assert "never counted against you" in text


# --- parsing -------------------------------------------------------------

@pytest.mark.parametrize("raw,label,conf", [
    ('{"answer":"speculative","probability_speculative":0.8}', 1, 0.8),
    ('{"answer":"investment_grade","probability_speculative":0.12}', 0, 0.12),
    ('{"answer":"abstain","probability_speculative":0.5}', None, 0.5),
    ('```json\n{"answer": "speculative", "probability_speculative": 0.9}\n```', 1, 0.9),
    ('Here you go: {"answer":"investment_grade","probability_speculative":0.2} hope that helps',
     0, 0.2),
])
def test_parse_accepts_the_shapes_models_actually_return(raw, label, conf):
    p = parse_response("X", raw)
    assert p.label == label
    assert p.confidence == pytest.approx(conf)


@pytest.mark.parametrize("raw", ["", "I cannot answer that.", "{not json", "[1,2,3]"])
def test_unreadable_responses_become_abstentions_not_guesses(raw):
    p = parse_response("X", raw)
    assert p.label is None
    assert p.confidence is None


def test_out_of_range_confidence_is_dropped_rather_than_clipped():
    p = parse_response("X", '{"answer":"speculative","probability_speculative":7}')
    assert p.label == 1
    assert p.confidence is None


# --- the runner ----------------------------------------------------------

def test_run_model_scores_every_item_and_counts_calls():
    items = load_items(PANEL)[:12]
    provider = FakeProvider()
    preds, stats = run_model(items, provider, cache_dir=None, progress=False)
    assert len(preds) == 12
    assert provider.calls == 12
    assert stats.called == 12
    assert stats.errors == 0
    assert {p.item_id for p in preds} == {i.item_id for i in items}


def test_cache_prevents_a_second_run_from_spending_anything(tmp_path):
    items = load_items(PANEL)[:8]
    first = FakeProvider()
    run_model(items, first, cache_dir=tmp_path, progress=False)
    assert first.calls == 8

    second = FakeProvider()
    preds, stats = run_model(items, second, cache_dir=tmp_path, progress=False)
    assert second.calls == 0
    assert stats.from_cache == 8
    assert len(preds) == 8


def test_changing_the_prompt_style_invalidates_the_cache(tmp_path):
    items = load_items(PANEL)[:5]
    run_model(items, FakeProvider(), cache_dir=tmp_path,
              style=PromptStyle.BLIND, progress=False)
    named = FakeProvider()
    _, stats = run_model(items, named, cache_dir=tmp_path,
                         style=PromptStyle.NAMED, progress=False)
    assert named.calls == 5, "a NAMED run must not be served BLIND answers"
    assert stats.from_cache == 0


def test_a_failing_provider_yields_abstentions_and_is_counted_not_hidden():
    class Broken(FakeProvider):
        def complete(self, system, prompt):
            raise ProviderError("429 forever")

    items = load_items(PANEL)[:4]
    preds, stats = run_model(items, Broken(), cache_dir=None, progress=False)
    assert stats.errors == 4
    assert stats.abstained == 4
    assert all(p.label is None for p in preds)
    assert stats.error_examples


def test_cache_file_holds_the_prompt_and_completion_and_no_key(tmp_path):
    items = load_items(PANEL)[:1]
    run_model(items, FakeProvider(), cache_dir=tmp_path, progress=False)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    blob = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(blob) == {"model", "style", "prompt", "completion"}


# --- providers -----------------------------------------------------------

def test_a_missing_api_key_is_a_clear_error_not_a_silent_empty_run(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        get_provider("anthropic").complete(SYSTEM, "hello")


def test_unknown_provider_names_are_refused():
    with pytest.raises(ProviderError, match="Unknown provider"):
        get_provider("acme")


def test_model_id_override_reaches_the_provider():
    assert get_provider("openai", "gpt-4.1-mini").model == "gpt-4.1-mini"
