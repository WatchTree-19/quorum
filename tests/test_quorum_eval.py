"""Tests for the Inspect AI tasks. Offline and deterministic: mockllm only.

The property that matters most is PARITY with the original harness. The
published results were produced by run_llm_history.py, so the Inspect task must
put the same items to the model and score the answers the same way. The replay
test below feeds GPT-4.1's recorded answers back through the Inspect task and
requires the published figures to come out.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from quorum.adapters.sovereign_history import load_items
from quorum.llm.prompt import PromptStyle
from quorum_eval.quorum_eval import (
    HISTORY_RAW,
    item_to_sample,
    quorum_scorer,
    quorum_sovereign,
    quorum_sovereign_history,
    stratified,
)
from run_llm_history import stratified as published_stratified

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = tempfile.mkdtemp(prefix="quorum_eval_logs_")


@pytest.fixture(scope="module")
def history_items():
    return load_items(HISTORY_RAW)


def _answer(label, confidence=0.5):
    verdict = {0: "investment_grade", 1: "speculative", None: "abstain"}[label]
    return json.dumps({"answer": verdict, "probability_speculative": confidence, "reason": "test"})


def _run(task, outputs):
    model = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs])
    (log,) = inspect_eval(task, model=model, max_samples=1, display="none", log_dir=LOG_DIR)
    assert log.status == "success"
    return log


def _metrics(log):
    return {name: m.value for name, m in log.results.scores[0].metrics.items()}


def test_stratified_matches_the_harness_that_produced_the_published_results(history_items):
    ours = [it.item_id for it in stratified(history_items, 500, 0)]
    theirs, *_ = published_stratified(history_items, 500, 0)
    assert ours == [it.item_id for it in theirs]


def test_history_task_has_the_published_shape():
    samples = list(quorum_sovereign_history().dataset)
    assert len(samples) == 1022
    counts = {a: sum(1 for s in samples if s.metadata["agreement"] == a) for a in ("unanimous", "majority", "split")}
    assert counts == {"unanimous": 500, "majority": 401, "split": 121}


def test_snapshot_task_covers_the_panel():
    assert len(quorum_sovereign().dataset) == 124


def test_blind_sample_withholds_country_and_quarter(history_items):
    item = next(it for it in history_items if it.item_id == "BRA:2008Q3")
    blind = item_to_sample(item, PromptStyle.BLIND)
    named = item_to_sample(item, PromptStyle.NAMED)
    assert "BRA" in named.input and "2008Q3" in named.input
    assert "BRA" not in blind.input and "2008" not in blind.input
    assert blind.metadata["country"] == "BRA"


def test_target_is_the_majority_and_split_items_have_no_verdict(history_items):
    for item in history_items[:400]:
        sample = item_to_sample(item, PromptStyle.BLIND)
        expected = {0: "investment_grade", 1: "speculative", None: "no_verdict"}[item.majority_label]
        assert sample.target == expected
        assert sample.metadata["labels"] == dict(zip(item.rater_ids, item.labels))


def test_invalid_style_is_refused():
    with pytest.raises(ValueError, match="style must be one of"):
        quorum_sovereign(style="anonymous")


def test_end_to_end_with_mockllm_default_output():
    """mockllm's default reply is not JSON, so every answer is an abstention."""
    (log,) = inspect_eval(quorum_sovereign(), model="mockllm/model", limit=10, display="none", log_dir=LOG_DIR)
    assert log.status == "success"
    metrics = _metrics(log)
    assert metrics["abstain_rate_unanimous"] == 1.0
    assert math.isnan(metrics["unanimous_accuracy"])


def test_quorum_scorer_outcomes(history_items):
    """Right, wrong, abstained and a verdict on a split item, one of each."""
    unanimous = next(it for it in history_items if it.agreement == "unanimous" and it.majority_label == 0)
    majority = next(it for it in history_items if it.agreement == "majority" and it.majority_label == 1)
    split = next(it for it in history_items if it.agreement == "split")
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.solver import generate

    samples = [item_to_sample(it, PromptStyle.BLIND) for it in (unanimous, majority, split, unanimous)]
    samples[3].id = "repeat"  # sample ids must be unique
    task = Task(dataset=MemoryDataset(samples), solver=generate(), scorer=quorum_scorer())
    log = _run(task, [_answer(0), _answer(0), _answer(1), _answer(None)])
    outcomes = {s.id: s.scores["quorum_scorer"].metadata["outcome"] for s in log.samples}
    assert outcomes == {
        unanimous.item_id: "correct",
        majority.item_id: "incorrect",
        split.item_id: "committed_on_split",
        "repeat": "abstained",
    }
    metrics = _metrics(log)
    assert metrics["unanimous_accuracy"] == 1.0
    assert metrics["contested_accuracy"] == 0.0
    assert metrics["reliability_gap"] == 1.0
    assert metrics["abstain_rate_unanimous"] == 0.5
    assert metrics["split_commit_rate"] == 1.0


@pytest.mark.parametrize("result_file", ["hist_gpt_blind.json", "hist_gpt_named.json"])
def test_replaying_the_published_run_reproduces_its_scorecard(result_file):
    """GPT-4.1's recorded answers, scored by the Inspect task, give the published numbers."""
    published = json.loads((ROOT / result_file).read_text(encoding="utf-8"))
    by_id = {p["item_id"]: p for p in published["predictions"]}
    task = quorum_sovereign_history(style=published["style"])
    outputs = [_answer(by_id[s.id]["label"], by_id[s.id]["confidence"]) for s in task.dataset]
    metrics = _metrics(_run(task, outputs))

    strata = published["strata"]
    assert metrics["unanimous_accuracy"] == pytest.approx(strata["unanimous"]["accuracy"])
    assert metrics["contested_accuracy"] == pytest.approx(published["headline_contested_accuracy"])
    assert metrics["reliability_gap"] == pytest.approx(published["reliability_gap"])
    assert metrics["overall_accuracy"] == pytest.approx(published["overall_accuracy"])
    assert metrics["abstain_rate_unanimous"] == pytest.approx(strata["unanimous"]["abstain_rate"])
    assert metrics["abstain_rate_contested"] == pytest.approx(strata["majority"]["abstain_rate"])
    assert metrics["split_commit_rate"] == pytest.approx(1 - strata["split"]["abstain_rate"])
