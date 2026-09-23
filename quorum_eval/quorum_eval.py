"""Quorum as an Inspect AI evaluation.

The same benchmark as run_llm_history.py and run_llm_sovereign.py, expressed as
Inspect tasks so that any model Inspect can reach is scored through one command.
The prompt, the system message, the output contract and the response parser are
imported from quorum.llm rather than restated, so an Inspect run and a run of the
original harness put exactly the same question to the model and read its answer
the same way.

    uv run inspect eval quorum_eval/quorum_eval.py@quorum_sovereign_history --model openai/gpt-4.1-mini
    uv run inspect eval quorum_eval/quorum_eval.py@quorum_sovereign_history -T style=named
    uv run inspect eval quorum_eval/quorum_eval.py@quorum_sovereign

What the metrics mean, briefly (README.md has the full account):

  contested_accuracy     accuracy on items where the agencies disagreed but a
                         majority exists, scored against that majority. The
                         headline. Evenly split items are reported separately.
  unanimous_accuracy     accuracy where all agencies agreed.
  reliability_gap        unanimous minus contested accuracy.
  abstain_rate_*         how often the model declined, by stratum. Abstention is
                         offered in the prompt and is never scored as an error.
  split_commit_rate      on items where the agencies are evenly split there is no
                         right answer; this is how often the model answered anyway.

ITEMS ON THE HISTORY PANEL ARE NOT INDEPENDENT. A country contributes many
quarters and, on the blind style, the four quarters of one year produce the same
prompt. Inspect's per-metric figures are point estimates; intervals should be
bootstrapped over countries, which every sample's metadata carries as "country"
so that the log alone is enough to do it (compare_runs.py in this repository
does exactly that).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState, generate, system_message

from quorum.adapters import sovereign_history, sovereign_ratings
from quorum.llm.prompt import SYSTEM, PromptStyle, build_prompt, parse_response
from quorum.schema import Item

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HISTORY_RAW = DATA_DIR / "history_raw"
SNAPSHOT_CSV = DATA_DIR / "sovereign_panel.csv"

# The same settings the original harness sends to every provider.
GENERATE_CONFIG = GenerateConfig(temperature=0.0, max_tokens=300)

_VERDICT = {0: "investment_grade", 1: "speculative", None: "no_verdict"}


# --------------------------------------------------------------------------- data


def stratified(items: list[Item], n_unanimous: int, seed: int) -> list[Item]:
    """Every contested item, plus a seeded random sample of unanimous ones.

    Identical to run_llm_history.stratified, which produced the published
    results; a test holds the two to the same item set.
    """
    contested = [it for it in items if it.agreement != "unanimous"]
    unanimous = [it for it in items if it.agreement == "unanimous"]
    rng = random.Random(seed)
    keep = unanimous if n_unanimous >= len(unanimous) else rng.sample(unanimous, n_unanimous)
    out = contested + keep
    out.sort(key=lambda it: it.item_id)
    return out


def item_to_sample(item: Item, style: PromptStyle) -> Sample:
    """One Quorum item as an Inspect sample.

    The target is the agencies' majority verdict, or "no_verdict" on an even
    split. The individual agency labels travel in the metadata, so a log can be
    rescored against any single rater, or re-aggregated, without re-running.
    """
    return Sample(
        id=item.item_id,
        input=build_prompt(item, style),
        target=_VERDICT[item.majority_label],
        metadata={
            "agreement": item.agreement,
            "labels": dict(zip(item.rater_ids, item.labels)),
            "country": item.metadata.get("iso3", item.metadata.get("group", item.item_id)),
            "quarter": item.metadata.get("quarter"),
            "style": style.value,
        },
    )


def _dataset(items: list[Item], style: PromptStyle, name: str) -> MemoryDataset:
    return MemoryDataset([item_to_sample(it, style) for it in items], name=name)


# ------------------------------------------------------------------------ scorer


def _stratum(score: SampleScore) -> str:
    return str((score.score.metadata or {}).get("agreement", ""))


def _outcome(score: SampleScore) -> str:
    return str((score.score.metadata or {}).get("outcome", ""))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def _accuracy_in(scores: list[SampleScore], strata: set[str]) -> float:
    scored = [s for s in scores if _stratum(s) in strata and _outcome(s) in ("correct", "incorrect")]
    return _ratio(sum(1 for s in scored if _outcome(s) == "correct"), len(scored))


def _abstain_rate_in(scores: list[SampleScore], strata: set[str]) -> float:
    group = [s for s in scores if _stratum(s) in strata]
    return _ratio(sum(1 for s in group if _outcome(s) == "abstained"), len(group))


@metric
def contested_accuracy() -> Metric:
    """Accuracy against the majority on items where the agencies disagreed."""

    def compute(scores: list[SampleScore]) -> float:
        return _accuracy_in(scores, {"majority"})

    return compute


@metric
def unanimous_accuracy() -> Metric:
    """Accuracy on items where every agency agreed."""

    def compute(scores: list[SampleScore]) -> float:
        return _accuracy_in(scores, {"unanimous"})

    return compute


@metric
def overall_accuracy() -> Metric:
    """Accuracy on every item with a majority verdict, abstentions excluded."""

    def compute(scores: list[SampleScore]) -> float:
        return _accuracy_in(scores, {"unanimous", "majority"})

    return compute


@metric
def reliability_gap() -> Metric:
    """Unanimous minus contested accuracy: how much the model leans on consensus."""

    def compute(scores: list[SampleScore]) -> float:
        return _accuracy_in(scores, {"unanimous"}) - _accuracy_in(scores, {"majority"})

    return compute


@metric
def abstain_rate_unanimous() -> Metric:
    """Share of unanimous items on which the model abstained."""

    def compute(scores: list[SampleScore]) -> float:
        return _abstain_rate_in(scores, {"unanimous"})

    return compute


@metric
def abstain_rate_contested() -> Metric:
    """Share of contested (majority) items on which the model abstained.

    Split items are left out, as in the published scorecard: there the model's
    behaviour is reported by split_commit_rate instead.
    """

    def compute(scores: list[SampleScore]) -> float:
        return _abstain_rate_in(scores, {"majority"})

    return compute


@metric
def split_commit_rate() -> Metric:
    """Share of evenly split items on which the model gave a verdict anyway."""

    def compute(scores: list[SampleScore]) -> float:
        group = [s for s in scores if _stratum(s) == "split"]
        return _ratio(sum(1 for s in group if _outcome(s) == "committed_on_split"), len(group))

    return compute


QUORUM_METRICS = [
    contested_accuracy(),
    unanimous_accuracy(),
    reliability_gap(),
    overall_accuracy(),
    abstain_rate_unanimous(),
    abstain_rate_contested(),
    split_commit_rate(),
]


@scorer(metrics=QUORUM_METRICS)
def quorum_scorer() -> Scorer:
    """Score one answer against the quorum, keeping abstention out of the error count.

    A verdict on an item with a majority is CORRECT or INCORRECT against that
    majority. An abstention, a response that does not follow the output
    contract, and any verdict on an evenly split item are recorded as NOANSWER,
    because in each case there is nothing to mark right or wrong; the metadata
    says which of the three it was, and the metrics count them separately.
    """

    async def score(state: TaskState, target: Target) -> Score:
        agreement = str(state.metadata.get("agreement", ""))
        completion = state.output.completion if state.output else ""
        prediction = parse_response(str(state.sample_id), completion)
        answer = _VERDICT[prediction.label] if prediction.label is not None else "abstain"
        metadata: dict[str, Any] = {
            "agreement": agreement,
            "country": state.metadata.get("country"),
            "prediction": prediction.label,
            "probability_speculative": prediction.confidence,
        }

        if prediction.label is None:
            metadata["outcome"] = "abstained"
            return Score(value=NOANSWER, answer=answer, metadata=metadata,
                         explanation="Abstained or did not follow the output contract.")
        if target.text == _VERDICT[None]:
            metadata["outcome"] = "committed_on_split"
            return Score(value=NOANSWER, answer=answer, metadata=metadata,
                         explanation="The agencies are evenly split, so no verdict is right or wrong.")
        hit = answer == target.text
        metadata["outcome"] = "correct" if hit else "incorrect"
        return Score(value=CORRECT if hit else INCORRECT, answer=answer, metadata=metadata)

    return score


# ------------------------------------------------------------------------- tasks


def _style(style: str) -> PromptStyle:
    try:
        return PromptStyle(style)
    except ValueError as exc:
        choices = ", ".join(s.value for s in PromptStyle)
        raise ValueError(f"style must be one of {choices}; got {style!r}") from exc


def _task(dataset: MemoryDataset) -> Task:
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM), generate()],
        scorer=quorum_scorer(),
        config=GENERATE_CONFIG,
        version="1.0.0",
    )


@task
def quorum_sovereign_history(style: str = "blind", unanimous: int = 500, seed: int = 0) -> Task:
    """Investment grade or speculative, country-quarter by country-quarter, 1995 to 2026.

    27 sovereigns that crossed the investment-grade boundary, with S&P, Moody's
    and Fitch as three independent raters. By default every one of the 522
    contested country-quarters is scored alongside a seeded sample of 500
    unanimous ones, the design behind the published results.

    Args:
        style: "blind" (fundamentals only, the default), "named" (country and
            quarter supplied, the contamination probe) or "named_undated"
            (country supplied, quarter withheld).
        unanimous: How many unanimous items to sample. 0 scores the contested
            stratum alone; a number at or above the panel's count scores all.
        seed: Sampling seed.
    """
    items = sovereign_history.load_items(HISTORY_RAW)
    work = stratified(items, unanimous, seed)
    return _task(_dataset(work, _style(style), "quorum_sovereign_history"))


@task
def quorum_sovereign(style: str = "blind") -> Task:
    """Investment grade or speculative for 124 sovereigns, as rated in September 2026.

    Args:
        style: "blind" (fundamentals only, the default) or "named" (the country
            is supplied, the contamination probe).
    """
    items = sovereign_ratings.load_items(SNAPSHOT_CSV)
    return _task(_dataset(items, _style(style), "quorum_sovereign"))


__all__ = [
    "quorum_sovereign",
    "quorum_sovereign_history",
    "quorum_scorer",
    "item_to_sample",
    "stratified",
]
