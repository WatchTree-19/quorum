"""Prompt construction and response parsing for the LLM harness.

Two design decisions here carry the whole measurement, and both are stated in
the open because either one could be got wrong quietly.

THE CONTAMINATION PROBLEM. Every sovereign in the panel has a real, published
credit rating that almost certainly sits in the training data of any frontier
model. If the prompt names the country, the model can recall the answer instead
of reasoning from the fundamentals, and a high score would measure memory
rather than judgement. So the harness runs BLIND by default: the model sees
GDP per capita, inflation and the current-account balance, and nothing that
identifies the country. The NAMED style exists as the contamination probe. The
difference between the two is not noise to be minimised, it is a finding: it
measures how much of a model's apparent skill is recall.

ABSTENTION MUST BE OFFERED. Quorum never scores abstention as error, and the
behaviour of a model on items where the experts split is one of the headline
things the scorecard reports. A prompt that forces a binary answer makes that
measurement impossible, so the instruction offers abstention explicitly and in
the same breath as the two substantive answers.
"""

from __future__ import annotations

import json
import re
from enum import Enum

from quorum.schema import Item, Prediction


class PromptStyle(str, Enum):
    """How much the model is told."""

    BLIND = "blind"    # fundamentals only; neither the country nor the period
    NAMED = "named"    # country AND period: the full contamination probe
    NAMED_UNDATED = "named_undated"   # the country, but NOT when

    # NAMED_UNDATED exists to split a confound the other two cannot. A model
    # told the country and the quarter can recall what happened next, so on a
    # forward-looking target its score is a hindsight ceiling rather than a
    # forecast. Told the country alone it still has real knowledge of that
    # sovereign's institutions and politics, which is legitimate judgement, but
    # it no longer knows which episode it is looking at. NAMED minus
    # NAMED_UNDATED is therefore the value of knowing WHEN, which on a forward
    # target is very nearly pure memorisation.


SYSTEM = (
    "You are a sovereign credit analyst. You classify sovereigns as investment "
    "grade or speculative grade from macroeconomic fundamentals. You answer "
    "only with the JSON object you are asked for, and nothing else."
)

_INSTRUCTION = """\
Classify this sovereign as investment grade or speculative grade.

Investment grade means BBB- / Baa3 or better on the major agency scales.
Speculative grade means BB+ / Ba1 or worse.

{body}

Answer with a single JSON object and no other text:

{{"answer": "investment_grade" | "speculative" | "abstain",
  "probability_speculative": <number between 0 and 1>,
  "reason": "<one short sentence>"}}

"abstain" is a legitimate answer and is never counted against you. Use it when
the fundamentals genuinely do not settle the question, which is the case for
sovereigns sitting on the boundary. "probability_speculative" is your honest
credence that the sovereign is speculative grade; give it even when you abstain.
"""


def _fmt(value: float | None, unit: str) -> str:
    return "not reported" if value is None else f"{value:,.1f} {unit}".strip()


def build_prompt(item: Item, style: PromptStyle = PromptStyle.BLIND) -> str:
    """Render one item as a prompt. Ratings are never included, in either style."""
    inp = item.inputs
    lines = [
        f"  GDP per capita           : {_fmt(inp.get('gdp_pc_usd'), 'USD')}",
        f"  CPI inflation            : {_fmt(inp.get('inflation_pct'), 'percent')}",
        f"  Current account balance  : {_fmt(inp.get('current_account_pct_gdp'), 'percent of GDP')}",
    ]
    if style in (PromptStyle.NAMED, PromptStyle.NAMED_UNDATED):
        country = item.metadata.get("country", item.item_id)
        head = f"Sovereign: {country}"
        quarter = item.metadata.get("quarter")
        if quarter and style is PromptStyle.NAMED:
            head += f"\nAs at: {quarter}"
        head += "\n\nMacroeconomic fundamentals:"
    else:
        # The period is withheld along with the identity. On the history panel
        # a quarter plus a distinctive inflation print would name the country
        # to anyone who knows the era, which is the leak this style exists to
        # prevent.
        head = ("Sovereign: identity withheld\n\n"
                "Macroeconomic fundamentals:")
    return _INSTRUCTION.format(body=head + "\n" + "\n".join(lines))


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_ANSWERS = {
    "investment_grade": 0,   # the panel's label is 1 for SPECULATIVE
    "investment": 0,
    "speculative": 1,
    "speculative_grade": 1,
    "abstain": None,
}


def parse_response(item_id: str, text: str) -> Prediction:
    """Turn raw model text into a Prediction.

    Defensive by design: models wrap JSON in prose or fences often enough that
    a strict parse would throw away answers that were perfectly clear. An
    unparseable response becomes an abstention with no confidence, which is
    counted and reported rather than silently dropped, because a model that
    cannot follow the output contract is telling you something too.
    """
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return Prediction(item_id=item_id, label=None, confidence=None)
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return Prediction(item_id=item_id, label=None, confidence=None)
    if not isinstance(data, dict):
        return Prediction(item_id=item_id, label=None, confidence=None)

    answer = str(data.get("answer", "")).strip().lower().replace(" ", "_")
    label = _ANSWERS.get(answer, "missing")
    if label == "missing":
        label = None

    conf = data.get("probability_speculative")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = None
    if conf is not None and not (0.0 <= conf <= 1.0):
        conf = None

    return Prediction(item_id=item_id, label=label, confidence=conf)
