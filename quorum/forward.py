"""Scoring contested items against what the raters eventually settled on.

THE PROBLEM THIS FIXES. Quorum's headline metric is accuracy on contested
items, measured against the contemporaneous majority of raters. That is
consensus tracking, stated openly in the whitepaper: agreeing with two agencies
out of three is not the same as being right, because on a contested item there
is no external truth to be right about.

On a panel that runs through time there is one. A disagreement spell ENDS. When
the agencies split over whether a sovereign is investment grade, one side is
eventually vindicated, because the others move to meet it. The verdict they
converge on is a realised outcome rather than a vote, and it is available for
every spell that closes inside the panel.

Scoring the contested stratum against that verdict turns the benchmark's
softest metric into a genuine forecasting task with a resolved answer: on this
quarter, when three agencies disagreed, could the model tell which way it was
going to go?

TWO THINGS TO HOLD ON TO WHEN READING THE RESULT.

The contemporaneous majority is itself a competitor here, and a real one. It is
not a coin flip: during a boundary transition it lags by construction, because
agencies move one at a time, but it still carries signal. Its own accuracy
against the settled verdict is reported as the baseline any model has to beat.

And a model told both the country and the quarter can simply recall how that
episode ended. On a forward-looking target that is not a forecast, it is
hindsight, so a NAMED score here is an upper bound rather than a skill
estimate. PromptStyle.NAMED_UNDATED exists to separate the two.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from quorum.schema import Item, Prediction


@dataclass(frozen=True)
class Spell:
    """One run of contested periods, and what the raters settled on after it."""

    group: str
    item_ids: tuple[str, ...]
    settled: int | None      # the verdict after the spell; None if still open
    length: int

    @property
    def resolved(self) -> bool:
        return self.settled is not None


def _period(item: Item) -> str:
    return item.metadata.get("quarter", item.item_id)


def find_spells(items: list[Item]) -> list[Spell]:
    """Group items by entity, order by period, and cut them into spells.

    A spell is a maximal run of non-unanimous periods. It resolves if a
    unanimous period follows it inside the panel; a spell still running at the
    end of the data is returned unresolved and is never scored.
    """
    by_group: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_group[it.metadata.get("group", it.item_id)].append(it)

    spells: list[Spell] = []
    for group, seq in sorted(by_group.items()):
        seq = sorted(seq, key=_period)
        i = 0
        while i < len(seq):
            if seq[i].agreement == "unanimous":
                i += 1
                continue
            j = i
            while j < len(seq) and seq[j].agreement != "unanimous":
                j += 1
            spells.append(Spell(
                group=group,
                item_ids=tuple(x.item_id for x in seq[i:j]),
                settled=seq[j].majority_label if j < len(seq) else None,
                length=j - i,
            ))
            i = j
    return spells


def settled_verdicts(items: list[Item]) -> dict[str, int]:
    """item_id -> the verdict its spell eventually settled on. Resolved only."""
    out: dict[str, int] = {}
    for spell in find_spells(items):
        if spell.resolved:
            for item_id in spell.item_ids:
                out[item_id] = spell.settled  # type: ignore[assignment]
    return out


@dataclass
class ForwardResult:
    n_spells: int = 0
    n_resolved_spells: int = 0
    n_items: int = 0             # contested items with a resolved outcome
    scored: int = 0              # ... on which the model committed
    correct: int = 0
    abstained: int = 0
    majority_scored: int = 0     # the contemporaneous majority, same items
    majority_correct: int = 0
    item_ids: list[str] = field(default_factory=list)
    outcomes: list[tuple[str, int, int | None]] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        return (self.correct / self.scored) if self.scored else None

    @property
    def majority_accuracy(self) -> float | None:
        return (self.majority_correct / self.majority_scored) if self.majority_scored else None

    @property
    def edge_over_majority(self) -> float | None:
        a, m = self.accuracy, self.majority_accuracy
        return None if (a is None or m is None) else a - m


def forward_score(items: list[Item], predictions: list[Prediction]) -> ForwardResult:
    """Score the contested stratum against the eventual settled verdict."""
    spells = find_spells(items)
    verdicts = settled_verdicts(items)
    by_id = {p.item_id: p for p in predictions}
    items_by_id = {it.item_id: it for it in items}

    res = ForwardResult(
        n_spells=len(spells),
        n_resolved_spells=sum(1 for s in spells if s.resolved),
        n_items=len(verdicts),
    )
    for item_id, outcome in sorted(verdicts.items()):
        res.item_ids.append(item_id)
        item = items_by_id[item_id]
        if item.majority_label is not None:
            res.majority_scored += 1
            res.majority_correct += int(item.majority_label == outcome)
        pred = by_id.get(item_id)
        if pred is None or pred.label is None:
            res.abstained += 1
            res.outcomes.append((item_id, outcome, None))
            continue
        res.scored += 1
        res.correct += int(pred.label == outcome)
        res.outcomes.append((item_id, outcome, pred.label))
    return res


def render_forward(res: ForwardResult, model_name: str = "model") -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    lines = [
        "=" * 74,
        f"FORWARD SCORE  |  contested items against the verdict the raters settled on",
        "=" * 74,
        f"  disagreement spells              {res.n_spells:,}"
        f"   ({res.n_resolved_spells:,} closed inside the panel)",
        f"  contested items with an outcome  {res.n_items:,}",
        f"  model committed on               {res.scored:,}   abstained on {res.abstained:,}",
        "",
        f"  {model_name}: {pct(res.accuracy)}",
        f"  contemporaneous majority of raters: {pct(res.majority_accuracy)}",
    ]
    edge = res.edge_over_majority
    if edge is not None:
        lines.append(f"  edge over the majority: {edge:+.1%}")
    lines += [
        "",
        "  The majority is a real baseline, not a coin flip. It lags during a",
        "  transition because raters move one at a time, and it still carries",
        "  signal. A model that only tracks consensus cannot beat it here.",
        "",
        "  If the model was told both the country and the period, read its score",
        "  as a hindsight ceiling rather than a forecast: it can recall how the",
        "  episode ended. PromptStyle.NAMED_UNDATED separates the two.",
        "=" * 74,
    ]
    return "\n".join(lines)
