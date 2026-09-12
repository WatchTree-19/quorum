"""Rater affinity: which expert does the model secretly think like?

On unanimous items every rater says the same thing, so agreement with any one
rater is just accuracy. On contested items the raters part ways, and a model's
prediction must land with someone. Agreement with each named rater, measured
on contested items only, therefore reveals whose rulebook the model has
internalised. No single-gold benchmark can produce this number at all, because
a single-gold benchmark has already collapsed the raters into one.

This is also the honest answer to "isn't contested accuracy just agreeing
with two experts out of three?" -- yes, headline accuracy on the contested
stratum is consensus tracking, stated openly; affinity is the decomposition
that shows which way the model leans when the consensus is thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from quorum.schema import Item, Prediction


@dataclass(frozen=True)
class RaterAffinity:
    rater_id: str
    n: int          # contested items where this rater labelled and the model committed
    agree: int

    @property
    def agreement(self) -> float | None:
        return (self.agree / self.n) if self.n else None


def rater_affinity(items: list[Item], predictions: list[Prediction]) -> list[RaterAffinity]:
    """Per-rater agreement of the model's committed predictions, contested items only.

    Contested means not unanimous (majority and split items both count; on a
    split there is no majority to be right against, but each rater still took
    a side, so affinity is still defined). Abstentions are excluded.
    """
    by_id = {p.item_id: p for p in predictions}
    counts: dict[str, list[int]] = {}
    for it in items:
        if it.agreement == "unanimous":
            continue
        p = by_id.get(it.item_id)
        if p is None or p.label is None:
            continue
        for rid, lab in zip(it.rater_ids, it.labels):
            if lab is None:
                continue
            n, a = counts.setdefault(rid, [0, 0])
            counts[rid][0] = n + 1
            counts[rid][1] = a + (1 if lab == p.label else 0)
    return [RaterAffinity(rid, n, a) for rid, (n, a) in counts.items()]


def render_affinity(affinities: list[RaterAffinity], model_name: str = "model") -> str:
    lines = [f"RATER AFFINITY on contested items ({model_name}): who does it think like?"]
    ranked = sorted(affinities, key=lambda x: -(x.agreement or 0.0))
    for a in ranked:
        pct = "n/a" if a.agreement is None else f"{a.agreement:.1%}"
        lines.append(f"  agrees with {a.rater_id:8} {pct:>7}  (n={a.n})")
    if ranked and ranked[0].agreement is not None:
        lines.append(f"  -> leans {ranked[0].rater_id}")
    return "\n".join(lines)
