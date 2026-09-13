"""Rescore existing history runs against the verdict the agencies settled on.

No API calls. This reads result JSONs that already carry per-item predictions
and scores their contested stratum against a realised outcome instead of the
contemporaneous majority, which is what the published headline uses.

    python forward_analysis.py hist_gpt_blind.json hist_gpt_named.json

Intervals are percentile bootstraps over countries, for the reason set out in
compare_runs.py: quarters within a country are not independent, and on the
blind condition quarters within a year are not even distinct prompts.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from quorum.adapters.sovereign_history import load_items
from quorum.forward import find_spells, forward_score, render_forward, settled_verdicts
from quorum.schema import Prediction

RAW = "data/history_raw"
DRAWS = 4000


def cluster_of(item_id: str) -> str:
    return item_id.split(":", 1)[0]


def bootstrap(rows: list[dict], stat, seed: int = 0) -> tuple[float | None, float | None]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[cluster_of(r["item_id"])].append(r)
    keys = list(groups)
    if len(keys) < 2:
        return (None, None)
    rng = random.Random(seed)
    draws = []
    for _ in range(DRAWS):
        sample: list[dict] = []
        for _ in keys:
            sample += groups[rng.choice(keys)]
        v = stat(sample)
        if v is not None:
            draws.append(v)
    if not draws:
        return (None, None)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    items = load_items(RAW)
    verdicts = settled_verdicts(items)
    spells = find_spells(items)
    items_by_id = {it.item_id: it for it in items}

    print("=" * 74)
    print("FORWARD SCORING: the contested stratum against a realised outcome")
    print("=" * 74)
    print(f"  spells {len(spells)}, of which {sum(1 for s in spells if s.resolved)} closed inside the panel")
    print(f"  contested items carrying a settled verdict: {len(verdicts):,}")

    maj_rows = [
        {"item_id": i, "hit": int(items_by_id[i].majority_label == v)}
        for i, v in verdicts.items() if items_by_id[i].majority_label is not None
    ]
    def rate(sample):
        return sum(r["hit"] for r in sample) / len(sample) if sample else None
    mlo, mhi = bootstrap(maj_rows, rate)
    print(f"\n  BASELINE, the contemporaneous majority of agencies: "
          f"{rate(maj_rows):.1%}  clustered CI [{mlo:.1%}, {mhi:.1%}]  on n={len(maj_rows)}")
    print("  It lags during a transition because agencies move one at a time,")
    print("  but it is not a coin flip and it is the number to beat.\n")

    for path in paths:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if "predictions" not in blob:
            print(f"  {path}: no per-item predictions, skipping")
            continue
        preds = [Prediction(item_id=p["item_id"], label=p["label"],
                            confidence=p["confidence"]) for p in blob["predictions"]]
        res = forward_score(items, preds)
        rows = [
            {"item_id": i, "hit": int(l == o)}
            for (i, o, l) in res.outcomes if l is not None
        ]
        lo, hi = bootstrap(rows, rate)
        style = blob.get("style", "?")
        acc = res.accuracy
        print(f"  {style.upper():14} {acc:.1%}  clustered CI [{lo:.1%}, {hi:.1%}]"
              f"   committed {res.scored}, abstained {res.abstained}")
        if style == "named":
            print("                 read as a HINDSIGHT CEILING: it was given the")
            print("                 quarter and can recall how the episode ended")
    print()
    print("  The gap between NAMED and NAMED_UNDATED is the value of knowing WHEN.")
    print("  On a forward target that is very nearly pure memorisation, because")
    print("  knowing a country's name can legitimately improve a forecast from")
    print("  its fundamentals whilst knowing the date cannot.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
