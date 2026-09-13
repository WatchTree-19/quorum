"""Compare two Quorum runs over the same items, correctly paired.

The blind and named runs score the SAME items, so comparing them with a
two-sample test throws away the pairing and most of the power. McNemar's test
on the discordant pairs is the right instrument, and it is what turns "the
numbers look different" into a claim that survives review.

    python compare_runs.py results_gpt.json results_gpt_named.json

Both files must carry a "predictions" block, which the runners write.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (centre - half, centre + half)


def load(path: str) -> dict:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if "predictions" not in blob:
        raise SystemExit(
            f"{path} has no per-item predictions. It was written by an older "
            f"version of the runner; re-run it (the cache makes that free)."
        )
    return blob


def outcome(pred: dict) -> int | None:
    """1 correct, 0 wrong, None if it abstained or the item has no right answer."""
    if pred["label"] is None or pred["reference"] is None:
        return None
    return int(pred["label"] == pred["reference"])


def compare(a: dict, b: dict, stratum: str) -> None:
    pa = {p["item_id"]: p for p in a["predictions"] if p["agreement"] == stratum}
    pb = {p["item_id"]: p for p in b["predictions"] if p["agreement"] == stratum}
    shared = sorted(set(pa) & set(pb))

    ka = sum(1 for i in shared if outcome(pa[i]) == 1)
    na = sum(1 for i in shared if outcome(pa[i]) is not None)
    kb = sum(1 for i in shared if outcome(pb[i]) == 1)
    nb = sum(1 for i in shared if outcome(pb[i]) is not None)

    print(f"\n{stratum.upper()} stratum, {len(shared)} shared items")
    for tag, k, n in ((a['style'], ka, na), (b['style'], kb, nb)):
        if n:
            lo, hi = wilson(k, n)
            print(f"  {tag:8} {k:4}/{n:<4} = {k/n:6.1%}   95% CI [{lo:5.1%}, {hi:5.1%}]")
        else:
            print(f"  {tag:8} nothing scored")

    # McNemar on items where BOTH runs committed: the pairing is the point.
    both = [i for i in shared if outcome(pa[i]) is not None and outcome(pb[i]) is not None]
    a_only = sum(1 for i in both if outcome(pa[i]) == 1 and outcome(pb[i]) == 0)
    b_only = sum(1 for i in both if outcome(pa[i]) == 0 and outcome(pb[i]) == 1)
    disc = a_only + b_only
    print(f"  paired on {len(both)} items where both committed: "
          f"{a_only} only {a['style']} right, {b_only} only {b['style']} right")
    if disc == 0:
        print("  McNemar: no discordant pairs, nothing to test")
    else:
        p = binomtest(a_only, disc, 0.5).pvalue
        verdict = "SIGNIFICANT at 5 percent" if p < 0.05 else "not significant"
        print(f"  McNemar exact p = {p:.4f}  ({disc} discordant pairs) -> {verdict}")
        if disc < 10:
            print("  NOTE: fewer than ten discordant pairs. The test cannot reach")
            print("  significance on a small imbalance here however large the effect;")
            print("  read a null as underpowered, not as evidence of no difference.")

    # Abstention is a reported behaviour in its own right, not an error.
    for tag, pr in ((a["style"], pa), (b["style"], pb)):
        ab = sum(1 for i in shared if pr[i]["label"] is None)
        print(f"  {tag:8} abstained on {ab}/{len(shared)} = {ab/len(shared):.1%}"
              if shared else f"  {tag}: no items")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    a, b = load(argv[0]), load(argv[1])
    if a["style"] == b["style"]:
        print(f"Both files are the '{a['style']}' style. Nothing to compare.")
        return 2

    print("=" * 74)
    print(f"PAIRED COMPARISON  {a['model']}  vs  {b['model']}")
    print("=" * 74)
    for stratum in ("unanimous", "majority"):
        compare(a, b, stratum)

    print("\nSPLIT ITEMS (no correct answer exists; behaviour only)")
    for blob in (a, b):
        st = blob["strata"]["split"]
        if st["n"]:
            print(f"  {blob['style']:8} n={st['n']:4}  abstained {st['abstain_rate']:.1%}   "
                  f"committed confidently {st['confident_commit_rate']:.1%}")
    print("\nRATER AFFINITY on contested items")
    for blob in (a, b):
        order = sorted(blob["affinity"].items(), key=lambda kv: -(kv[1]["agreement"] or 0))
        parts = " ".join(f"{k} {v['agreement']:.1%}" for k, v in order)
        print(f"  {blob['style']:8} {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
