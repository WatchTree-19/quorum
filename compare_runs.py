"""Compare two Quorum runs over the same items, paired and cluster-robust.

Two corrections live here, and the second was found the hard way.

PAIRING. The blind and named runs score the SAME items, so a two-sample test
throws away the pairing and most of the power. McNemar's test on the discordant
pairs is the right instrument for that.

CLUSTERING, which matters more. Items on the history panel are NOT independent.
A country contributes up to 126 quarters, its rating moves rarely, and on the
BLIND condition every quarter inside one calendar year produces a byte-identical
prompt, because the fundamentals are annual. On the first blind history run,
1,022 items carried only 495 distinct prompts and the contested stratum's 401
items carried 123. Treating quarters as independent therefore understates every
interval badly: the unanimous-minus-contested accuracy gap looked significant at
p = 0.028 and, bootstrapped over countries, sat at [-12.1, +27.1] points and
crossed zero. So accuracy figures here are reported with a cluster bootstrap
over countries as well as the naive interval, and the two are printed side by
side so the difference is visible rather than assumed away.

Clustering is inferred from the item id: "AZE:2010Q2" clusters on AZE, and an id
with no colon is its own cluster, which is the right behaviour for the
cross-sectional panel where each item is a different country.

    python compare_runs.py results_gpt.json results_gpt_named.json
    python compare_runs.py hist_gpt_blind.json hist_gpt_named.json

Both files must carry a "predictions" block, which the runners write.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

BOOTSTRAP_DRAWS = 4000


def cluster_of(item_id: str) -> str:
    """Cluster key for an item. 'AZE:2010Q2' -> 'AZE'; 'AZE' -> 'AZE'."""
    return item_id.split(":", 1)[0]


def _accuracy(rows: list[dict]) -> float | None:
    scored = [r for r in rows if r["label"] is not None and r["reference"] is not None]
    if not scored:
        return None
    return sum(1 for r in scored if r["label"] == r["reference"]) / len(scored)


def cluster_bootstrap(rows: list[dict], stat, seed: int = 0):
    """Percentile CI for `stat`, resampling whole clusters with replacement.

    Returns (lo, hi, n_clusters), or (None, None, n) if the statistic could not
    be computed on enough draws.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[cluster_of(r["item_id"])].append(r)
    keys = list(groups)
    if len(keys) < 2:
        return (None, None, len(keys))
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample: list[dict] = []
        for _ in keys:
            sample += groups[rng.choice(keys)]
        value = stat(sample)
        if value is not None:
            draws.append(value)
    if len(draws) < BOOTSTRAP_DRAWS // 2:
        return (None, None, len(keys))
    draws.sort()
    return (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))], len(keys))


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

    clusters = {cluster_of(i) for i in shared}
    prompts_note = ""
    if len(clusters) < len(shared):
        prompts_note = f", {len(clusters)} clusters"
    print(f"\n{stratum.upper()} stratum, {len(shared)} shared items{prompts_note}")
    for tag, k, n, rows in ((a['style'], ka, na, [pa[i] for i in shared]),
                            (b['style'], kb, nb, [pb[i] for i in shared])):
        if not n:
            print(f"  {tag:8} nothing scored")
            continue
        lo, hi = wilson(k, n)
        clo, chi, ncl = cluster_bootstrap(rows, _accuracy)
        line = f"  {tag:8} {k:4}/{n:<4} = {k/n:6.1%}   naive CI [{lo:5.1%}, {hi:5.1%}]"
        if clo is not None:
            line += f"   clustered CI [{clo:5.1%}, {chi:5.1%}] over {ncl} countries"
        print(line)

    # The gap between the two runs, bootstrapped over whole countries. This is
    # the number to quote; the naive intervals above are printed only so the
    # size of the correction is visible.
    if na and nb:
        paired_rows = [{"item_id": i, "a": pa[i], "b": pb[i]} for i in shared]

        def gap(sample):
            ra = [r["a"] for r in sample]
            rb = [r["b"] for r in sample]
            aa, ab = _accuracy(ra), _accuracy(rb)
            return None if (aa is None or ab is None) else aa - ab

        glo, ghi, ncl = cluster_bootstrap(paired_rows, gap)
        point = (ka / na) - (kb / nb)
        if glo is not None:
            crosses = glo < 0 < ghi
            print(f"  gap {a['style']} minus {b['style']}: {point:+.1%}   "
                  f"clustered CI [{glo:+.1%}, {ghi:+.1%}] -> "
                  + ("crosses zero, NOT established" if crosses else "excludes zero"))

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

    # Abstention is a reported behaviour in its own right, not an error, and on
    # the evidence so far it is the more robust signal of the two.
    if shared:
        for tag, pr in ((a["style"], pa), (b["style"], pb)):
            ab = sum(1 for i in shared if pr[i]["label"] is None)
            print(f"  {tag:8} abstained on {ab}/{len(shared)} = {ab/len(shared):.1%}")

        paired_rows = [{"item_id": i, "a": pa[i], "b": pb[i]} for i in shared]

        def abstain_gap(sample):
            if not sample:
                return None
            ra = sum(1 for r in sample if r["a"]["label"] is None) / len(sample)
            rb = sum(1 for r in sample if r["b"]["label"] is None) / len(sample)
            return ra - rb

        alo, ahi, _ = cluster_bootstrap(paired_rows, abstain_gap)
        if alo is not None:
            point = abstain_gap(paired_rows)
            crosses = alo < 0 < ahi
            print(f"  abstention gap {a['style']} minus {b['style']}: {point:+.1%}   "
                  f"clustered CI [{alo:+.1%}, {ahi:+.1%}] -> "
                  + ("crosses zero, NOT established" if crosses else "excludes zero"))


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
