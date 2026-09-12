"""Boundary crossings and disagreement spells in the sovereign history panel.

The question a snapshot cannot answer: when a country crosses the
investment-grade boundary, HOW LONG do the agencies disagree about which side
it is on? Every quarter inside such a spell is a live question -- an index
membership, a collateral rule and a mandate all resolve it differently
depending on which agency the document names.

Usage:
    PYTHONPATH=. python analysis_crossings.py
"""

from __future__ import annotations

from collections import defaultdict

from grader_reliability import krippendorff_alpha
import numpy as np

from quorum.adapters.sovereign_history import load_items

RAW = "data/history_raw"


def main() -> None:
    items = load_items(RAW)
    print(f"Country-quarter items: {len(items):,} across "
          f"{len({it.metadata['group'] for it in items})} boundary sovereigns "
          f"(1995Q1-2026Q2, three agencies forward-filled)\n")

    contested = [it for it in items if it.agreement != "unanimous"]
    ratings = np.array(
        [[np.nan if l is None else float(l) for l in it.labels] for it in items]
    )
    alpha = krippendorff_alpha(ratings, level="nominal")
    print(f"Krippendorff alpha across agencies : {alpha:.3f}")
    print(f"Contested country-quarters         : {len(contested):,} ({len(contested)/len(items):.1%})")
    splits = sum(1 for it in items if it.agreement == "split")
    print(f"Even-tie country-quarters          : {splits:,} ({splits/len(items):.1%})\n")

    # Disagreement spells: consecutive contested quarters per country.
    by_country: dict[str, list] = defaultdict(list)
    for it in items:
        by_country[it.metadata["group"]].append(it)
    spells = []
    for iso3, its in by_country.items():
        its.sort(key=lambda i: i.metadata["quarter"])
        run = 0
        for it in its:
            if it.agreement != "unanimous":
                run += 1
            elif run:
                spells.append((iso3, run))
                run = 0
        if run:
            spells.append((iso3, run + 0))  # spell still open at panel end
    lengths = [n for _, n in spells]
    lengths.sort()
    print(f"Disagreement spells (consecutive contested quarters): {len(spells)}")
    print(f"  median length : {lengths[len(lengths)//2]} quarters")
    print(f"  mean length   : {sum(lengths)/len(lengths):.1f} quarters")
    print(f"  longest       : {max(lengths)} quarters "
          f"({[c for c, n in spells if n == max(lengths)][0]})")
    total_contested_q = sum(lengths)
    print(f"  total quarters inside a spell: {total_contested_q:,}\n")

    print("Longest spells (country, quarters the agencies disagreed):")
    for iso3, n in sorted(spells, key=lambda s: -s[1])[:12]:
        print(f"  {iso3}  {n:3d} quarters  (~{n/4:.1f} years)")

    # Alpha by year: is the parliament getting more or less agreed?
    print("\nAgreement through time (alpha by year):")
    by_year: dict[str, list] = defaultdict(list)
    for it in items:
        by_year[it.metadata["quarter"][:4]].append(it)
    for year in sorted(by_year):
        if int(year) % 3 != 0 and year not in (min(by_year), max(by_year)):
            continue
        yr_items = by_year[year]
        arr = np.array([[np.nan if l is None else float(l) for l in it.labels] for it in yr_items])
        a = krippendorff_alpha(arr, level="nominal")
        c = sum(1 for it in yr_items if it.agreement != "unanimous") / len(yr_items)
        print(f"  {year}: alpha {a:.3f}  contested {c:.1%}  (n={len(yr_items)})")


if __name__ == "__main__":
    main()
