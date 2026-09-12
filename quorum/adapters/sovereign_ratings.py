"""Adapter: sovereign investment-grade classification as a Quorum task.

The three major agencies (S&P, Moody's, Fitch) each publish a sovereign credit
rating for the same country. They are treated as three independent expert
raters on one binary question: is this sovereign investment grade? The task is
real, the data is fully public, and the disagreement sits exactly where it
should -- at the BBB-/BB+ boundary, where "split ratings" are a documented
market phenomenon with pricing consequences.

The binary label is 1 for SPECULATIVE grade (below BBB- / Baa3), 0 for
investment grade. Default-tier ratings (SD, RD, D, C, Ca) count as speculative.
Countries rated by only two agencies are admitted (the schema requires two);
those with one rating are refused by the schema, by design.

Inputs given to the model under test are public macro fundamentals (World Bank
WDI): GDP per capita, CPI inflation, and current-account balance as a share of
GDP. The ratings themselves are never part of the inputs.
"""

from __future__ import annotations

import csv
from pathlib import Path

from quorum.schema import Item

RATER_IDS = ("SP", "MOODYS", "FITCH")

# Notch scales, best to worst. Index 0 is AAA/Aaa; investment grade is any
# notch at or above BBB- (index 9). Default-tier entries map to the bottom.
_SP_FITCH = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C",
]
_MOODYS = [
    "Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3",
    "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1", "Caa2", "Caa3", "Ca", "C",
]
_DEFAULT_TIER = {"SD", "RD", "D"}
_IG_CUTOFF = 9  # index of BBB- / Baa3


def notch(rating: str, agency: str) -> int | None:
    """Rating letter -> notch index (0 = AAA/Aaa), or None if blank/unknown."""
    r = (rating or "").strip()
    if not r:
        return None
    if r.upper() in _DEFAULT_TIER:
        return len(_SP_FITCH) - 1
    scale = _MOODYS if agency == "MOODYS" else _SP_FITCH
    try:
        return scale.index(r)
    except ValueError:
        return None


def is_speculative(rating: str, agency: str) -> int | None:
    """1 if speculative grade, 0 if investment grade, None if unrated."""
    n = notch(rating, agency)
    if n is None:
        return None
    return 1 if n > _IG_CUTOFF else 0


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_items(panel_csv: str | Path) -> list[Item]:
    """Load Quorum items from the sovereign panel CSV.

    Expected columns: country, iso3, sp, moodys, fitch, gdp_pc_usd,
    inflation_pct, current_account_pct_gdp.
    """
    items: list[Item] = []
    with open(panel_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            labels = (
                is_speculative(row.get("sp", ""), "SP"),
                is_speculative(row.get("moodys", ""), "MOODYS"),
                is_speculative(row.get("fitch", ""), "FITCH"),
            )
            if sum(1 for l in labels if l is not None) < 2:
                continue  # single-gold sovereigns are not admitted
            items.append(
                Item(
                    item_id=row["iso3"],
                    task="sovereign_investment_grade",
                    inputs={
                        "gdp_pc_usd": _f(row.get("gdp_pc_usd")),
                        "inflation_pct": _f(row.get("inflation_pct")),
                        "current_account_pct_gdp": _f(row.get("current_account_pct_gdp")),
                    },
                    labels=labels,
                    rater_ids=RATER_IDS,
                    metadata={
                        "group": row["iso3"],
                        "country": row["country"],
                        "ratings": {
                            "SP": row.get("sp", ""),
                            "MOODYS": row.get("moodys", ""),
                            "FITCH": row.get("fitch", ""),
                        },
                    },
                )
            )
    return items
