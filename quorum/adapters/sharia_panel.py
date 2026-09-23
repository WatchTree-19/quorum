"""Adapter: Shariah compliance screening as a Quorum task.

Four screening standards (AAOIFI, MSCI, DJIM, S&P) each render a verdict on the
same firm-quarter. They are treated as four independent expert raters. This is a
real financial task with genuinely contested ground truth: the standards
disagree on roughly one firm-quarter in three.

Reads the raw fundamentals panel (ticker, q, debt, shares, price, mcap, assets)
and computes each standard's verdict. The screens are the ones implemented in
the companion research engine; the AAOIFI denominator (market capitalisation)
and threshold (0.30) were verified against that engine's own output.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from quorum.schema import Item

THIRD = 1.0 / 3.0
STANDARDS = (
    ("AAOIFI", "mcap", 0.30),
    ("MSCI", "assets", THIRD),
    ("DJIM", "mcap_avg24", THIRD),
    ("SP", "mcap_avg36", THIRD),
)
RATER_IDS = tuple(name for name, _, _ in STANDARDS)


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _tmean(vals, i, w):
    c = [x for x in vals[max(0, i - w + 1): i + 1] if not np.isnan(x)]
    return float(np.mean(c)) if c else float("nan")


def load_items(panel_csv: str | Path) -> list[Item]:
    rows = list(csv.DictReader(open(panel_csv, newline="", encoding="utf-8-sig")))
    by = defaultdict(list)
    for r in rows:
        by[r["ticker"]].append(r)

    items: list[Item] = []
    for ticker, rs in by.items():
        rs.sort(key=lambda r: r["q"])
        mc = [_f(r["mcap"]) for r in rs]
        for i, r in enumerate(rs):
            debt, assets = _f(r["debt"]), _f(r["assets"])
            denoms = {
                "assets": assets,
                "mcap": mc[i],
                "mcap_avg24": _tmean(mc, i, 8),
                "mcap_avg36": _tmean(mc, i, 12),
            }
            labels = []
            for _, den, th in STANDARDS:
                d = denoms[den]
                if np.isnan(debt) or np.isnan(d) or d <= 0:
                    labels.append(None)
                else:
                    labels.append(1 if debt / d > th else 0)
            if sum(1 for l in labels if l is not None) < 2:
                continue
            inputs = {
                "ticker": ticker,
                "period": r["q"],
                "debt": debt,
                "assets": assets,
                "mcap": mc[i],
                "mcap_avg24": denoms["mcap_avg24"],
                "mcap_avg36": denoms["mcap_avg36"],
            }
            items.append(Item(
                item_id=f"{ticker}:{r['q']}",
                task="sharia_compliance",
                inputs=inputs,
                labels=tuple(labels),
                rater_ids=RATER_IDS,
                metadata={"group": ticker},
            ))
    return items
