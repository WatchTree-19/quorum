"""Adapter: the sovereign investment-grade panel THROUGH TIME.

The cross-sectional pilot observes the parliament today; this adapter observes
it through time. Each agency's rating actions are forward-filled to a quarterly
panel, and each country-quarter where at least two agencies held a rating
becomes a Quorum item on the same binary question: investment grade or not.

The reason this panel exists: the moments that matter are the boundary
CROSSINGS -- the quarters in which a country moves between investment grade
and speculative, when the agencies routinely disagree about which side the
country is on. A snapshot sees eight contested sovereigns; the panel sees
every disagreement spell around every crossing since the 1990s.

Raw inputs are per-country files of dated rating actions
(data/history_raw/<ISO3>.txt, lines "AGENCY | YYYY-MM-DD | RATING").
"""

from __future__ import annotations

from pathlib import Path

from quorum.adapters.sovereign_ratings import is_speculative
from quorum.schema import Item

RATER_IDS = ("SP", "MOODYS", "FITCH")
_AGENCY_KEY = {"S&P": "SP", "MOODY'S": "MOODYS", "MOODYS": "MOODYS", "FITCH": "FITCH"}


def _parse_actions(path: Path) -> dict[str, list[tuple[str, str]]]:
    """File -> {agency: [(date, rating), ...] sorted by date}."""
    actions: dict[str, list[tuple[str, str]]] = {k: [] for k in RATER_IDS}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        agency = _AGENCY_KEY.get(parts[0].upper())
        if agency is None:
            continue
        actions[agency].append((parts[1], parts[2]))
    for k in actions:
        actions[k].sort()
    return actions


def _quarters(start: str, end: str) -> list[str]:
    (sy, sq), (ey, eq) = (int(start[:4]), (int(start[5:7]) - 1) // 3 + 1), (
        int(end[:4]), (int(end[5:7]) - 1) // 3 + 1)
    out, y, q = [], sy, sq
    while (y, q) <= (ey, eq):
        out.append(f"{y}Q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def _q_of(date: str) -> str:
    return f"{int(date[:4])}Q{(int(date[5:7]) - 1) // 3 + 1}"


def load_items(
    raw_dir: str | Path,
    start: str = "1995-01-01",
    end: str = "2026-06-30",
) -> list[Item]:
    """Build country-quarter items from the per-country action files."""
    raw_dir = Path(raw_dir)
    all_q = _quarters(start, end)
    items: list[Item] = []
    for path in sorted(raw_dir.glob("*.txt")):
        iso3 = path.stem
        actions = _parse_actions(path)
        # Forward-fill each agency's IG/speculative state by quarter.
        state: dict[str, dict[str, int | None]] = {}
        for agency, acts in actions.items():
            cur: int | None = None
            i = 0
            col: dict[str, int | None] = {}
            for q in all_q:
                while i < len(acts) and _q_of(acts[i][0]) <= q:
                    cur = is_speculative(acts[i][1], agency)
                    i += 1
                col[q] = cur
            state[agency] = col
        for q in all_q:
            labels = tuple(state[a][q] for a in RATER_IDS)
            if sum(1 for l in labels if l is not None) < 2:
                continue
            items.append(
                Item(
                    item_id=f"{iso3}:{q}",
                    task="sovereign_investment_grade_history",
                    inputs={"iso3": iso3, "quarter": q},
                    labels=labels,
                    rater_ids=RATER_IDS,
                    metadata={"group": iso3, "quarter": q},
                )
            )
    return items
