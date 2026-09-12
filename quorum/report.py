"""Human-readable rendering of a Scorecard."""

from __future__ import annotations

from quorum.scorecard import Scorecard


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _num(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_report(sc: Scorecard, model_name: str = "model under test") -> str:
    L = []
    L.append("=" * 74)
    L.append(f"QUORUM SCORECARD  |  task: {sc.task}  |  {model_name}")
    L.append("=" * 74)
    L.append(f"Items: {sc.n_items:,}   Experts per item: up to {sc.max_raters}   "
             f"Missing predictions: {sc.missing_predictions:,}")
    L.append("")
    L.append("GROUND TRUTH RELIABILITY (a property of the benchmark, reported first)")
    L.append(f"  Krippendorff alpha across experts : {_num(sc.label_alpha)}")
    L.append(f"  Contested items (not unanimous)   : {_pct(sc.contested_share)}")
    L.append(f"  Split items (even tie)            : {_pct(sc.split_share)}")
    L.append("")
    L.append("MODEL BEHAVIOUR BY AGREEMENT STRATUM")
    hdr = f"  {'stratum':10} {'n':>7} {'accuracy':>9} {'abstain':>8} {'confident':>10} {'mean conf':>10}"
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for s in ("unanimous", "majority", "split"):
        st = sc.strata[s]
        acc = _pct(st.accuracy) if s != "split" else "(no answer)"
        L.append(f"  {s:10} {st.n:7,} {acc:>9} {_pct(st.abstain_rate):>8} "
                 f"{_pct(st.confident_commit_rate):>10} {_num(st.mean_confidence_in_label, 2):>10}")
    L.append("")
    L.append("AGGREGATES")
    L.append(f"  Overall accuracy (what a leaderboard would show)   : {_pct(sc.overall_accuracy)}")
    L.append(f"  HEADLINE: accuracy on contested (majority) items    : {_pct(sc.headline)}")
    L.append(f"  Reliability gap  acc(unanimous) - acc(majority)     : "
             f"{'n/a' if sc.reliability_gap is None else f'{sc.reliability_gap:+.1%}'}")
    L.append(f"  Expected calibration error                          : {_num(sc.expected_calibration_error)}")
    L.append("")
    L.append("READ-OUT")
    L.append("  A large reliability gap means the headline accuracy is riding the easy")
    L.append("  cases. On split items there is no correct answer; a trustworthy model")
    L.append("  abstains or reports low confidence rather than committing.")
    L.append("=" * 74)
    return "\n".join(L)
