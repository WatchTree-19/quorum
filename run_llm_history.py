"""Score a language model on the sovereign history panel, stratified.

The snapshot panel carries 8 contested sovereigns out of 124, which is too few
to put a confidence interval around. This panel carries 522 contested items out
of 2,697 country-quarters, sixty-five times as many, and contested accuracy is
the benchmark's headline metric. This is therefore the run that can support a
published number.

Scoring all 2,697 items is wasteful, because the unanimous stratum is already
measured precisely by the snapshot and adds little here. By default this script
takes EVERY contested item and a random sample of unanimous ones, which gets
the contested interval down to a few percentage points for roughly a third of
the calls.

    set PYTHONPATH=.
    python run_llm_history.py --provider fake --dry-run
    python run_llm_history.py --provider openai --out hist_gpt_blind.json
    python run_llm_history.py --provider openai --style named --out hist_gpt_named.json

Stratification is reported in the output so the numbers cannot be read as if
they came from the natural panel. Contested accuracy, the reliability gap and
rater affinity are all computed WITHIN stratum and are unaffected by the
sampling. Overall accuracy is not, and is labelled accordingly.

ITEMS ON THIS PANEL ARE NOT INDEPENDENT, and any interval computed as though
they were will be far too narrow. A country contributes up to 126 quarters and
its rating moves rarely. Worse, on the BLIND condition the World Bank
fundamentals are annual, so all four quarters of one country-year produce a
byte-identical prompt: on the first real run, 1,022 items carried only 495
distinct prompts and the 401 contested items only 123, which is why that run
reported 541 cache hits on a panel that had never been scored. Use
compare_runs.py, which resamples whole countries, and treat any naive
binomial interval or McNemar p-value from this panel as optimistic.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from quorum import compute_scorecard, render_report
from quorum.adapters.sovereign_history import load_items
from quorum.affinity import rater_affinity, render_affinity
from quorum.llm import PromptStyle, build_prompt, get_provider, run_model


def stratified(items, n_unanimous: int, seed: int):
    """Every contested item, plus a random sample of unanimous ones."""
    contested = [it for it in items if it.agreement != "unanimous"]
    unanimous = [it for it in items if it.agreement == "unanimous"]
    rng = random.Random(seed)
    keep = unanimous if n_unanimous >= len(unanimous) else rng.sample(unanimous, n_unanimous)
    out = contested + keep
    out.sort(key=lambda it: it.item_id)
    return out, len(contested), len(keep), len(unanimous)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/history_raw")
    p.add_argument("--provider", default="fake",
                   choices=["anthropic", "openai", "gemini", "fake"])
    p.add_argument("--model", default=None)
    p.add_argument("--style", default="blind", choices=["blind", "named", "named_undated"])
    p.add_argument("--unanimous", type=int, default=500,
                   help="How many unanimous items to sample. 0 scores contested only; "
                        "a number at or above the panel's count scores all of them.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed, reported in the output.")
    p.add_argument("--limit", type=int, default=None, help="Cap the total, for pricing a run.")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--cache-dir", default="llm_cache")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true", help="Makes no API calls.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    style = PromptStyle(args.style)
    items = load_items(args.raw_dir)
    work, n_con, n_kept, n_all_unan = stratified(items, args.unanimous, args.seed)
    if args.limit:
        work = work[: args.limit]

    no_data = sum(1 for it in work if it.inputs.get("gdp_pc_usd") is None)

    if args.dry_run:
        print(f"Panel        : {args.raw_dir}   {len(items):,} country-quarters")
        print(f"Stratified   : {n_con:,} contested (all) + {n_kept:,} of {n_all_unan:,} unanimous")
        print(f"To score     : {len(work):,} items" + (f"  (--limit {args.limit})" if args.limit else ""))
        print(f"No WDI figure: {no_data:,} items will show 'not reported' for GDP per capita")
        print(f"Provider     : {args.provider}   model: {args.model or 'provider default'}")
        blurb = {
            PromptStyle.BLIND: "   (identity and period WITHHELD)",
            PromptStyle.NAMED: "   (country and quarter SUPPLIED: full contamination probe)",
            PromptStyle.NAMED_UNDATED: "   (country supplied, PERIOD WITHHELD: isolates knowing WHEN)",
        }[style]
        print(f"Style        : {style.value}{blurb}")
        print(f"Calls        : {len(work):,}   seed {args.seed}   cache {args.cache_dir or 'disabled'}")
        print("\n" + "-" * 74 + "\nONE PROMPT AS THE MODEL WILL RECEIVE IT\n" + "-" * 74)
        print(build_prompt(work[0], style))
        print("-" * 74 + "\n\nNo API calls were made. Drop --dry-run to run it.")
        return 0

    provider = get_provider(args.provider, args.model)
    label = f"{provider.name}:{provider.model} ({style.value}, history)"
    print(f"Scoring {len(work):,} country-quarters with {label}", file=sys.stderr)

    preds, stats = run_model(work, provider, style=style,
                             cache_dir=args.cache_dir or None, sleep_s=args.sleep)
    sc = compute_scorecard(work, preds)
    # The scorecard's alpha and contested share describe whatever set it was
    # given. On a stratified set both are distorted by construction, so the
    # panel's own figures are computed separately and printed beside them.
    panel_sc = compute_scorecard(items, [])

    print()
    print(render_report(sc, model_name=label))
    print()
    print("SAMPLING")
    print(f"  Every one of the {n_con:,} contested items is scored, alongside a random")
    print(f"  {n_kept:,} of {n_all_unan:,} unanimous ones (seed {args.seed}). Contested accuracy,")
    print("  the reliability gap and affinity are within-stratum and unaffected.")
    print("  OVERALL ACCURACY IS NOT: it is on the stratified set, not the panel.")
    print("  NOR ARE THE TWO RELIABILITY FIGURES PRINTED ABOVE. Oversampling the")
    print("  contested stratum distorts both by construction. On the full panel:")
    print(f"    Krippendorff alpha  {panel_sc.label_alpha:.3f}   (stratified set: {sc.label_alpha:.3f})")
    print(f"    contested share     {panel_sc.contested_share:.1%}   (stratified set: {sc.contested_share:.1%})")
    print()
    print(render_affinity(rater_affinity(work, preds), model_name=label))
    print()
    print(stats.render())

    if args.out:
        by_id = {it.item_id: it for it in work}
        payload = {
            "model": label, "provider": provider.name, "model_id": provider.model,
            "style": style.value, "panel": "sovereign_history",
            "sampling": {"contested_scored": n_con, "unanimous_scored": n_kept,
                         "unanimous_available": n_all_unan, "seed": args.seed,
                         "panel_total": len(items),
                         "note": "overall_accuracy is on the stratified set, not the panel"},
            "n_items": sc.n_items,
            "label_alpha": panel_sc.label_alpha,
            "contested_share": panel_sc.contested_share,
            "label_alpha_stratified_set": sc.label_alpha,
            "contested_share_stratified_set": sc.contested_share,
            "overall_accuracy": sc.overall_accuracy,
            "headline_contested_accuracy": sc.headline,
            "reliability_gap": sc.reliability_gap,
            "expected_calibration_error": sc.expected_calibration_error,
            "strata": {n: {"n": s.n, "scored": s.scored, "accuracy": s.accuracy,
                           "abstain_rate": s.abstain_rate,
                           "confident_commit_rate": s.confident_commit_rate}
                       for n, s in sc.strata.items()},
            "affinity": {a.rater_id: {"n": a.n, "agreement": a.agreement}
                         for a in rater_affinity(work, preds)},
            "predictions": [
                {"item_id": p.item_id, "label": p.label, "confidence": p.confidence,
                 "agreement": by_id[p.item_id].agreement,
                 "reference": by_id[p.item_id].majority_label}
                for p in preds
            ],
            "run": {"called": stats.called, "from_cache": stats.from_cache,
                    "errors": stats.errors, "unparseable": stats.unparseable,
                    "abstained": stats.abstained, "seconds": round(stats.seconds, 1)},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
