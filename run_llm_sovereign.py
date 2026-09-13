"""Score a language model on the sovereign investment-grade panel.

The panel is fully public (agency ratings as published, World Bank WDI
fundamentals), so results from this script are publishable as they stand.

    set PYTHONPATH=.
    python run_llm_sovereign.py --provider fake --dry-run
    python run_llm_sovereign.py --provider anthropic
    python run_llm_sovereign.py --provider anthropic --style named
    python run_llm_sovereign.py --provider openai --model gpt-4.1 --limit 20

API keys are read from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY) and from nowhere else.

Start with --dry-run. It prints the item count, one full prompt and the cost
shape of the run without making a single call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quorum import compute_scorecard, render_report
from quorum.adapters.sovereign_ratings import load_items
from quorum.affinity import rater_affinity, render_affinity
from quorum.llm import PromptStyle, build_prompt, get_provider, run_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default="data/sovereign_panel.csv")
    p.add_argument("--provider", default="fake",
                   choices=["anthropic", "openai", "gemini", "fake"])
    p.add_argument("--model", default=None, help="Override the provider's default model id.")
    p.add_argument("--style", default="blind", choices=["blind", "named"],
                   help="blind withholds the country name; named is the contamination probe.")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N items.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds between live calls.")
    p.add_argument("--cache-dir", default="llm_cache", help="Set to '' to disable caching.")
    p.add_argument("--out", default=None, help="Write the scorecard to this JSON file.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be sent and stop. Makes no API calls.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    style = PromptStyle(args.style)
    items = load_items(args.panel)
    work = items[: args.limit] if args.limit else items

    if args.dry_run:
        contested = [it for it in work if it.agreement != "unanimous"]
        print(f"Panel      : {args.panel}")
        print(f"Items      : {len(work):,} of {len(items):,} loaded"
              f"   contested: {len(contested):,}")
        print(f"Provider   : {args.provider}   model: {args.model or 'provider default'}")
        print(f"Style      : {style.value}"
              + ("   (country name WITHHELD)" if style is PromptStyle.BLIND
                 else "   (country name SUPPLIED: contamination probe)"))
        print(f"Calls      : {len(work):,} one-shot completions, ~300 output tokens each")
        print(f"Cache      : {args.cache_dir or 'disabled'}"
              "   (a repeat run of an unchanged experiment costs nothing)")
        print("\n" + "-" * 74 + "\nONE PROMPT AS THE MODEL WILL RECEIVE IT\n" + "-" * 74)
        print(build_prompt(work[0], style))
        print("-" * 74)
        print("\nNo API calls were made. Drop --dry-run to run it.")
        return 0

    provider = get_provider(args.provider, args.model)
    label = f"{provider.name}:{provider.model} ({style.value})"
    print(f"Scoring {len(work):,} sovereigns with {label}", file=sys.stderr)

    preds, stats = run_model(
        work, provider, style=style,
        cache_dir=args.cache_dir or None, limit=None, sleep_s=args.sleep,
    )
    sc = compute_scorecard(work, preds)

    print()
    print(render_report(sc, model_name=label))
    print()
    print(render_affinity(rater_affinity(work, preds), model_name=label))
    print()
    print(stats.render())

    if args.out:
        by_id = {it.item_id: it for it in work}
        payload = {
            "model": label,
            "provider": provider.name,
            "model_id": provider.model,
            "style": style.value,
            "n_items": sc.n_items,
            "label_alpha": sc.label_alpha,
            "contested_share": sc.contested_share,
            "overall_accuracy": sc.overall_accuracy,
            "headline_contested_accuracy": sc.headline,
            "reliability_gap": sc.reliability_gap,
            "expected_calibration_error": sc.expected_calibration_error,
            "strata": {
                name: {
                    "n": st.n, "scored": st.scored, "accuracy": st.accuracy,
                    "abstain_rate": st.abstain_rate,
                    "confident_commit_rate": st.confident_commit_rate,
                }
                for name, st in sc.strata.items()
            },
            "affinity": {
                a.rater_id: {"n": a.n, "agreement": a.agreement}
                for a in rater_affinity(work, preds)
            },
            "predictions": [
                {"item_id": p.item_id, "label": p.label, "confidence": p.confidence,
                 "agreement": by_id[p.item_id].agreement,
                 "reference": by_id[p.item_id].majority_label}
                for p in preds
            ],
            "run": {
                "called": stats.called, "from_cache": stats.from_cache,
                "errors": stats.errors, "unparseable": stats.unparseable,
                "abstained": stats.abstained, "seconds": round(stats.seconds, 1),
            },
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
