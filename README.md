# Quorum

A benchmark for financial AI under contested ground truth. Every item carries a
quorum of independent expert labels rather than a single gold answer; models are
scored on how they behave where experts agree, where they lean, and where they
split. See WHITEPAPER.md for the full design and the pilot results.

## Layout

    quorum/            the harness: schema, scorecard, report, rater affinity
    grader_reliability/  vendored reliability machinery (Krippendorff alpha, agreement)
    quorum/adapters/   dataset adapters:
                         sharia_panel.py        Shariah screening, 4 standards as raters
                         sovereign_ratings.py   sovereign investment grade, 3 agencies as raters
                         sovereign_history.py   the same question through time (country-quarters)
    baselines/         out-of-fold logistic baselines (informed / generalist / abstaining)
    data/              sovereign panels, fully public (agency ratings + World Bank WDI;
                       history_raw/ holds each boundary country's dated rating actions)
    tests/             offline deterministic tests
    run_pilot.py       end-to-end Sharia pilot on the private panel
    run_pilot_sovereign.py  end-to-end sovereign pilot on the public panel
    analysis_crossings.py   boundary crossings and disagreement spells, 1995-2026
    quorum/llm/             the LLM harness: prompts, providers, cached runner
    run_llm_sovereign.py    score a frontier model on the public sovereign panel
    run_llm_history.py      the same, stratified over the history panel
    compare_runs.py         paired (McNemar) comparison of two runs

## Run

Needs numpy and scikit-learn; the reliability machinery (grader_reliability,
Krippendorff's alpha cross-checked against the reference implementation) is
vendored in this repository. From the repository root:

    set PYTHONPATH=.
    python -m pytest tests -q
    python run_pilot_sovereign.py data\sovereign_panel.csv
    python analysis_crossings.py

The sovereign panels in data/ are built entirely from public sources (agency
sovereign ratings as publicly listed, September 2026; World Bank WDI
fundamentals; dated agency rating actions for the history panel), so both
sovereign pilots run out of the box. The Shariah pilot's firm-quarter panel is
private research data and is not distributed; the adapter ships so that the
pilot is reproducible by anyone holding an equivalent panel:

    python run_pilot.py path\to\panel_wide.csv

## Scoring a language model

The pilots above run statistical baselines. To put a frontier model through the
same scorecard, on the public sovereign panel:

    set PYTHONPATH=.
    python run_llm_sovereign.py --provider fake --dry-run
    python run_llm_sovereign.py --provider anthropic
    python run_llm_sovereign.py --provider openai --model gpt-4.1 --out results_gpt41.json

Providers are `anthropic`, `openai`, `gemini`, and `fake`, which is a
deterministic offline rule used by the tests and by `--dry-run`. API keys are
read from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`) and from nowhere else; no key is ever read from disk, logged,
or written into the cache. Completions are cached under `llm_cache/`, keyed by
provider, model, prompt style and prompt text, so re-running an unchanged
experiment costs nothing whilst changing the prompt correctly invalidates it.
`--dry-run` prints the item count and one full prompt without making a call,
and `--limit` prices a run before you commit to the whole panel.

**The country is not named.** Every sovereign in the panel has a published
rating that is almost certainly in the training data of any frontier model, so
naming it would let the model recall the answer instead of reasoning from the
fundamentals. The default `blind` style therefore supplies GDP per capita,
inflation and the current-account balance and withholds every identifier. The
`named` style exists as the contamination probe, and the gap between the two is
itself a measurement: it is how much of a model's apparent skill is recall.
Three tests assert the blindness, and each was confirmed to fail when the
identifier was deliberately put back.

Failed calls and unparseable responses are recorded as abstentions and counted
in the run statistics, which print beneath the scorecard. That matters because
Quorum never scores an abstention as an error, so a broken run would otherwise
read as a thoughtful one.

### The history panel, which is where the headline number lives

The snapshot carries 8 contested sovereigns out of 124, which is too few to put
a usable interval around. The history panel carries 522 contested items out of
2,697 country-quarters, and contested accuracy is the headline metric, so this
is the run that can support a published figure.

    python run_llm_history.py --provider fake --dry-run
    python run_llm_history.py --provider openai --out hist_gpt_blind.json
    python run_llm_history.py --provider openai --style named --out hist_gpt_named.json

By default it scores every contested item plus 500 sampled unanimous ones, so
roughly a third of the calls buy the whole of the headline metric. Contested
accuracy, the reliability gap and rater affinity are within-stratum and are
unaffected by that sampling; overall accuracy is not, and the run says so.
Ground-truth reliability is reported for the panel as well as for the
stratified set, because oversampling the contested stratum drags alpha from
0.695 down to 0.224 by construction and the stratified figure on its own would
misdescribe the benchmark.

Items on this panel carry World Bank WDI fundamentals (`data/wdi_history.json`,
provenance in `data/SOURCES.md`) so that the blind condition is possible at all.
Blind withholds the quarter as well as the country here, since a period plus a
distinctive inflation print names a country to anyone who knows the era.

### Comparing two runs

    python compare_runs.py hist_gpt_blind.json hist_gpt_named.json

Blind and named score the same items, so the comparison is paired and
`compare_runs.py` uses McNemar's exact test on the discordant pairs rather than
treating the runs as two independent samples. It reports Wilson intervals per
stratum, abstention rates, split-item behaviour and affinity side by side, and
it says out loud when a null result is merely underpowered.

## Author and citation

Sandeep Singh Rai (ORCID 0009-0001-3360-9205). See CITATION.cff; a methods
paper is on SSRN. Comments and task-family proposals are welcome as issues.

## Design rules

Single-gold items are refused at the schema level. Abstention is never scored
as error. Ground-truth reliability (Krippendorff alpha, contested share) is
printed before any model number. The headline metric is accuracy on contested
items, not overall accuracy.
