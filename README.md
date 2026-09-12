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

## Author and citation

Sandeep Singh Rai (ORCID 0009-0001-3360-9205). See CITATION.cff; a methods
paper is on SSRN. Comments and task-family proposals are welcome as issues.

## Design rules

Single-gold items are refused at the schema level. Abstention is never scored
as error. Ground-truth reliability (Krippendorff alpha, contested share) is
printed before any model number. The headline metric is accuracy on contested
items, not overall accuracy.
