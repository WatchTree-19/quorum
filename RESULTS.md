# GPT-4.1 through Quorum: what held and what did not

13 September 2026. Four runs of GPT-4.1: the 124-sovereign snapshot and the
2,697 country-quarter history panel, each blind (country withheld) and named
(country supplied). 2,268 calls, 0 failures, 0 unparseable responses.

## The correction that has to come first

The snapshot said naming the country DESTROYED accuracy on contested items,
83.3 percent falling to 28.6 percent, and that reading was written up as the
most interesting thing the instrument had produced. It does not survive.

The contested stratum of the snapshot holds seven items. On the history panel,
with 401 scored contested items, naming the country RAISES contested accuracy
from 48.6 percent to 62.8 percent. The direction is opposite. The snapshot
result was noise, and the larger experiment is the one to believe.

That is the benchmark working as intended. It was built so that a claim about
contested items could be checked rather than asserted, and the first thing it
checked was its own headline.

## The history panel, GPT-4.1, 1,022 items scored

                              blind          named        gap (blind - named)
    unanimous accuracy        57.1%          67.7%        -10.6  [-20.3, -1.9]
    contested accuracy        48.6%          62.8%        -14.1  [-32.8, +5.2]
    abstains, unanimous       13.8%           6.0%         +7.8  [ +3.6, +12.5]
    abstains, contested       26.7%          13.0%        +13.7  [ +4.1, +24.5]
    abstains, split items     14.9%           5.0%         +9.9  [ -1.1, +24.4]
    leans                     Fitch 53.5%    S&P 61.5%

Intervals are percentile bootstraps over the 27 countries, 4,000 draws.

## Why the intervals are clustered, and why it matters

Items on this panel are not independent. A country contributes up to 126
quarters, its rating moves rarely, and on the blind condition every quarter
inside one calendar year produces a byte-identical prompt because the World
Bank fundamentals are annual. The 1,022 scored items carry only 495 distinct
prompts, and the 401 contested items only 123.

Treating quarters as independent inflates everything. The blind unanimous
versus contested accuracy gap tests at p = 0.028 naively and sits at
[-12.1, +27.1] once resampled over countries, crossing zero. McNemar likewise
reports p = 0.0001 and p = 0.0013 on the two strata above, but McNemar assumes
independent pairs and these are not, so the clustered intervals are the ones to
quote and the McNemar figures are reported only for comparison.

## What holds

NAMING THE COUNTRY ROUGHLY HALVES ABSTENTION. 13.8 to 6.0 percent on unanimous
items and 26.7 to 13.0 percent on contested ones, both intervals excluding
zero. On split items, where the agencies are evenly divided and no correct
answer exists, it falls from 14.9 to 5.0 percent.

NAMING IMPROVES ACCURACY ON SETTLED QUESTIONS. 57.1 to 67.7 percent, interval
[-20.3, -1.9], excluding zero. Recall helps where the agencies already agree,
which is what one would expect if the model is retrieving a published rating.

BLIND, THE MODEL IS AT CHANCE ON CONTESTED ITEMS. 48.6 percent with a clustered
interval of [31.5, 66.8]. Reasoning from GDP per capita, inflation and the
current account does not settle a question three rating agencies could not
settle either.

THE AFFINITY FLIP REPRODUCES. Blind the model leans Fitch, named it leans S&P,
on both the snapshot and the history panel. Two panels, same direction.

## What does not hold

The contested accuracy gain from naming is not established: [-32.8, +5.2]
crosses zero.

The abstention suppression is not specifically about contested items. It is
larger there in point estimate, 13.7 against 7.8 points, but the difference of
differences has a clustered interval of [-0.1, +0.2] and crosses zero. Naming
suppresses abstention broadly, not selectively.

The split-item abstention gap crosses zero at [-1.1, +24.4].

## The claim worth making

Naming the country makes the model more accurate and much less willing to say
it does not know, including on items where three rating agencies actively
disagree and on items where they are evenly split and there is no answer to be
had.

The accuracy gain is real on settled questions. On contested ones the accuracy
gain is not established whilst the confidence gain is. Contamination therefore
shows up as confidence before it shows up as accuracy, and a leaderboard score
cannot separate the two. A model-risk function evaluating a vendor on headline
accuracy would see the 67.7 percent and not the collapse in abstention that
came with it.

That is a narrower claim than the one the snapshot appeared to support. It is
also the one that survived being tested.

---

# Forward scoring: the contested stratum against a realised outcome

Added 13 September 2026, after the runs above.

## Why

Everything above scores contested items against the contemporaneous majority of
agencies. That is consensus tracking. Agreeing with two agencies out of three is
not being right, because on a contested item there is no external truth to be
right about, and this repository has conceded that from the start.

A panel that runs through time contains one. A disagreement spell ends: when the
agencies split over whether a sovereign is investment grade, one side is
eventually vindicated because the others move to meet it. The verdict they
converge on is a realised outcome, not a vote. The history panel carries 47
spells, 39 closing inside the data, over 424 contested country-quarters.

## Two corrections made whilst building this, both worth recording

FIRST. The design anticipated that a model told the country AND the quarter
could simply recall how that episode ended, making its score a hindsight
ceiling. A third condition was built to test it, NAMED_UNDATED, which supplies
the country and withholds the period. Knowing the date turns out to be worth
almost nothing: +6.3 points over the undated condition with an interval that
crosses zero, and +0.0 points on the full item set. The concern was real enough
to test and did not survive testing.

SECOND, and more embarrassing. The obvious trivial baseline is "always say
whatever this country usually ends up as", since rating states are persistent.
Computed over the same items it was scored on, that rule reached 86.8 percent
and appeared to destroy the metric. It was leaking: the modal verdict was fitted
on the items being scored. Out of fold, taking each spell's prior from that
country's OTHER spells, the same rule collapses to 44.8 percent, because
consecutive spells tend to settle in opposite directions. The rule is
anti-predictive, and the metric survives. The in-sample figure is recorded here
because a benchmark that hides its own near-misses is not worth much.

## The result

On the 252 contested items sitting in countries with at least two resolved
spells, which is where an out-of-fold baseline can compete at all:

                                      accuracy   clustered 95% CI
    country base rate, out-of-fold      44.8%    [19.1, 69.6]
    GPT-4.1 blind                       46.9%    [35.1, 56.9]
    GPT-4.1 named, no date              64.3%    [50.5, 79.6]
    GPT-4.1 named and dated             71.9%    [61.4, 83.1]
    the agencies' own majority          77.8%    [61.4, 90.9]

Intervals bootstrap over countries, 4,000 draws, for the reason set out above.

Knowing the country is worth +10.9 points over blind, with the lower bound of
the interval sitting on zero, so it is marginal rather than established. Knowing
the date on top of that is worth +6.3 and is not established.

## What it says

On a forecasting task with a realised outcome, a frontier model told which
country it is looking at reaches roughly two thirds. That is well clear of a
trivial heuristic and well short of the humans whose disagreement created the
question in the first place. The agencies' own lagging, one-at-a-time consensus
beats every model condition tested.

One caveat on that comparison: the majority is scored on 198 items against the
models' 231 to 238, because it has no verdict on evenly split items. Treat 77.8
percent as indicative rather than exact.

An earlier reading of the full 424-item set had the named condition at 67.5
percent against the majority's 61.0 percent, which looked like the model
beating the agencies. It was an artefact of scoring the two on different item
sets. On the matched subset the ordering reverses.
