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
