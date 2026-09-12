**'Quorum'**

*A benchmark for financial AI under contested ground truth*

**Sandeep Singh Rai**

*Design document, v0.4, September 2026. Comments welcome.*

---

## The thesis

Financial institutions are placing language models into credit assessment, compliance screening, research and suitability workflows, and they are selecting and approving those models on benchmark accuracy. Every benchmark they can consult shares one design assumption, being that each question has a single correct answer against which the model is marked. In finance that assumption is false in exactly the places where the decisions are hard. Whether a promotion breaches a principle, whether a borrower is investment grade, whether a firm is compliant with a mandate, whether a disclosure is adequate: on questions of this kind, qualified experts disagree with one another at material rates, and the disagreement is not noise to be averaged away but a property of the domain.

A benchmark built on single gold labels therefore does two damaging things at once. It hides the unreliability of its own answer key, and it rewards models for agreeing with whichever expert happened to write the key. The result is a familiar failure: a model that scores well overall whilst being close to chance on precisely the contested cases a deployment turns on. We have measured this directly. A calibrated harm classifier we built for Microsoft's PyRIT scored 100% where three human raters agreed and 67% where they split, and its headline figure of 89.8% was, in effect, a weighted average of a solved problem and an unsolved one.

Quorum inverts the assumption. Every item carries a quorum of independent expert labels rather than a single gold answer, and a model is scored on how it behaves where the experts agree, where they lean, and where they genuinely split. The reliability of the ground truth is measured and published as part of the benchmark itself, whilst the headline metric is accuracy on the contested stratum, because that is where a financial decision is actually difficult and that is where trust is actually spent. Put in one line: every benchmark to date assumes an oracle; Quorum assumes a parliament.

Two consequences follow that are easy to miss. First, the benchmark accumulates something beyond scores: every task family added maps where professional consensus ends, item by item, and no such map of finance exists anywhere. The scores are the visible product; the growing catalogue of measured disagreement is the durable one. Second, Quorum cannot saturate the way benchmarks die. When models exhaust the unanimous stratum, the reliability gap becomes the entire signal, and the split stratum never had an accuracy to max out, only behaviour. Most benchmarks depreciate as capability rises; this one is built to become more discriminating.

## Why nothing existing does this

The financial benchmark literature is active and none of it addresses the problem. FinBen and PIXIU assemble broad task suites over single-labelled datasets. FINESSE-Bench and FinTradeBench extend coverage into domain knowledge and trading-style reasoning, again against fixed answer keys. FinanceBench does the same for filings question-answering. These are useful measures of capability and none of them reports the reliability of its own labels, conditions accuracy on expert agreement, or scores behaviour on items where no answer exists.

Two recent lines of work come closer and stop short. A 2026 system-level critique ("Benchmarks Are Not Validation") argues that benchmark scores are being misused as deployment evidence in finance and recommends multiple judges with agreement checks, which concedes the premise whilst building no benchmark on it. A four-axis study of LLM judges on FCA financial-promotion principles finds two judges from different model families agreeing at kappa 0.16 and a 120B judge losing 47 accuracy points under adversarial keyword stuffing, which demonstrates both that the ground truth in principle-based regulation is contested and that judge robustness collapses exactly there. Meanwhile MLCommons' AILuminate, the most institutional AI benchmark effort in existence, covers general chat safety and has no finance vertical at all.

The gap, stated plainly: there is no benchmark anywhere that evaluates financial AI against measured-reliability, multi-expert ground truth. That is the whole of what Quorum is.

## The design

**Item schema.** An item is a task input plus at least two independent expert labels, with the experts named. Single-gold items are refused at the schema level; this is enforced in code, not in guidance. Items are stratified by what the quorum did: unanimous, majority (a leaning with dissent), or split (an even tie).

**The scorecard.** For each system under test, Quorum reports, side by side: the reliability of the ground truth itself (Krippendorff's alpha across the expert labels, and the contested share of items); accuracy on unanimous items and on majority items, and the gap between them, which we call the reliability gap; behaviour on split items, where there is no right answer to be had, scored as abstention rate and confident-commit rate rather than accuracy; and calibration, when the system supplies confidences. Overall accuracy is reported too, precisely so a reader can see how much it flatters.

**The headline.** The leaderboard number is accuracy on contested (majority) items. A large reliability gap is a warning label: the model's overall figure is riding the easy cases. A high confident-commit rate on split items is a second warning label: the system asserts answers to questions the profession itself has not settled. Abstention is never scored as error; a system that declines the undecidable is behaving correctly, and the scorecard is built so that this shows up as a virtue rather than a penalty.

**What the headline honestly measures.** Accuracy on contested items is consensus tracking: agreement with the thin majority of a divided expert panel, and the scorecard says so openly rather than dressing it up as correctness. The decomposition that keeps this honest is rater affinity, reported alongside every scorecard: on contested items only, agreement with each named expert separately, which reveals whose rulebook the model has internalised. No single-gold benchmark can produce this number, because a single-gold benchmark has already collapsed the panel into one voice. On the Shariah pilot the affinity table independently rediscovers the panel's known structure: both baselines agree with MSCI on barely 12 to 14% of contested items, because MSCI is the lone standard screening on total assets rather than market capitalisation, and the informed baseline leans DJIM (96%) whilst the generalist leans AAOIFI (95%). On the sovereign pilot both baselines lean S&P (75%) and agree with Moody's least (38%). Which expert a model secretly thinks like is a fact a deployer currently has no way to know.

**Falsifiability.** If, on a given task, models show no reliability gap and behave sensibly on splits, Quorum will say so, and the conclusion would be that single-gold evaluation was adequate for that task. The benchmark is designed so it can lose. One deliberate corollary: Quorum is a benchmark its designers want overfitted. A model trained to the test here has been trained on the structure of institutional disagreement, which is precisely the capability the benchmark exists to demand; Goodhart's law is, for once, pointing the right way.

## The pilot, run on real data

The first task family is Shariah equity screening, chosen because it offers something rare: four codified, independent expert rulebooks (AAOIFI, MSCI, Dow Jones Islamic, S&P Shariah) rendering verdicts on the same firm-quarters, computable point-in-time from primary filings with no annotation budget. On a panel of 23,248 firm-quarters across 426 US firms (2010 to 2026), the four standards agree unanimously on only 61.0% of items; alpha across them is 0.588, beneath the conventional 0.667 reliability floor. The ground truth is genuinely contested, and Quorum publishes that fact as the first line of the scorecard.

Two out-of-fold baselines were then scored, differing only in what they are permitted to know. The informed baseline sees the four leverage ratios the rulebooks actually use; the generalist sees only conventional leverage and size, which is the honest analogue of a capable financial model that has not internalised the specific standards.

| Baseline | Overall | Unanimous | Contested (headline) | Reliability gap | Confident commits on splits |
|---|---|---|---|---|---|
| Informed | 99.5% | 100.0% | 98.5% | +1.5% | 41.6% |
| Generalist | 96.9% | 99.6% | 92.2% | +7.4% | 46.5% |
| Generalist + abstention | 98.5% | 99.8% | 96.2% | +3.6% | 46.5%, abstains 29.2% |

The instrument discriminates in the way the thesis requires. Domain knowledge shows up as a five-fold reduction in the reliability gap, not as a point of overall accuracy. Adding an abstention band halves the generalist's gap and produces refusals concentrated on the undecidable items. And both baselines commit confidently on roughly four in ten split items, which is a behaviour no existing benchmark can even see, and which is exactly what a compliance officer needs to know before delegating a contested call to a model.

## The second pilot: split sovereign ratings, fully public data

To show the design is a framework rather than a Shariah artifact, a second task family was built entirely from public data: sovereign investment-grade classification. S&P, Moody's and Fitch publish ratings on the same sovereigns; the quorum is the three agencies, the binary question is whether the sovereign is investment grade, and the model inputs are public World Bank fundamentals (GDP per capita, inflation, current account). The panel covers 124 rated sovereigns, 102 of them rated by all three agencies.

The disagreement lands precisely where the split-ratings literature says it should. Alpha across the agencies is 0.909 and 6.5% of sovereigns are contested, and every contested case sits on the BBB-/BB+ boundary: Azerbaijan, Colombia, Hungary, Morocco, Panama, Paraguay, Serbia, and a two-agency even split on Trinidad and Tobago. This is the boundary at which index inclusion, collateral eligibility and mandate compliance actually turn.

| Baseline | Overall | Unanimous | Contested (headline) | Reliability gap |
|---|---|---|---|---|
| Informed (macro fundamentals) | 87.0% | 87.9% | 71.4% | +16.5% |
| Generalist (GDP per capita only) | 84.6% | 85.3% | 71.4% | +13.9% |
| Generalist + abstention | 89.5% | 90.0% | 80.0% | +10.0%, abstains 28.6% of contested |

The pattern replicates: accuracy on the contested stratum runs 14 to 17 points below the unanimous stratum, the abstaining variant narrows the gap by refusing exactly where the agencies themselves disagree, and a single-gold benchmark built from any one agency's ratings would have silently marked a third of the contested calls as model errors that are in fact expert disputes. Two task families, one from private research data and one from fully public sources, now show the same structure, which is the claim the benchmark exists to make measurable.

## The parliament observed through time

A snapshot of 124 sovereigns yields eight contested cases, which is an anecdote. The history panel fixes this by observing the same three agencies through time: their dated rating actions for 27 deliberately selected boundary sovereigns (the ones that crossed or approached the investment-grade line), forward-filled to a quarterly panel from 1995. That yields 2,697 country-quarter items, of which 522 are contested, a sixty-five-fold increase in the stratum that matters, from public data alone. The selection is intentional and stated: this panel measures behaviour at the boundary, not the boundary's base rate among all sovereigns.

The finding is the kind a snapshot cannot see. When a country crosses the investment-grade boundary, the agencies disagree about which side it is on for a median of eight quarters -- two full years per crossing, mean eleven quarters, across 47 disagreement spells. The longest spells are not exotic: Morocco spent roughly thirteen years with the agencies split on its investment-grade status, South Africa seven and a half, Trinidad seven, Uruguay six. Alpha across the agencies on this boundary panel is 0.695, hovering at the conventional reliability floor, against 0.909 for the reassuring cross-section; and agreement is not improving -- the contested share among boundary sovereigns in 2026 (29.6%) is at the top of its thirty-year range. Every quarter inside a spell is a live question that an index rule, a collateral schedule and an investment mandate resolve differently depending on which agency the document happens to name. A financial model deployed into any of those workflows is taking sides in these spells whether its operators know it or not; Quorum makes the taking of sides measurable, per agency, per quarter.

## Where the task families come from

The design generalises to any financial question on which independent expert institutions publish verdicts about the same objects, and the craft is in choosing families where the expert labels can be computed from primary sources rather than bought from annotators: the Shariah standards are codified rulebooks applied to filings, and the agencies publish their sovereign actions, so both existing panels cost nothing to label and re-verify. That cost asymmetry, not the scorecard, is the hard part to copy. The roadmap, in order of tractability:

1. **Shariah screening** (built): four standards per firm-quarter, computable from filings.
2. **Split credit ratings** (built, sovereign tier): Moody's, S&P and Fitch on the same sovereigns, fully public data; the corporate and issue-level tier extends the same adapter to commercial ratings feeds when a data partner brings them.
3. **ESG ratings**: the "aggregate confusion" result puts inter-rater correlation among major ESG raters near 0.5, which makes ESG classification the single most contested labelled domain in finance. Every major ESG feed is proprietary, so this family is the natural contribution for an institutional partner; the adapter interface is the same.
4. **Principle-based compliance**: financial promotions against FCA principles, seeded from the published scenario sets and re-labelled by multiple compliance professionals; the four-axis study above shows this is where judges are weakest.
5. **Analyst calls**: buy/hold/sell on the same ticker-date across brokers, where the quorum is large and the splits are routine.

Each family reuses the same schema, scorecard and harness; only the adapter changes. That is deliberate, because the ambition is a standard, not a dataset.

## Governance and the institutional path

A benchmark becomes a standard through governance, not through a repository. The intended path is a working group under MLCommons, where AILuminate has established the pattern of consortium-governed AI benchmarks with versioned releases and a published grading methodology, and where finance is conspicuously absent from the vertical coverage. The author is an MLCommons member with merged contributions to its modelbench evaluation infrastructure, and the proposal to be put to the working group is precisely the scorecard defined here: reliability-of-ground-truth reported first, accuracy conditioned on agreement, and split-item behaviour as a first-class metric.

Two design commitments are made now to make later governance possible. Versioned, hash-pinned item sets, so that a score is always a score against a stated release. And a strict separation between the harness, which is open source from day one, and any held-out evaluation items, which the working group may keep private to resist overfitting.

## Status and roadmap

One audience deserves naming, because it is not the audience benchmarks usually court. Model-risk validation functions -- the teams executing SR 11-7 in US banking and the conformity assessments the EU AI Act requires for credit-relevant systems -- are obliged to evidence how a model behaves under uncertain ground truth, and no instrument currently exists that measures it. The scorecard's vocabulary (label reliability, behaviour by agreement stratum, abstention, calibration) maps directly onto that obligation, and the reliability gap has a plain operational reading: it locates the boundary of safe delegation, the share of a workflow whose contested calls should still route to a human. Positioned there, Quorum does not compete with capability suites at all.

Built and verified as of this document: the schema (single-gold refused in code), the scorecard and report, rater affinity, the reliability machinery (Krippendorff cross-checked against the reference implementation), the Sharia, sovereign-ratings and sovereign-history adapters, four baselines, and a test suite; both pilots and the crossings analysis run end to end on real data in under a minute. Next: the ESG and corporate split-ratings adapters, an LLM harness so frontier models can be scored alongside the statistical baselines, and a short methods paper. The methods paper's companion evidence already exists, in the harm-scorer result and in the Shariah reliability paper, both of which instantiate the same claim in different domains: headline accuracy against a single gold label systematically overstates what a model knows about the cases that matter.

The one-sentence version, for whoever asks what this is: **Quorum measures whether a financial AI knows the difference between a settled question and a live one, and no other benchmark measures that.**

---

## Sources

- FinBen: https://arxiv.org/abs/2402.12659 and NeurIPS 2024 datasets track
- PIXIU: https://github.com/the-finai/pixiu
- FINESSE-Bench: https://arxiv.org/abs/2605.15482
- FinTradeBench: https://arxiv.org/abs/2603.19225
- Benchmarks Are Not Validation: https://arxiv.org/abs/2607.28840
- Four-axis LLM-judge study (FCA principles): https://arxiv.org/abs/2608.14329
- MLCommons AILuminate: https://mlcommons.org/ailuminate/ and https://arxiv.org/abs/2503.05731
- Companion evidence: Rai (2026), price-driven churn and inter-standard reliability working papers; PyRIT ViolenceClassifierScorer agreement-split result (microsoft/PyRIT#2628)
