# Names as scaffolding: frozen 30-inscription pilot

Written before scoring on 2026-09-24. CPU only. This changes the task from word
class prediction to whole-inscription relationship transfer. It is an exploratory
recovery test of supplied scholarship, not new Etruscan meanings.

## Question and outcome

Can recognised names, their grammar, and repeated anonymous word stems recover
who is related to whom, or who gives something to whom, when the relevant word
meanings and their sentence translations are hidden?

There are five English-side relations:

- CHILD_OF(child, parent), directed; learned from son and mother descriptions.
- SPOUSE_OF(person, person), symmetric.
- OWNED_BY(OBJECT, person), the coarse “of/associated with” relation.
- TRANSFER(donor, OBJECT, recipient), pooling giving and dedication.
- MADE(maker, OBJECT).

UNSPECIFIED is an explicitly missing participant. No inferred person may fill it.
Recovering CHILD_OF for a female name is only evidence for the relationship. It
does not by itself prove that a particular Etruscan word lexically means daughter.
Similarly, TRANSFER does not distinguish religious dedication from ordinary giving.

## Data and hidden meanings

See [AUDIT.md](AUDIT.md) and [manifest.json](manifest.json). Thirty manually selected,
distinct monuments with 22 training translations and 8 hidden translations:

- Three daughter-family inscriptions containing `sech`.
- Five inscriptions containing `mulu`, `muluvanice`, or `mulvanice`.

Their entire translations and relation graphs are withheld from the predictor.
No training text contains these target stems, and no training translation says
daughter. Only name/deity dictionary POS, case and gender fields enter features;
the predictor receives no dictionary meanings for any other word. Supplied names
and person/deity distinctions are favourable manual anchors, not discoveries.

The known deictics `mi`, `mini`, `ecn`, `cn`, `itun` share a generic DEICTIC marker.
Each other non-name token becomes W plus a SHA-256-derived anonymous identity for
its first three letters. Anonymisation is an implementation information barrier,
not cryptographic secrecy. Exact word identities cannot be looked up in a language
model: all predictors here are deterministic local Python code.

## Models fixed before the run

**Alignment.** Collapse consecutive tokens of the same entity; preserve separated
occurrences of a discontinuous name. Align the complete sequences by dynamic
programming. Gap cost: 1 for an entity, 0.5 otherwise. Entity/nonentity substitution
cost 2. Person/deity mismatch costs 0.5; case mismatch adds 0.45 and gender mismatch
0.15 when both values are known. Matching generic token kinds costs 0; W/deictic
mismatch costs 0.8. Normalise by the larger sequence's total gap cost. Ties prefer
a diagonal step, then deletion, then insertion. Drop inconsistent or non-injective
entity mappings. Distance above 0.55 is ineligible.

**Primary: joint constraint tournament.** For each eligible training alignment,
transfer its role graph through the name correspondence. Reject missing endpoints,
self-child/spouse edges, and self-transfers. Each candidate edge keeps its best
alignment merit `exp(-6 * distance)`. Within each queried anonymous stem, each
distinct abstract inscription template contributes once. For each possible relation,
sum `log(0.02 + best merit for that relation)` across these templates, then softmax
the five totals. These are heuristic scores, not calibrated probabilities.

Select the highest-support relation only if its family support is >=0.60. Within
each inscription, accept at most one edge of that relation, and only if its merit
is >=0.60 of the sum of merits of competing endpoint assignments. Otherwise abstain.
The joint predictor sees all eight untranslated query inscriptions at once. It
does not see any query's correct answer. Correlated queries are explicitly part of
the model; eight inscriptions are not eight independent discoveries.

**Local alignment ablation.** Three nearest eligible training templates vote for
mapped edges, weighted by `1/(0.1+distance)`. Accept an edge with >=0.60 of total
neighbour weight. This does not pool evidence across queried stems.

**No-morphology ablation.** The same local predictor without name case/gender costs.

**Count-only baseline.** The most common complete training graph for the same
number of named entities, binding participants by mention order. It sees no text
sequence, name type, morphology or queried stem. Lexical order breaks ties.

## Evaluation flow

```text
Verify the pre-run hashes and train/test information barrier
Fit templates from the 22 visible English-derived graphs
Predict the eight untranslated queries with all frozen models
Score exact relationship labels and participant bindings
Shuffle visible translation graphs and rerun the joint predictor 999 times
Apply the continuation rule without changing settings
```

Report target-edge recall over the eight selected hidden meanings, precision over
all accepted edges against complete gold graphs, coverage (documents with a call),
complete-graph recall/exact match, and separate daughter/mulu target results.
Missing predictions count as misses. For a daughter target, the right child and
parent are required; a spouse call or the wrong parent does not count. For a
transfer target, donor, recipient and unspecified slots must all be correct.

All source monuments and raw inscriptions differ across train/test. Abstract
templates may recur: transfer between comparable formulas is the stated task.
No claim of generalisation to unseen formula types is made.

## Controls and continuation rule

Perform 999 fixed-seed permutations (Python Random seed 240925), shuffling whole
training translation graphs among inscriptions with the same entity count. This
preserves valid participant IDs and class frequencies. A count stratum with only
one example cannot be permuted (the three-entity ETP 189 case); report this limit.
Compute one-sided `(1 + null target recalls >= real) / 1000` for the single primary
joint model. Do not choose a model by its eventual score.

Also report local-alignment leave-one-monument-out performance on the 22 training
records as a descriptive competence check. It is not used to fit thresholds or
select models. Synthetic positive controls and regression tests run before freeze.

The primary model justifies a larger independent annotation exercise only if all
of the following hold:

1. Target-edge recall >=75% (at least 6/8), with each hidden family >=60%.
2. Accepted-edge precision >=80% and document coverage >=50%.
3. Target recall exceeds the count-only baseline by at least 25 percentage points.
4. Translation-permutation p <=0.05.

These are screening requirements for this small selected pilot. They do not turn
two hidden lexical families into a representative test or a discovery claim.
Even a pass requires a separately sourced, expert-reviewed validation.

```text
If every primary requirement passes:
    Recommend a larger independently annotated validation
Otherwise:
    Report the specific failure and retain the audited benchmark
In either case:
    Do not publish new unknown-word meanings or tune on these eight answers
```

## Reproduction

```sh
.venv/bin/python -m unittest discover -s tests -p test_etruscan_scaffolding.py -v
.venv/bin/python -m experiments.etruscan_scaffolding build
.venv/bin/python -m experiments.etruscan_scaffolding freeze
.venv/bin/python -m experiments.etruscan_scaffolding run
```

Build, freeze and run refuse to overwrite their outputs. Freeze records code,
tests, protocol, audit, manifest, public inputs and pinned source hashes before
scoring. It is a local timestamp/hash record, not external preregistration. Prior
Etruscan experiments remain unchanged. The initial audit files document the
pre-score addition of anonymous stem identities; no real score prompted it.
