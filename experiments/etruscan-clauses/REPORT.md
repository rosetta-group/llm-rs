# Clause Coverage: failed; stop local tuning

This experiment learns predicate anchors and local participant roles from
46 translated Etruscan inscriptions, then evaluates 20 different inscriptions.
Clause coverage does not improve complex interpretation or preserve precision;
the agreed stopping rule now applies.

**Anchor:** a token occurrence assigned a predicate from training associations.
**Edge:** a relationship with specified roles, such as CHILD_OF(child,parent).
**Exact graph:** every reference edge for an inscription, with no extra edges.
**Confound amplification:** treating a correlated token as semantic evidence,
then increasing confidence by requiring that mistaken evidence to be explained.

## What was done

- Added training-only token/predicate associations, local role-position templates
  and penalties for uncovered spans. No manual predicate spans or query
  translations enter the model.
- Used the previous 16 tests as development evidence and froze a 20-inscription,
  36-edge evaluation, including eight mixed-predicate inscriptions.
- Ran three ablations, an unchanged previous-model comparison, a name-count
  baseline and 99 complete training-translation shuffles.
- Passed 36 unit tests, including a synthetic case where coverage restores an
  omitted construction predicate. Verified 67 hashes across four prior frozen
  experiments and 24 hashes for this experiment.
- Preserved all frozen models and results. Post-run diagnostics are separate.

## Why it was done

The previous model could account for every person in Cr 5.3 through parentage
while omitting construction. This experiment tested whether learned textual
evidence could require that second statement without manual test annotations.

## Frozen sequence

```text
Train from the 46 already exposed monuments
Freeze code, data, comparisons, thresholds, and success criteria
Predict the 20 different inscriptions without their translations
Score all relationships and complete graphs
Refit and rerun with 99 shuffled training translations
Apply the gate unchanged
Stop local tuning after failure
```

The [protocol](PROTOCOL.md), [audit](AUDIT.md), [freeze](freeze.json),
[results](results.json) and [diagnostics](diagnostics.json) preserve the evidence.

## Results

| Method | Correct / called edges | Precision | Recall of 36 edges | Edge F1 | Exact graphs | Exact mixed graphs |
|---|---:|---:|---:|---:|---:|---:|
| Clause coverage, primary | 4 / 8 | 50.0% | 11.1% | 18.2% | 4 / 20 | 0 / 8 |
| No coverage penalty | 4 / 7 | 57.1% | 11.1% | 18.6% | 4 / 20 | 0 / 8 |
| No local role offsets | 6 / 10 | 60.0% | 16.7% | 26.1% | 5 / 20 | 0 / 8 |
| Extended graph search, no spans | 4 / 7 | 57.1% | 11.1% | 18.6% | 4 / 20 | 0 / 8 |
| Previous graph model | 4 / 7 | 57.1% | 11.1% | 18.6% | 4 / 20 | 0 / 8 |
| Name-count majority | 19 / 26 | 73.1% | 52.8% | 61.3% | 6 / 20 | 0 / 8 |

The primary makes calls on eight documents (40% coverage). Its four correct
graphs are ETP 120 (making), Cm 2.32, Cm 2.65 and ETP 331 (ownership).
It recovers none of the 20 mixed-case reference edges or seven kinship-cohort
edges. The incorrect accepted graphs are Cr 3.17, Ta 1.168, Cr 3.18 and Vc 6.6.

The earlier model's 10/16 result does not carry over: with the same 46 training
examples as the primary, it scores 4/20 here. These are different, deliberately
harder cases, so this is evidence of limited generalisation rather than a
same-test accuracy comparison between the two historical runs.

## Gate and negative control

| Required criterion | Observed | Outcome |
|---|---:|---|
| Precision >=80% | 50.0% | Fail |
| Precision no lower than previous model | 50.0% vs 57.1% | Fail |
| Overall recall >=60% | 11.1% | Fail |
| Document coverage >=50% | 40.0% | Fail |
| Mixed-case recall >=60% | 0.0% | Fail |
| Mixed exact graphs >=4/8 | 0/8 | Fail |
| More exact mixed graphs than previous model | 0 vs 0 | Fail |
| F1 improvement over previous >=10 points | -0.4 points | Fail |
| F1 improvement over extended search >=5 points | -0.4 points | Fail |
| Translation-shuffle p <=0.05 | 0.01 | Pass |

The null F1 mean is 3.5% and maximum 14.6%; the observed 18.2% exceeds all 99
shuffles. That does not rescue the failed accuracy gate: there is some real
association in the corpus, but the tested use of it is unreliable. The p-value
is conditional on this selected sample and these shuffles, not historical proof.

## Mechanisms found after the run

1. **Wrong anchors add confidence to a wrong relationship.** Ta 1.168's
   reference says Ramtha Semni was wife of Larth Spitus. Its `puia` stem correctly
   supports SPOUSE_OF, but `lupu`, `avils` and a numeral stem all acquire CHILD_OF
   associations from training co-occurrence. The new model confidently calls
   CHILD_OF(e0,e1), with margin 0.169. The previous model abstains on this text.
   This is Confound amplification: covering noisy evidence worsens the answer.

2. **A predicate does not identify all its participants.** Ru 5.1 and Cr 5.2
   produce a MADE anchor from `cerichunce`, learned from the single training
   example Cr 5.3. The highest-ranked graphs make all three people makers,
   including parents. The margin rule correctly abstains, but recognising a
   construction-associated stem does not separate builders from relatives.
   Clipping role distances and pooling local templates leaves that distinction
   unresolved; this experiment does not identify which alternative role model
   would solve it.

3. **Coverage cannot recover evidence that was never grounded.** Nine of 20
   test inscriptions have no selected anchors, including five of eight mixed
   cases. In Cr 3.17, the model turns a parent into a gift recipient. In ownership
   plus parentage texts such as Vs 1.28, it still prefers a graph omitting a
   statement. Source references often express relations through names, endings
   or implicit constructions rather than an independently learned content word.

4. **Coarse stems introduce further conflicts.** Ve 3.2 has a `mene` token
   whose three-letter ID matches the training `men` maker example. It therefore
   gets both MADE and TRANSFER anchors, although its supplied graph is a gift
   from two donors. The primary ranks the correct graph first but abstains at
   margin 0.0875. Without local role offsets, this case is accepted correctly.
   Changing the representation after seeing this example would be another
   development iteration, not confirmation of the frozen hypothesis.

All 36 reference edges are present in the generated candidate inventories.
This does not claim every reference graph survives edge pruning, but it rules
out missing edge generation as the universal explanation. The four-edge
exhaustive-search control matches the previous model's accepted predictions;
simply increasing search capacity did not improve these outcomes.

Cr 3.18 deserves separate source review: the supplied translation explicitly
says “given by” a donor, while the name morphology encourages the model to call
a recipient. The evaluation keeps the reference unchanged. Resolving that
potential scholarly issue would not fix the other three wrong calls or zero
mixed-graph recoveries.

## Decision and data handoff

**Stop tuning this dataset.** The stronger statement “all computational Etruscan
approaches fail” is not supported. The narrower result is that these coarse
name/stem representations and weak graph supervision do not support reliable
complex interpretation, even with favourable manual name anchors.

The next useful work is independently checked data, with explicit predicate
spans, role bindings, implicit-relation flags, background-token annotations,
restoration uncertainty and source citations. Concrete audit priorities are
Ru 5.1/Cr 5.2 for makers versus parents, Ta 1.168 for wife versus age/death
clauses, Vs 1.28/Vt 1.58 for implicit ownership and parentage, and Cr 3.18 for
donor/recipient disagreement. The same 66 exposed monuments may support
development, but a later success claim needs separately reserved evaluation.

No independent annotations were obtained in this task. No new Etruscan meanings
or unknown-word predictions are produced, and no further model round is launched.

## Verification and reproduction

The 36 tests verify implementation behaviours, not linguistic correctness. The
synthetic omitted-predicate test succeeds while the real mixed cases fail; both
outcomes are retained. `diagnostics.json` records result and diagnostic-code hashes,
and recomputes all six saved summary tables from the individual predictions.

For reproduction in a clean copy preserving repository paths, retain the two
prior experiment directories and pinned sources. Copy the unchanged audit and
protocol, omit only this experiment's generated JSON outputs, and run:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_etruscan*.py' -v
.venv/bin/python -m experiments.etruscan_clauses build
.venv/bin/python -m experiments.etruscan_clauses develop
.venv/bin/python -m experiments.etruscan_clauses freeze
.venv/bin/python -m experiments.etruscan_clauses run
.venv/bin/python -m experiments.etruscan_clause_diagnostics
```

Outputs refuse overwrite. Freeze timestamps will differ; compare predictions,
metrics and null summaries rather than the timestamp-dependent result hash.
