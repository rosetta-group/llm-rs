# Three rejection-screen follow-ups: bounded development

Declared 2026-09-25, before executing these comparisons. The user authorized all
three follow-ups. This is development on released records, not fresh confirmation.
The preceding rejection-transfer protocols and their primary results stay unchanged.
No Voynich text, paid computation or new sealed source passages.

## What is being tested

1. **Decision calibration.** Compare the original two-passage acceptance rule with a
   transfer-centered candidate: retain agreement of language winners, both margins
   >= 0.25, transfer excess <= 0.50, coverage >= 95%, and no cap; remove only the fit
   excess ceiling. Record threshold sensitivity from 0.00 to 1.00 in 0.05 increments.
   Use every graded resource-repair case, including each paired absent-language
   decision. The earlier capped attempt remains inconclusive. Do not select an
   operating point or claim a false-positive rate from three source/key blocks.
2. **Stronger copying control.** Generate a new pair for each released resource-repair
   positive. Retain each positive passage's exact token multiset, length, glyph
   inventory and frequencies. Sequentially consume that bag: with probability 0.8
   choose uniformly from positions in the last 50 emitted tokens that still have
   copies remaining; otherwise sample the remaining bag in proportion to its counts.
   If the recent window has no eligible token, use the remaining bag. This changes
   order and encourages local copying without introducing unseen glyph pieces.
   Use seeds 20260925 + 2*block + role (fit=0, transfer=1). Fit every candidate
   language independently on the control fit text; seal its key before transfer.
   This is a custom stress generator, not the published Timm generator.
3. **Cheaper key search.** In a new module, chunk the exhaustive pair-swap batch while
   retaining its order, full candidate set, old score function and earliest-minimum
   tie rule. Benchmark memory and warmed runtime against the legacy batch on
   deterministic one/two-letter keys and representative released inputs. An optional
   direct swap scorer may avoid materializing candidate key matrices. For one-letter
   keys, an incremental scorer may evaluate only n-grams touched by the proposed
   change, plus the homophone cost. It must match every tested legacy candidate
   score within a recorded floating-point tolerance and recheck close minima with
   the original scorer before choosing a move. Two-letter keys retain the chunked
   full scorer. Add explicit sweep/evaluation
   limits; report limits as incomplete search, never convergence. Compare full
   uncapped refinement outputs on small fixtures, including ties and bigrams.

## Selection and resource limits

Implement and test the refiner before fitting stronger controls. Use the fastest
verified backend that bounds candidate memory; record the choice and hash code and
settings before the first control fit. A slower backend is not a speed improvement;
report any memory benefit separately. CPU only, at most five workers and two Numba
threads each. Keep existing priors, joint-EM stages, repair settings and score.

Benchmarks: at most 15 minutes wall time, two threads, warm compilation reported
separately. Test sizes and repetitions must be recorded before timing. Stronger
controls: at most three paired inputs, five priors each; 3,600 seconds per fit,
1,200-second refinement allowance, 50 sweeps and 20 million candidate evaluations
per refinement. Reserve five fit-worker hours before each input under an aggregate
12-hour fit-worker budget, as in the preceding screen. A cap or resource stop is
inconclusive for that input. Do not increase limits after seeing outcomes.

Only the current screen's released source pairs may be used. Fixed keys from the
original positive are not a substitute for independently fitting the controls.
Archive generated controls, saved mappings, transfer outputs, diagnostics and the
code/settings freeze. Source IDs are already consumed development data; no result
here becomes an independent confirmation unit.

## Reporting and decision

```text
Finish and archive the preceding frozen screen
Verify original records without opening ungraded answers
Develop and test chunked refinement against the legacy objective
Benchmark warmed runtime and candidate memory under fixed work
Record the selected backend and freeze control-fit code/settings
Fit and transfer stronger copying controls once
Compare original and transfer-centered decisions on all released development cases
Record errors, cap stops, coverage, timing and limitations
Write a costed plan for fresh confirmation; do not run it as part of development
```

Report all three follow-ups, including failures. An apparent development separation
licenses only a proposed rule, not a manuscript run. Fresh confirmation would need
independent sources/keys, all five languages, multiple negative generators and a
sample/budget plan justified by measured cost. With zero false acceptances, at least
59 independent cases are needed for a one-sided 95% binomial upper bound below 5%;
shared sources and paired controls still limit independence.
