# The tablet as a program: integer-account feasibility pilot

Date: 2026-09-24. CPU only. Scope: a bounded first implementation, not the whole proposed
accounting programme. Published fraction values, mixed measurement conversions, inter-tablet
joins, nested subtotals and allocation ratios are not implemented in this version.

## Question and exposure

Can opaque recurring words select a fixed arithmetic operation that predicts an unobserved
quantity vector on another physical tablet? The first required result is a readable Linear B
control. Linear A cannot be fitted or scored before that control passes.

Mainland Linear B transcriptions were inspected during parser development, including records
containing to-so/to-sa. Earlier work also exposed parts of both corpora. This is procedural
blinding of word identities to the learner, not an independently blind or previously unseen
corpus. A public discussion of KN Fp 1 was read while scoping the work; no claim of untouched
Knossos data is made. The mathematical programme remains a new question.

**Coverage was inspected before this freeze.** The first parser yielded six eligible Knossos
objects. A development correction accepted intact one-sign words such as `o` and rejected
unquantified commodity sequences; the final parser yields seven. Linear A has 33 eligible
objects. Neither arithmetic model fits, candidate successes nor permutation scores were
computed during this inspection. The preflight intentionally records that the proposed
control is presently not evaluable. The minimum control requirement of 10 was in the driver
before the initial coverage result; the preflight was then added to prevent a pointless run.
This freeze is not presented as an unseen-data coverage prediction.

```text
Parse quantities and record every exclusion reason
Check that a control is possible at the target object count
If coverage is insufficient, record not_evaluable and stop before fitting
Otherwise learn rules across whole-object folds in opaque Linear B
Grade numeric predictions and reference-marker recovery
Refit on quantity-shuffled controls
Require whole-object size-matched controls to pass
Only then fit and test opaque Linear A
```

## Inputs and parsing

Use the already pinned 5,932 DĀMOS items and Navarre-AI Linear A corpus. Hash every raw file,
code, tests, opaque extracted inputs, evaluator vocabulary, coverage and control review records.
Primary source licences: CC BY-NC-SA 4.0 for DĀMOS and SigLA-derived fields; original source
manifests are `experiments/linear-a-context/sources.json` and `experiments/linear-a/sources.json`.

- Control: all KN records; development: all other DĀMOS sites, never pooled into the control
  to repair low coverage. Do not select accounts by whether their arithmetic balances.
- A row contains literal word/sign IDs and a dictionary of commodity IDs to nonnegative
  integer counts. Different commodities are separate dimensions, never added as scalar units.
  Both words and commodities are hashed to opaque IDs before learning. Identical strings
  retain identity; the model gets no phonetics, Greek dictionary or commodity meaning.
- B: remove only the line label; split tokens on whitespace/comma/slash. Accept whole literal
  syllabic strings, commodity labels and integers. Editorial marks, unknown formatting, repeated
  quantity keys or an unquantified commodity reject the **whole line**. This is deliberately
  restrictive: even an uncertainty mark in a personal name makes that line unusable here.
  Unit markers T/S/V/Z/M/N/P/Q/L/ZE/MO reject the line rather than being ignored or guessed.
- A: tablets only, with a SigLA sign layer. Reject the entire record if that layer contains an
  uncertain sign or the collation records a conflict. Retain unknown sound values as sign IDs.
  The inherited fixed 80% logogram-role threshold defines commodity boundaries. Sum adjacent
  Aegean numeral characters into one integer. Fraction signs and damage reject the line;
  neither is silently zero. Commodity-free quantities have an explicit UNSPECIFIED dimension;
  it is not equated with an explicit commodity on another row.
- Blanks, text-only headings, rejected lines and record ends bound contiguous blocks. A block
  needs >=3 numeric lines. No crossing damage, inventing joins, arbitrary subset selection,
  interpreting indentation or silently carrying an omitted commodity across rows.
- Each word-bearing row with >=2 rows on either side is an eligible prediction case. Counts
  are independent of arithmetic agreement. Preserve line numbers and every rejection reason.
- A faces share `parent_object`; B records group by site and tablet number. Exact duplicate
  contents within a site are co-grouped. Five folds are assigned by SHA-256 of `ledger-v1:`
  plus object group, modulo 5. Word types may recur across folds; physical objects may not.

## Fixed arithmetic language and learner

**Quantity vector:** separate integer coordinates for commodities; for example `{grain: 2,
oil: 3}` is not the scalar 5. **Program:** one of these three fixed expressions:

1. `sum_before`: sum all preceding rows in the contiguous block (at least two).
2. `sum_after`: sum all following rows (at least two).
3. `balance_before`: first preceding vector minus the sum of the remaining preceding rows;
   at least two preceding rows, no new commodity keys, and no negative output.

The target quantity is not accessed by any prediction expression. The model may see the other
numbers, row words and boundaries on the held-out tablet. No fitted coefficients, arbitrary
row subsets, numerical tolerance or target-dependent selection among rules is permitted.
This grammar cannot yet handle nested totals, unit conversion, or arbitrary allocation ratios.

For each opaque word/program, count eligible training objects and objects whose eligible
occurrences all fit exactly. Retain a rule only with >=3 objects and >=80% exact-object fits.
At test time, apply the retained rules triggered by the row's words. Predict only if all
applicable rules give one identical quantity vector; conflicting predictions abstain.
Abstentions count as misses in the control recall metric. All training and rule selection are
repeated inside each whole-object fold.

## Preflight and conditional gates

First require >=10 eligible control objects, >=10 target objects and at least as many eligible
control objects as target objects. These are necessary, not sufficient, conditions. If any
fails, save `status: not_evaluable` with counts and reasons, and leave control predictions,
permutations and Linear A results null. Do not lower the gate, resample with replacement,
silently include development objects or claim that arithmetic structure failed.

The following conditional stages are frozen for clarity but will not run on this release's
insufficient coverage:

- Known-marker check: only after opaque B predictions are saved, open the evaluator dictionary.
  The predeclared reference sum markers are `to-so`, `to-sa`. Require >=10 eligible objects
  containing them, exact prediction on >=80% of their eligible rows, numeric prediction
  precision >=90%, and >=90% of words triggering predictions in the reference marker set.
  These words can mean a specified amount rather than a computed sum in particular contexts;
  their occurrence is **not a source-adjudicated gold label for a total equation**. This
  deliberately stringent reference-marker check would need independently annotated cases
  before any stronger claim about general semantic recovery. Other valid operations can fail it.
- Fixed baselines: always sum before and always sum after, without word identities. Report
  their exact known-marker counts. No claim of added numeric predictive value over them is
  licensed without improvement; word-role recognition is a distinct question.
- Negative control: 999 draws, seed 24092411. Shuffle complete quantity vectors within every
  eligible block, keeping words, length, boundaries and vector inventory fixed; refit the full
  learner and folds each time. Statistic: objects with >=1 prediction and all predictions exact.
  Require plus-one p < .05 and false gate passes <=5%. Minimum p = .001 is attainable.
- Size control: 20 whole-object draws of size equal to the A eligible object count, no
  replacement, seeds 24092500..24092519. Require >=3 reference-marker objects per draw and the
  same recall, numeric precision and marker precision cutoffs. At least 18/20 must pass.
- A: only after all gates pass, run the same learner and object folds; no ku-ro hint. Refit
  against 999 whole-vector shuffles with seed 24092412. Report exact objects, predictions and
  p. This global arithmetic-association result alone would not assign word translations.

## Software validation, interpretation and next work

Tests recover an opaque total word on 30 synthetic objects across withheld-object folds,
reject a quantity-shuffled case, demonstrate target-number independence, keep commodity
dimensions separate, reject fractional/damaged lines, retain unknown A signs, prevent repeated
faces supplying independent support, abstain on conflicting rules, and catch impossible
coverage before fitting. These are software checks, not a claim of power on natural tablets.

The original integer ku-ro check remains frozen. For example, HT 13 has fraction signs in its
entries and total, so the older displayed 130=130 compares integer parts only. It is not a
verification of the complete quantities; the new parser marks those lines as unsupported.

The useful next step after this feasibility result is a source-checked benchmark of complete
accounts with explicit row grouping, units, uncertainties and target-function labels. Extend
the parser with cited unit conversions and accepted joins only after that annotation. The
seven current review objects are candidates, not seven verified balance sheets. Keep any
inspected examples in development and obtain separate objects for a later control. New data
or grammar requires a new module/protocol version; never modify this frozen release to pass.

Background for later fractional work: Corazza, Ferrara, Montecchi, Tamburini and Valério,
*The mathematical values of fraction signs in the Linear A script* (2021),
https://doi.org/10.1016/j.jas.2020.105214. Proposed fraction values are not assumed in this pilot.
