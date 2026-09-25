# Etruscan round three: repair and abstain

Written 2026-09-24 before evaluating any round-three classifier on real labels.
CPU only. This is an internally held-out follow-up on previously studied source
data, **not** a new independent confirmation or an Etruscan translation.

## Question

After repairing dictionary handling and excluding damaged/ambiguous records,
does a simple classifier make reliable, nontrivial predictions for a useful share
of held-out word families? Do word endings improve on context alone?

## Data and audit

See [AUDIT.md](AUDIT.md) and machine-readable `audit.json`. ETP only, 260 texts,
1,154 tokens. Labels come from merged certain ETP rows; ambiguous semantic classes
and uncertain/unknown glosses are excluded. No prediction-informed label changes.
Original files and earlier frozen results are untouched.

The 49 round-two fresh words are development evidence only. Their connected
families are removed from all labelled seeds, calibration, and scoring. Roman
numerals are fixed NUM anchors, never evaluation items. The 427 remaining word
forms belong to 303 conservative spelling/gloss families; these groups, not
individual spellings, are the unit of splitting and uncertainty resampling.
No new sources or labels are acquired for this bounded experiment.

## Frozen models

- **Context:** multinomial naive Bayes with add-one smoothing and empirical class
  priors. Features are distributions over the left/right known neighbour classes,
  UNK/boundary indicators, and word position. Each block is averaged over a word's
  occurrences. No iterations or guessed neighbour labels.
- **Context + endings:** the identical classifier with final 1, 2, and 3 letters
  as additional features. An ending must be shorter than the whole word.
- **M2 bridge:** round two's unchanged neighbour-class predictor, run descriptively
  on the repaired data and the same outer splits. It has no abstention and is not
  eligible for the continuation gate.
- **Majority baseline:** the most common training word class, evaluated on exactly
  the subset each abstaining method accepts.

Both models hide same-family neighbour labels even when constructing training
features, so training words cannot use a privilege denied to held-out families.
All texts, including unlabelled held-out occurrences, are visible: this is a
transductive vocabulary test, not a test on unseen inscriptions.

## Splits and abstention

```text
Partition families into five outer folds by fixed hash and type count
For each outer fold:
    Reserve its labels for evaluation
    Make three family-disjoint inner folds from the remaining labels
    Choose each method's acceptance threshold using only inner predictions
    Fit on all outer-training labels and score the reserved fold once
Pool the five disjoint outer predictions
Apply the frozen gate and family bootstrap
```

Exact splitting is in the runner; no label stratification or split searching.
Threshold candidates: 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99. The softmax score
is a ranking score, not a claimed calibrated probability. Choose the candidate
with maximum inner coverage subject to at least 15 accepted types from 5 families,
precision >=80%, and a Wilson 95% lower bound >=65%. Lower threshold breaks ties.
The Wilson bound is only a calibration heuristic; correlated forms are handled
by the family bootstrap for final uncertainty. If no threshold qualifies, abstain
on every item in that outer fold.

Regardless of score, an accepted context prediction requires at least one directly
labelled neighbour. The endings method can alternatively use an ending of length
2 or 3 observed in at least three distinct training families. A singleton with no
usable context or ending support receives no accepted prediction.

## Outcomes and continuation gate

Report full balanced accuracy (mean recall across classes), full accuracy,
accepted precision, coverage (accepted/all test types), accepted class breakdown,
and non-NAME calls. A method passes only if all apply to pooled outer predictions:

1. Accepted precision >=80%, coverage >=20%, and at least 20 accepted families.
2. Lower endpoint of the 95% family-bootstrap precision interval >=70%.
3. Precision exceeds the majority baseline on the **same calls** by >=5 points.
4. At least 10 non-NAME calls across at least 3 families, with precision >=70%.
5. Full-classifier balanced accuracy beats the seed-label null at p <=0.05.

Use 1,000 paired family-bootstrap replicates, fixed RNG seed 240924. Resample
families with replacement, retaining all their types and frozen outer predictions.
Zero-call replicates count as zero precision for a conservative lower bound.
This interval reflects test-family variation, not full retraining uncertainty.

For the seed-label null, perform 99 repetitions of all five outer fits with seed
labels shuffled within each fold (including Roman anchors). Gold remains fixed.
Report `(1 + null balanced accuracies >= real) / 100`. This is a full-classifier
negative-control test; it does **not** establish a p-value for the abstention
policy, and does not account for previous experimentation on these sources.

Report the paired bootstrap interval for full balanced-accuracy improvement of
endings over context. Call endings an improvement only if the interval's lower
endpoint is above zero. Two methods are evaluated, so p-values are descriptive
screening evidence, not a multiple-testing-corrected discovery claim.

## Interpretation fixed before the run

```text
If a method passes:
    Continue only to a separately sourced and independently reviewed validation
Else:
    Stop this implementation and report which requirements failed
In either case:
    Keep unknown-word meanings unpublished
    Do not tune on outer results or reuse the old 49-word test as confirmation
```

The repair, cleaning, model change, and stricter split change together. Differences
from round two cannot be attributed causally to the dictionary repair alone.
There is no new Latin benchmark in this bounded follow-up. Neither a pass nor a
failure demonstrates what all Etruscan methods could achieve.

## Reproduction

```sh
.venv/bin/python -m unittest discover -s tests -p test_etruscan_repair.py -v
.venv/bin/python -m experiments.etruscan_repair audit
.venv/bin/python -m experiments.etruscan_repair freeze
.venv/bin/python -m experiments.etruscan_repair run
```

Audit, freeze, and run refuse to overwrite their outputs. The run verifies frozen
code, tests, protocol, audit, and source hashes before scoring. `freeze.json` is a
local pre-run hash record, not an externally timestamped preregistration. No
post-result code or protocol changes without an explicitly versioned new run.
