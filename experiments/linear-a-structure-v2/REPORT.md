# Entry-ending experiment: known-answer gate failed

Completed 2026-09-24. [Protocol](PROTOCOL.md), code and [freeze](freeze.json) committed at
`6504470` before observing any scores. [Results](results.json). No Linear A model was fitted
or scored after the known-answer gate failed.

## What was done

- Parsed Linear A into sign IDs, retaining unknown readings and rejecting damaged runs.
  Coverage: 857 tokens on 336 documents; 511 training types and 85 unseen test types.
  Ten test types contain signs without known sound values.
- Grouped tablet faces; withheld documents by a fixed hash. Removed every test word type seen
  in training and discarded types with inconsistent numeric-entry roles within a partition.
- Compared a bag-of-signs/length baseline with the same model plus final sign and final two
  signs. Both use fixed naive Bayes settings and equal class priors.
- Tested Linear B on 20 samples with exactly the target's training/test class counts. Tested
  the same samples after shuffling signs within each word, preserving lengths and sign counts.
- Recorded a full-size Linear B control and 199 within-site/length training-label permutations.

## Why

The profile audit cannot answer whether endings predict a word's role on an unseen inscription.
This bounded experiment tests that relationship without guessing sound values or memorising
the complete test words.

## Results

Balanced accuracy (BA) averages entry and non-entry recall; a constant prediction scores 50%.
The fixed pass rule requires BA >= 60% and a gain of >= 5 percentage points over the baseline.

| Check | Baseline BA | Endings BA | Gain | Samples passing |
|---|---:|---:|---:|---:|
| Full Linear B: 3,193 train / 544 test types | 55.39% | 52.84% | −2.54 points | failed |
| Target-sized Linear B: 511 train / 85 test, mean of 20 | 54.66% | 55.19% | +0.53 points | **4/20**, required 18 |
| Same samples, within-word sign shuffle | 54.66% | 51.68% | −2.98 points | **0/20**, maximum 1 |
| Full Linear B, training-label shuffle | — | — | observed −2.54 points | diagnostic p = 0.655 |
| Linear A | — | — | — | **not run: gate failed** |

1. **No reliable transfer at the target's size.** Ending-model BA ranges from 46.62% to 70.51%
   across the 20 overlapping samples. Four satisfy the joint accuracy/gain rule; 18 were required.
   The average gain is only 0.53 points, and full-size Linear B is worse with endings added.
2. **The negative control behaves as intended.** Shuffling within words leaves the baseline
   exactly unchanged and removes the candidate's average advantage. None of its samples passes
   the joint rule. This does not compensate for the failed positive control.
3. **Unknown signs are now available for later structural methods.** The parser represents them
   by sign identity, rather than dropping the whole word. That improves the representation's
   coverage, but is not a demonstrated predictive gain: Linear A was deliberately not scored.

## What this does and does not establish

The fixed final-sign/final-pair naive Bayes model is retired for this task. This is a negative
result for one model, parse and role definition, not proof that Linear A lacks grammar or that
context cannot help. Site, genre, partial formula dependence and automatic role labels remain
limitations. A positive result would have established predictable numeric-entry position,
not a language identification or translation.

The artificial order-only unit test recovers its planted rule exactly, but does not substitute
for the real Linear B control. The 20 samples overlap and the permutation p-value depends on
the declared within-site/length exchangeability assumption; neither is independent replication.

## Deviation and reproduction

The [first driver](../linear-a-structure/REPORT.md), frozen at `fd3f6b2`, aborted while assembling
its first score because its reporting function lacked `import numpy as np`. It printed/saved
no metric. This v2 adds that import with unchanged model, split and thresholds; the original
frozen file remains intact. A driver-level test covers this reporting path.

Use a worktree at `6504470` with the pinned sources attached:

```sh
.venv/bin/python -m experiments.linear_a_structure_v2 verify
.venv/bin/python -m experiments.linear_a_structure_v2 run
```

The output refuses overwrite. Source origins/licences remain in the original Linear A and
DĀMOS manifests; the new freeze hashes all consumed files. Derived results: CC BY-NC-SA 4.0.
