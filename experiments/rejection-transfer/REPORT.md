# Rejection and frozen-key transfer: inconclusive

## What was done

- Tested the frozen five-language decoder on fresh fit/transfer pairs.
- Sealed learned keys before loading transfer ciphertext; no transfer key refit.
- Used the declared early-stop rule. No Voynich text was scored.

## Why

A decoder that always produces text needs a tested rejection rule. A mapping must also work on text it was not fitted to.

## Presentation correction

The raw summary's absent-language `correct: 1` counts a null decision despite a compute
cap. It is **inconclusive**, not a validated rejection; neither row establishes a pass
or failure of the statistical decision rule. This count is corrected in the resource
repair round's evaluator. English itself won both rankings (excess 0.407 / 0.382), with
6.21% / 6.65% letter error and 98.62% transfer token coverage. Three wrong-prior fits
(French, German, Italian) hit refinement's 300-second cap. No shuffle or copy input ran.

## Results

| Input | Evaluated | Accepted | Correct decision |
|---|---:|---:|---:|
| absent | 1 | 0 | 1 |
| copy | 0 | 0 | 0 |
| positive | 1 | 0 | 0 |
| shuffle | 0 | 0 | 0 |

**Stop:** `inconclusive_compute_cap`. Planned: 10 independent passage/key blocks. Evaluated positives: 1. Fit-worker time: 0.72 hours.

| Block | Input | Source language | Fit winner / excess | Transfer winner / excess | Accepted | Reasons |
|---|---|---|---|---|---|---|
| 1 | positive | english | english / 0.407 | english / 0.382 | none | compute_cap |
| 1 | absent | english | old_french / 1.563 | old_french / 2.380 | none | compute_cap, fit_excess, transfer_excess |

## Positive recovery diagnostics

| Block | True language | Fit CER | Transfer CER | Transfer token coverage |
|---|---|---:|---:|---:|
| 1 | english | 6.21% | 6.65% | 98.62% |

## Limits and decision

This is a bounded screen with thresholds chosen from the released five-case language-ID pilot. It is not a fresh estimate of a 90% sensitivity / 5% false-positive operating point. Paired negatives share their positive source and are not independent replicates. Early-stop proportions are descriptive; unrun cases are not counted as rejections.

The old language-ID decoder and its default Naibbe spacing were tested. The newer long-passage Italian decoder, heavier pairing, other cipher families, other languages, and the published Timm generator were not tested.

**No manuscript run is licensed.** A failed screen parks this version; an inconclusive screen needs a declared resource/protocol repair; a pass would require a larger independent control study.

## Records and reproduction

[Protocol](PROTOCOL.md) · [Freeze](freeze.json) · [Calibration](calibration.json) · [Sources](sources.json) · [Results](results.json) · [Released records](evaluated-records.tar.gz)

```sh
.venv/bin/python -m experiments.rejection_transfer verify
.venv/bin/python -m unittest tests.test_rejection_transfer -v
```

Driver subcommands: `sources`, `freeze`, commit, `prepare`, `run`, `report`. Creation and grading refuse overwrites. `run` resumes verified per-input checkpoints. Archive contains only graded answers; unrun answers remain evaluator-only.

Final archive replay and source audit passed; all 169 tests passed. See
[final verification](final-verification.json), which completes the pending checks
in the earlier pre-run `verification.json`. The capped outcomes remain inconclusive.
