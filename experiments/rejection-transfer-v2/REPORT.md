# Rejection and frozen-key transfer: inconclusive

## What was done

- Tested the frozen five-language decoder on fresh fit/transfer pairs.
- Sealed learned keys before loading transfer ciphertext; no transfer key refit.
- Used the declared early-stop rule. No Voynich text was scored.

## Why

A decoder that always produces text needs a tested rejection rule. A mapping must also work on text it was not fitted to.

## Results

**Conclusion: inconclusive after three of ten planned blocks.** Two genuine ciphers
passed; English was rejected by the fit-score ceiling. Eight negative decisions
were conclusive rejections, and the Latin copy/mutate decision was inconclusive.
These are three shared source/key blocks, not nine independent negative trials.

| Input | Evaluated | Accepted | Correct decision | Inconclusive |
|---|---:|---:|---:|---:|
| absent | 3 | 0 | 3 | 0 |
| copy | 3 | 0 | 2 | 1 |
| positive | 3 | 2 | 2 | 0 |
| shuffle | 3 | 0 | 3 | 0 |

**Stop:** `inconclusive_compute_cap`. Planned: 10 independent passage/key blocks. Evaluated positives: 3. Fit-worker time: 8.88 hours.

| Block | Input | Source language | Fit winner / excess | Transfer winner / excess | Accepted | Reasons |
|---|---|---|---|---|---|---|
| 1 | positive | english | english / 0.584 | english / 0.374 | none | fit_excess |
| 1 | absent | english | old_french / 1.430 | old_french / 2.428 | none | fit_excess, transfer_excess |
| 1 | shuffle | english | old_french / 1.877 | old_french / 2.828 | none | fit_excess, fit_margin, transfer_excess, transfer_margin |
| 1 | copy | english | old_french / 1.156 | old_french / 2.884 | none | fit_excess, fit_margin, transfer_excess, coverage |
| 2 | positive | italian | italian / 0.405 | italian / 0.404 | italian | passed |
| 2 | absent | italian | old_french / 1.303 | old_french / 2.022 | none | fit_excess, transfer_excess |
| 2 | shuffle | italian | old_french / 1.975 | old_french / 2.905 | none | fit_excess, fit_margin, transfer_excess, transfer_margin |
| 2 | copy | italian | old_french / 1.299 | italian / 3.191 | none | fit_excess, fit_margin, transfer_excess, transfer_margin, winner_changed, coverage |
| 3 | positive | latin | latin / 0.111 | latin / 0.419 | latin | passed |
| 3 | absent | latin | old_french / 1.290 | old_french / 2.177 | none | fit_excess, fit_margin, transfer_excess |
| 3 | shuffle | latin | old_french / 2.011 | old_french / 2.904 | none | fit_excess, fit_margin, transfer_excess, transfer_margin |
| 3 | copy | latin | old_french / 1.208 | italian / 2.917 | none | compute_cap, fit_excess, fit_margin, transfer_excess, transfer_margin, winner_changed, coverage |

## Positive recovery diagnostics

| Block | True language | Fit CER | Transfer CER | Transfer token coverage |
|---|---|---:|---:|---:|
| 1 | english | 6.30% | 6.91% | 98.88% |
| 2 | italian | 5.56% | 5.85% | 98.67% |
| 3 | latin | 6.94% | 9.37% | 98.47% |

## Interpretation

1. **Language ranking and fixed-key transfer worked on all three positives.** English,
   Italian and Latin each won both rankings. Their transfer CER was 6.91%, 5.85% and
   9.37%; each retained at least 98.47% token coverage. These results concern the
   five-language short-passage pipeline, not the newer Italian-only round-six method.
2. **The absolute fit ceiling missed a genuine cipher.** English fit excess was 0.584
   against the frozen 0.50 ceiling; transfer excess was 0.374 and all other conditions
   passed. A post-grading diagnostic finds that the two genuine English plaintexts
   themselves differ by 0.370 bits per letter in prior excess. The fixed ceiling mixes
   source variation with decoding error. Dropping it is a development proposal, not
   a retroactive pass ([diagnostic](english-postgrade-diagnostic.json)).
3. **The copying control was both costly and relatively easy on coverage.** Best-ranked
   transfer keys covered 54.24%, 58.97% and 60.40% of tokens on the three copies, below
   the 95% gate. Their excesses also failed. English and German refinements on the
   Latin copy hit the 1,200-second allowance; its decision is therefore inconclusive,
   regardless of the poor scores. No positive, shuffle or absent-language input hit
   a cap. Stronger copying controls should preserve usable pieces more closely.
4. **Search cost stopped confirmation.** The 45 fits used 8.884 aggregate fit-worker
   hours. The actual stop was `inconclusive_compute_cap`, not exhaustion of the
   12-hour aggregate budget. The next batch would also have failed the declared
   five-hour reserve check: 8.884 + 5 > 12. Seven complete blocks remain unrun,
   including all German and Old French source positives and every second key.

The [first attempt](../rejection-transfer/REPORT.md) had already stopped on three
300-second refinement caps. This resource-repair attempt was independently frozen
at `7915b9e` before generating fresh ciphertext and excluded the first graded English
source IDs. Its larger allowance did not eliminate caps. No further resource increase
was made to this screen. Elapsed calendar time exceeded the recorded active-fit timing; fit-worker time is
the driver's monotonic-clock measurement, not elapsed calendar time.

## Limits and decision

This is a bounded screen with thresholds chosen from the released five-case language-ID pilot. It is not a fresh estimate of a 90% sensitivity / 5% false-positive operating point. Paired negatives share their positive source and are not independent replicates. Early-stop proportions are descriptive; unrun cases are not counted as rejections.

The old language-ID decoder with a 1,200-second refinement cap and two Numba threads, and its default Naibbe spacing, were tested. The newer long-passage Italian decoder, heavier pairing, other cipher families, other languages, and the published Timm generator were not tested.

**No manuscript run is licensed.** A failed screen parks this version; an inconclusive screen needs a declared resource/protocol repair; a pass would require a larger independent control study.

The [three follow-ups](FOLLOW_UP.md) are calibration on released cases, a stronger
copying generator and a cheaper refiner. They belong to separately recorded development.
This screen's thresholds, scores and cap decisions remain unchanged.

## Records and reproduction

[Protocol](PROTOCOL.md) · [Freeze](freeze.json) · [Calibration](calibration.json) · [Sources](sources.json) · [Results](results.json) · [Released records](evaluated-records.tar.gz)

```sh
.venv/bin/python -m experiments.rejection_transfer_v2 verify
.venv/bin/python -m unittest tests.test_rejection_transfer -v
```

Driver subcommands: `sources`, `freeze`, commit, `prepare`, `run`, `report`. Creation and grading refuse overwrites. `run` resumes verified per-input checkpoints. Archive contains only graded answers; unrun answers remain evaluator-only.

[Replay and final verification](final-verification.json) · [Consumed source IDs from both attempts](released-source-ids.json) · [Runtime environment](environment.json)
