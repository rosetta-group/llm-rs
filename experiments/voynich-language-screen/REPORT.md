# Language screen on the non-reserved Voynich pages: no decision reached; manuscript scores like its own shuffle

Pattern: **one-shot screen**, run once as pre-registered in [PROTOCOL.md](PROTOCOL.md) (commits
`0cf1c3b`, `ec029ad`), 2026-09-30. 4 blocks × 3 inputs × 8 priors = 96 fits, decoder A, keys
sealed before transfer. The 30 reserved pages were not read.

The confirmed screen reaches no language decision on any block: three blocks exceed its frozen
work limit and the fourth cannot read enough of its second passage. Nothing is accepted, for the
manuscript or for its negatives. Descriptively, every manuscript passage scores within 0.1 bits per
letter of its own token shuffle and frequency copy, where a real cipher passage beats its shuffle
by about 1.2 bits on the fit passage and 2.2 on transfer.

**Fit / transfer excess:** decoded bits per letter minus the prior's calibration score, on the
fitted passage and on the second passage decoded with the sealed key.
**Coverage:** share of second-passage tokens the key can read at all; the rule needs 0.95.
**Cap:** the key search reached its frozen 20,000,000-proposal limit; the rule marks the input
inconclusive.

## Pre-registered outcomes

| Block | Manuscript | Shuffle | Frequency copy | Capped fits | Coverage |
|---|---|---|---|---|---|
| herbal_A | inconclusive | inconclusive | inconclusive | 24 / 24 | 0.882–0.886 |
| stars_B | inconclusive | inconclusive | inconclusive | 24 / 24 | 0.905–0.909 |
| balneological_B | **unreadable** | unreadable | unreadable | 0 / 24 | 0.915–0.918 |
| mixed_B | inconclusive | inconclusive | inconclusive | 24 / 24 | 0.855–0.864 |

Accepted: 0 of 12 inputs under the primary rule and under the secondary (one length code,
Catalan–Occitan grouped). Rejected with coverage met: 0. Decoder B gives the same classes.

## Why the screen cannot conclude here

1. **Twice the key.** The fitted keys have 652–815 units for the three 3,400-token blocks, against
   a median 356 (maximum 390) on the released Naibbe controls. Pair-swap refinement grows with the
   square of the unit count, so `refine` hit the work limit in all 72 fits of those blocks; the
   2,600-token block (475 units) did not.
2. **Coverage below every control.** 0.855–0.918 here; the lowest control coverage was 0.961.
   The readability estimate in the protocol (0.914–0.952) predicted this.
   Both facts say the same thing: these passages are not composed from a small set of recurring
   pieces the way Naibbe ciphertext of these languages is.

## Descriptive table (pre-registered item 5), decoder A, best-fitting prior

| Block | Input | Best prior | Fit excess | Transfer excess | Coverage |
|---|---|---|---:|---:|---:|
| herbal_A | manuscript | occitan | +0.36 | +1.61 | 0.882 |
| herbal_A | shuffle | occitan | +0.38 | +1.71 | 0.886 |
| herbal_A | frequency copy | occitan | +0.49 | +1.83 | 0.886 |
| stars_B | manuscript | occitan | +0.20 | +1.50 | 0.906 |
| stars_B | shuffle | occitan | +0.32 | +1.52 | 0.905 |
| stars_B | frequency copy | occitan | +0.31 | +1.53 | 0.909 |
| balneological_B | manuscript | occitan | +0.33 | +1.46 | 0.917 |
| balneological_B | shuffle | occitan | +0.29 | +1.36 | 0.917 |
| balneological_B | frequency copy | occitan | +0.43 | +1.56 | 0.915 |
| mixed_B | manuscript | occitan | +0.29 | +1.81 | 0.856 |
| mixed_B | shuffle | occitan | +0.25 | +1.75 | 0.864 |
| mixed_B | frequency copy | occitan | +0.42 | +1.71 | 0.855 |

Reference bands from the second confirmation (decoder A): true model on real cipher passages, fit
−0.40 to +0.38, transfer −0.22 to +1.13; best model on negatives, fit +0.32 to +1.04, transfer
+1.05 to +2.37. Every manuscript transfer excess (1.46–1.81) lies in the negative band; the
fit excess (0.20–0.36) lies where the two bands overlap. Occitan is the best-fitting prior for 11 of
12 inputs, German for one; the winning prior is not meaningful, because it wins for the shuffles too.
Decoder B, without admission, puts the manuscript's fit excess at +0.69 to +0.91, inside the
negative band.

## Post-hoc observation (not pre-registered)

The manuscript passages are indistinguishable from their own scrambles on every measure: fit excess
within 0.12, transfer excess within 0.10, coverage within 0.01, in all four blocks. On the released
controls, a real cipher passage beats its shuffle under the true prior by 0.72–1.79 bits per letter
on fit (median 1.18) and 0.93–3.08 on transfer (median 2.17). Under a Naibbe-class cipher of these
eight languages, the token order of these pages carries no structure the screen can use.

## What this does and does not say

1. **No language of the eight is identified, and none is rejected by the rule.** The pre-registered
   classes are inconclusive and unreadable; by the protocol these are uninformative about language.
2. **The bounded negative is descriptive.** If a block were Naibbe ciphertext of one of these
   languages, the screen's own controls say its original would separate from its shuffle by more than
   a bit per letter. Here it does not. This concerns this cipher family, these eight priors and this
   pairing setting; a cipher of an unmodelled language, another mechanism, or heavier pairing is not
   excluded.
3. **Nothing about meaning.** No word or reading follows. The recovery gate (1% CER, 10% WER) is
   unmet; the mechanism test stays closed; the reserved pages were not read.
4. **A second manuscript run would need two changes first**, each re-confirmed on known-answer
   controls: a higher work limit (the RESPACING 9 negatives needed it too) and a rule that tolerates
   coverage below 0.95. Neither is planned.

## Records

- [results.json](results.json): every score, decision and class; [challenge.json in the archive]
  lists the pages, offsets, seeds and readability of each block.
- [evaluated-records.tar.gz](evaluated-records.tar.gz): inputs, all 96 fits and transfers, logs.
  Hashes in [archive.json](archive.json).
- `python -m experiments.voynich_language_screen evaluate` regrades from the fits.
