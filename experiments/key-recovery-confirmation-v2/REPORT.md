# Second fresh confirmation: passes at the threshold (16/24), no false acceptance

Pattern: **sealed known-answer confirmation**. Frozen at `7d56dd1` before any key or passage was
drawn; run once; answers opened once.

The confirmation **passes**. A with the broadened priors accepts 16 of 24 true languages, exactly
the declared minimum. Safety holds: no wrong language, no omitted-language acceptance, 0 of 48
negatives. All 24 blocks completed with no cap.

**Transfer excess:** decoded bits per letter on the transfer passage with the sealed fit key, minus
the model's calibration score; the frozen rule accepts at most 0.50.
**Recovery cost:** transfer excess of the decoded text minus that of the true text, same prior.

## What was done

- 24 blocks, 3 per language, 3 inputs each (positive, shuffle, frequency copy), 8 priors: 576 fits.
- 43.9 fit-worker hours on this Mac; no budget stop, no input inconclusive.
- 48 passages from 46 new works; sources pinned in [sources.json](sources.json).

## Why it was done

Round one failed (13/24) because three priors were too narrow. The rebuilt priors reached 19/24 on
the released round one ([development](../key-recovery-v2-development/REPORT.md)). This round
tests them on new works and keys ([protocol](PROTOCOL.md)).

## Endpoints

| Endpoint | Target | Result | Pass |
|---|---|---|---|
| Safety: wrong language / omitted accepted / negatives accepted | 0 / 0 / 0 | 0 / 0 / 0 of 48 | yes |
| Sensitivity: true language accepted by A | ≥ 16 of 24 | **16** | **yes** |
| Completeness | 24 blocks, none inconclusive | 24, 0 | yes |
| A against B (reported only) | — | 16 vs 8 | — |

## Results by language

| Language | Prior | A accepted | A transfer CER | Why A failed where it failed |
|---|---|---:|---|---|
| English | unchanged | 3 | 2.5, 5.7, 4.1% | — |
| Italian | unchanged | 3 | 3.7, 4.8, 5.3% | — (block 3 shares works) |
| German | broadened | 3 | 6.5, 1.2, 4.6% | — |
| Czech | unchanged | 3 | 3.8, 4.5, 5.1% | — (two later-period blocks pass) |
| Catalan | broadened | 2 | 7.5, 5.6, 8.0% | block 2: Occitan wins the fit passage by 0.13 |
| Latin | broadened | 1 | 6.0, 4.4, 5.5% | blocks 0, 16: fit margin 0.00 and 0.09 against Occitan |
| Old French | unchanged | 1 | 5.9, 10.2, 9.9% | blocks 11, 19: transfer excess 0.589, 0.552 |
| Occitan | unchanged | 0 | 9.5, 15.3, 10.8% | transfer excess 1.128, 0.647, 0.692 |

A lowered transfer CER in all 24 positives. The lowest negative transfer excess was 1.05.

## Post-run diagnostic

True plaintexts were scored under every prior after grading; this did not change any decision.
Details in [plaintext-diagnostics.json](plaintext-diagnostics.json).

```text
failure is a prior mismatch   if the true text alone fails the gate
failure is a recovery failure otherwise
```

- **7 of the 8 failures are recovery failures.** The true text would pass. Median recovery cost
  over all 24 positives is 0.34 bits per letter.
- **Latin blocks 0 and 16:** the true fit text beats Occitan by 1.22 and 1.04 bits; decoding errors
  shrink that margin to 0.00 and 0.09.
- **Occitan blocks 15, 23 and Old French blocks 11, 19:** true transfer excess −0.05 to +0.43; decoding
  adds 0.26–0.70.
- **One prior mismatch:** Occitan block 7 (Girart, a diplomatic transcription with unexpanded
  abbreviations) scores 0.92 as true text.

## What the result supports

1. **The system meets its declared confirmation target on fresh works and keys.** A with these eight
   priors accepts 16 of 24 true languages and none of 72 wrong, omitted or negative decisions.
2. **The pass is marginal.** One fewer acceptance would have failed; per-language results range from
   3/3 to 0/3. With 48 negatives and none accepted, the one-sided 95% upper bound on the negative
   acceptance rate is 6.1%, indicative only because blocks share languages and sources.
3. **Recovery is now the bottleneck, not the priors.** Decoding adds about a third of a bit per
   letter; that cost is what fails Latin, Old French and most Occitan blocks.
4. **Occitan fails on a new source type.** All three blocks come from OCR or diplomatic editions,
   unlike the COMETA text behind its prior; its round-one blocks passed 3/3.
5. **No Voynich conclusion follows.** The manuscript recovery gate (1% CER and 10% WER per case) is
   unchanged, and the reserved pages were not used.

## Records

- [results.json](results.json), [freeze.json](freeze.json), [archive.json](archive.json).
- [evaluated-records.tar.gz](evaluated-records.tar.gz): ciphertext, fits, transfers, sealed keys,
  answers and logs.
- [released-source-ids.json](released-source-ids.json): all used works; exclude from future hidden tests.
- `python -m experiments.key_recovery_confirmation_v2 verify` checks the freeze; `evaluate` regrades.
