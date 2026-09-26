# Fresh confirmation of whole-token admission: fails on sensitivity (13/24)

Pattern: **sealed known-answer confirmation**. Frozen at `443098f` before any key or passage was
drawn; run once; answers opened once.

The confirmation **fails**. A accepts 13 of 24 true languages, below the declared 16. Safety holds:
no wrong language, no omitted-language acceptance, 0 of 48 negatives. A beats B, 13 against 8.

**Transfer excess:** decoded bits per letter on the transfer passage with the sealed fit key, minus
the model's calibration score; the frozen rule accepts at most 0.50.
**Recovery cost:** transfer excess of the decoded text minus that of the true text, same prior.

## What was done

- 24 blocks, 3 per language, 3 inputs each (positive, shuffle, frequency copy), 8 priors: 576 fits.
- 48.3 fit-worker hours on this Mac; no cap hit, no budget stop, no input inconclusive.
- 48 fresh passages from 46 works; sources pinned in [sources.json](sources.json).

## Why it was done

Development on released cases licensed a fresh test of A ([development](../key-recovery-development/REPORT.md)).
The protocol fixed four endpoints in advance ([protocol](PROTOCOL.md)).

## Endpoints

| Endpoint | Target | Result | Pass |
|---|---|---|---|
| Safety: wrong language / omitted accepted / negatives accepted | 0 / 0 / 0 | 0 / 0 / 0 of 48 | yes |
| Sensitivity: true language accepted by A | ≥ 16 of 24 | **13** | **no** |
| Paired improvement: A over B | A > B | 13 vs 8 | yes |
| Completeness | 24 blocks, none inconclusive | 24, 0 | yes |

## Results by language

| Language | B accepted | A accepted | A transfer CER (3 blocks) | Why A failed where it failed |
|---|---:|---:|---|---|
| Italian | 3 | 3 | 5.2, 4.9, 4.6% | — |
| English | 1 | 3 | 3.2, 4.2, 4.5% | — |
| Occitan | 2 | 3 | 10.8, 5.7, 9.9% | — |
| Czech | 2 | 2 | 4.6, 3.4, 4.0% | block 22: true text already 0.86 over calibration |
| Old French | 0 | 2 | 5.4, 7.7, 9.1% | block 19: recovery cost 0.53, fit margin 0.20 |
| German | 0 | 0 | 11.8, 10.7, 13.3% | true text 0.52–0.56 over calibration |
| Latin | 0 | 0 | 9.4, 4.7, 9.5% | true text 1.15–1.40 over calibration; Occitan or Old French wins |
| Catalan | 0 | 0 | 4.3, 12.9, 9.3% | true text 0.99–1.24 under Catalan, 0.32–0.55 under Occitan; Occitan wins |

A lowered transfer CER in all 24 positives. The lowest negative transfer excess was 0.82.

## Post-run diagnostic

The true plaintext of each transfer passage was scored under every prior after grading. This did
not change any decision. Details in [plaintext-diagnostics.json](plaintext-diagnostics.json).

```text
failure is a prior mismatch   if the true text alone exceeds 0.50 under its own prior
failure is a recovery failure otherwise
```

- **10 of the 11 failures are prior mismatches.** With a perfect key they would still fail the
  0.50 ceiling. Latin narrative prose scores 1.15–1.40 bits over the Aquinas-and-charters
  calibration; German psalms, charters and a martyrology 0.52–0.56 over prose calibration; the
  alchemical Czech block 0.86.
- **Catalan is closer to the Occitan prior than to its own.** The Catalan prior comes from one
  chronicle. On all three new Catalan works the true text scores lower under Occitan.
- **Only Old French block 19 is a recovery failure.** Its true text scores 0.21; decoding adds 0.53.
- Recovery cost across the 24 positives is −0.02 to 0.59 bits, median 0.34.

## What the result supports

1. **Admission improves recovery on fresh text.** Accepted true languages rose from 8 to 13 with
   no added false acceptance. CER fell in all 24 blocks.
2. **The acceptance gate now fails mainly on the priors, not on the key.** Three candidates are too
   narrow for other genres of their own language, so a correct decoding still looks foreign.
3. **Sensitivity cannot be claimed.** The declared target failed; 13 of 24 stands. No threshold,
   prior or passage is changed after the fact.
4. **The next development step is prior coverage.** Broaden the Latin, Catalan and German priors
   across genres, with calibration drawn from matching genres. That needs new development and a
   new fresh confirmation, because these 48 passages are now released.
5. **No Voynich conclusion follows.** The manuscript recovery gate is unchanged and the reserved
   pages were not used.

## Records

- [results.json](results.json): every score, decision and CER. [freeze.json](freeze.json),
  [archive.json](archive.json) (sha256 of the records).
- [evaluated-records.tar.gz](evaluated-records.tar.gz): ciphertext, fits, transfers, sealed keys,
  answers and logs.
- [released-source-ids.json](released-source-ids.json): all 46 works; exclude from future hidden tests.
- `python -m experiments.key_recovery_confirmation verify` checks the freeze; `evaluate` regrades.
