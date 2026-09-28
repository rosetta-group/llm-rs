# Broadened priors on the released round one: 19/24 accepted, no false acceptance

Development only, declared in [PROTOCOL.md](PROTOCOL.md) before any fit. The 72 round-one inputs
are released; no round-one work is in any new prior.

The broadened Latin, German and Catalan priors raise A from 13 to 19 of 24 true languages. No wrong
language, omitted language or negative is accepted, and no cap is hit. This meets the declared bar
(16/24) for a second fresh confirmation.

**Transfer excess:** decoded bits per letter on the transfer passage with the sealed fit key, minus
the model's calibration score; the frozen rule accepts at most 0.50.

## What was done

- Built `latin_broad2`, `german_broad2`, `catalan_broad2`: 400,000 training letters each over many
  works, calibration from two held-out works; roles by sha256 order ([sources](../key-recovery-confirmation-v2/sources.json)).
- Fitted all 72 round-one inputs under the three new priors (216 fits, A and B arms); reused the
  round-one fits under the five unchanged priors. Full scores in [results.json](results.json).

## Why it was done

Round one failed because the true text itself scored 0.52–1.40 bits over calibration under the
three narrow priors. Under the new priors the same texts score −0.21 to +0.36.

## Results

| Language | Round one A | Broadened A | Remaining failure |
|---|---:|---:|---|
| Latin | 0 | 2 | block 8: transfer excess 0.534 |
| German | 0 | 3 | — |
| Catalan | 0 | 1 | block 2: fit margin 0.16; block 10: transfer excess 0.651 |
| Italian, English, Occitan | 3, 3, 3 | 3, 3, 3 | — (unchanged priors) |
| Czech, Old French | 2, 2 | 2, 2 | unchanged priors; blocks 22 and 19 as in round one |

- B rises from 8 to 11; A stays ahead.
- Every remaining failure picks the correct language; each misses one gate.
- The lowest negative transfer excess is 0.82; margins against negatives are unchanged.

## What the result supports

1. **Prior coverage was the main bottleneck.** Fixing it for three languages adds six acceptances
   without any false acceptance.
2. **This is development.** The priors were rebuilt after round one showed the problem, and these
   blocks are released. A second fresh confirmation on new works and keys is required.
