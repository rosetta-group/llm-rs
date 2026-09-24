# Proto-Elamite, round five (exploratory): no constant exchange ratio between entries survives

Status: complete, 2026-09-24, branch `proto-elamite`. Protocol and code committed before the run
([PROTOCOL.md](PROTOCOL.md)); results in [results.json](results.json). Exploratory: no known-answer
control was available. No sign is read.

## What was done

- Took ordered pairs of head signs on consecutive entry lines that recur on at least 5
  Proto-Elamite tablets: 46 pairs.
- For each tablet, computed the ratio of the second entry's quantity to the first, with N14 read
  as 10 or 6 N01. Only lines whose numerals are N01 and N14 were used.
- Counted how many tablets share the most common ratio, against 2,000 runs that re-pair
  quantities across tablets. Threshold p < 0.05 / 46 = 0.0011, with at least 4 tablets.

## Why

Fixed ratios, such as grain per worker or per animal, would give roles to the people and animal
signs that round four could not separate.

## Results

| Pair | Tablets with quantities | Most common ratio | Tablets sharing it | Chance | p |
|---|---:|---:|---:|---:|---:|
| **M288 → M124** (grain → overseer) | 5 | **1** | **5** | 3.0 | 0.0005 |
| M305 → M036+N30D | 4 | 5 | 3 | 1.1 | 0.0005 (below 4 tablets) |
| M371 → M124 | 3 | 1 | 3 | 1.0 | 0.0005 (below 4 tablets) |
| M387 → M288 | 4 | 1/2 | 3 | 2.0 | 0.0005 (below 4 tablets) |
| all other 42 pairs | — | — | — | — | ≥ 0.12 |

1. **The one survivor is a ratio of 1.** On 5 tablets the overseer line (M124) repeats the grain
   quantity of the line before it. Equal consecutive quantities are a writing pattern (a repeated
   amount, or an attribution of the same amount). The null re-pairs quantities across tablets, so it
   cannot model repetition inside a tablet. The p-value overstates the evidence.
2. **No non-trivial rate reaches the bar.** The strongest, M305 → M036 composite at 5 : 1, rests on
   3 tablets.
3. **Coverage is the limit again.** Most of the 46 pairs have 2 to 5 tablets with readable N01/N14
   quantities, because lines with other numerals are skipped.

## What this does and does not establish

With the quantities that can be read without assuming numeral values beyond round two's, no fixed
exchange rate between entry signs is found. The M305 → M036 5 : 1 pattern and the M288 → M124
repetition are leads for a specialist, nothing more. A better test would need the other numeral
values fixed first, and a null that keeps within-tablet repetition.

## Records and reproduction

```bash
.venv/bin/python -m experiments.proto_elamite_round_five   # refuses to overwrite results.json
```
