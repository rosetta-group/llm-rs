# Proto-Elamite, round five: constant quantity ratios between consecutive entries (exploratory)

Status: written before any ratio was computed, 2026-09-24, branch `proto-elamite`. CPU only.
**Exploratory:** no proto-cuneiform ratio with a documented answer was available to serve as a
known-answer control, so this round has a null model but no gate. Any surviving pair is a lead
for specialist review, not a result.

A structure count done before this protocol (head signs only, no quantities) found 38 ordered
pairs of head signs that follow each other on at least 5 Proto-Elamite tablets.

## Entries and quantities

- Administrative Proto-Elamite tablets; lines of the form `SIGNS , NUMERALS`.
- **Head sign:** the first sign of the line, variant suffix removed.
- **Quantity:** numerals are used only if every token is N01 or N14 and none is broken or
  uncertain. Two readings: N14 = 10 N01 and N14 = 6 N01 (round two's recovered ratios). No other
  numeral value is assumed; lines with other numerals are skipped.

## Statistic

```text
for each ordered pair (A, B) of head signs on consecutive lines, on >= 5 tablets:
  per tablet: ratios q(B) / q(A) under the 10 and 6 readings (one or two values; first occurrence)
  support(r) = tablets whose ratio set contains r; statistic = max over r of support(r)
  null: 2,000 runs, each re-pairing the A quantities with B quantities of other tablets of the
        same pair (random derangement)
  p = share of null runs with statistic >= observed
a pair survives if p < 0.05 / (number of pairs tested) and its support is at least 4 tablets
```

## Records

`experiments/proto_elamite_round_five.py`; results in `experiments/proto-elamite-ratios/results.json`.
