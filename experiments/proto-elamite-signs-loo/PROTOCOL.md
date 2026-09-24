# Proto-Elamite, round four: object signs by number system, labels from held-out fits

Status: written before any code or result, 2026-09-24, branch `proto-elamite`. CPU only.
Round three ([report](../proto-elamite-signs/REPORT.md)) could label only 15 control tablets.
This round labels every informative tablet that round two's systems balance, with each tablet's
label taken from systems fitted **without** that tablet. Sign spellings were checked against the
data before this protocol (control: `SZE` on 607 tablets, `UDU` 401, `SAL` 440; `KUR2` absent,
so unused; Proto-Elamite groups all present).

## Labels

```text
for each informative tablet t (round two's definition):
  fit round two's model (K = 3, 20 restarts, fixed seed per tablet) on all other informative tablets
  a fitted system is "capacity" if its N14 = 6, "counting" if its N14 = 10, and it counts only if
    N14 is fixed by at least 5 of the tablets it balances (round two's reliability bar)
  label t capacity if a capacity system balances it and no counting system does; vice versa;
    otherwise unlabelled
```

## Object signs

As round three (`proto_elamite/signs.tablet_signs`): base signs, variants removed, compounds split.

## Tests (one-sided Fisher exact on labelled tablets)

| # | Corpus | Group | Prediction | Role |
|---|---|---|---|---|
| G | proto-cuneiform | `SZE` (barley) | more on capacity tablets | **gate**, p < 0.05 |
| C1 | proto-cuneiform | `UDU` (sheep), `SAL` (woman) | more on counting tablets | reported |
| 1 | Proto-Elamite | grain: M288, M036, M297 | more on capacity tablets | test, p < 0.025 |
| 2 | Proto-Elamite | animals and people: M388, M124, M346, M367, M006, M362, M376 | more on counting tablets | test, p < 0.025 |

Tests 1 and 2 run only if the gate passes.

## Records

`experiments/proto_elamite_round_four.py`; results in `experiments/proto-elamite-signs-loo/results.json`.
