# Proto-Elamite, round one: arithmetic recovers the counting ratio blind, but not the grain ratio

Status: complete, 2026-09-24, branch `proto-elamite`. Protocol, sources and code committed before
the run ([PROTOCOL.md](PROTOCOL.md), [sources.json](sources.json)); results in
[results.json](results.json). **The gate failed, so Proto-Elamite was not analysed.**

## What was done

- Downloaded 9,577 CDLI records at one request per second: Proto-Elamite 1,756; Uruk III 5,929;
  Uruk IV 1,892 (`experiments/proto_elamite_fetch.py`).
- Parsed administrative tablets into obverse entries and a reverse total (`proto_elamite/numerals.py`).
  Tablets with any broken or uncertain numeral were dropped.
- Solved every tablet whose numerals use exactly two sign types for the ratio between them, and
  counted which ratio the most tablets support. The null pairs each tablet's entries with another
  tablet's total, 1,000 times.

## Why

Every published Proto-Elamite analysis assumes the textbook ratios between numeral signs. If
arithmetic alone can recover them on proto-cuneiform, where they are known, the same method can
test them on Proto-Elamite, and estimate the uncertain fraction signs.

## Results

| Corpus | Administrative | Clean | Usable (≥ 2 entries and a total) |
|---|---:|---:|---:|
| Proto-cuneiform (control) | 7,001 | 3,710 | 174 |
| Proto-Elamite (target) | 1,616 | 798 | 128 |

Control, pair N14/N01, 31 two-sign tablets:

| Ratio | Tablets | p | Known value |
|---:|---:|---:|---|
| **10** | **10** | **0.001** | counting systems |
| 9 | 3 | 0.26 | — |
| 11, 15 | 2 each | — | — |
| 6 | 2 | — | grain capacity |

Chance gives a largest support of 2.2 tablets on average. Gate: 10 and 6 as the two best, each
p < 0.01, and in 18 of 20 subsamples at Proto-Elamite's size (33 tablets). Observed: 10 yes, 6 no;
0 of 20. **Failed.**

1. **The counting ratio comes out of the arithmetic alone.** Ten tablets balance only with
   N14 = 10 N01 (for example MS 2504, CUSAS 01, 181), against 2.2 by chance.
2. **The grain ratio is invisible to this test, not refuted.** Grain accounts use three or more
   numeral signs (capacity fractions such as N39 and N24), so they never enter a two-sign test.
   The 33 usable tablets with capacity signs all have three or more sign types.
3. **Most tablets cannot enter any sum test.** 3,491 of 3,710 clean control tablets have no numbered
   reverse line, and 120 of the 174 usable ones use three or more sign types.

## What this does and does not establish

Blind ratio recovery works where a tablet reduces to one unknown: the counting ratio is found at
p = 0.001 from 31 tablets. It does not yet reach the grain system, so it cannot be trusted on
Proto-Elamite, whose ambiguity is exactly 10 against 6. The next step is a joint search over
tablets with three or more numeral signs, with each tablet's number system as a latent choice. That
design needs its own protocol and the same control.

## Records and reproduction

```bash
.venv/bin/python -m experiments.proto_elamite_fetch        # resumes; 1 request per second
.venv/bin/python -m experiments.proto_elamite_round_one    # refuses to overwrite results.json
```

Diagnostic counts in point 3 were run after the gate, on the control only.
