# Proto-Elamite, round three: too few tablets can be labelled to test sign classes

Status: complete, 2026-09-24, branch `proto-elamite`. Protocol and code committed before the run
([PROTOCOL.md](PROTOCOL.md)); results in [results.json](results.json). **The gate failed, so
Proto-Elamite was not tested.**

## What was done

- Labelled tablets as capacity (N14 = 6 N01) or counting (N14 = 10 N01) from arithmetic alone: only
  tablets whose unbalanced part involves just N01 and N14.
- Listed the object signs on each labelled tablet.
- Gate on proto-cuneiform: barley signs should be more frequent on capacity tablets.

## Why

Round two recovered the two ratios. If the tablets that need the capacity ratio are the ones
that name grain, the arithmetic would support sign readings independently.

## Results

| Corpus | Capacity tablets | Counting tablets |
|---|---:|---:|
| Proto-cuneiform (control) | 4 | 11 |

Gate test as run: barley signs ("starting ŠE") on 0 of 4 capacity and 0 of 11 counting tablets,
p = 1.0. **Failed.**

1. **The protocol spelled the barley sign wrongly.** CDLI's ATF writes ŠE as `SZE`, so the test as
   declared could not match. Corrected after the run: `SZE` is on 1 of 4 capacity tablets
   (MSVO 3, 85) and 0 of 11 counting tablets, one-sided p = 0.27. The gate fails either way.
2. **The real limit is the label.** Only 15 of 7,001 control tablets have an unbalanced part confined
   to N01 and N14. That is too few for any association test.
3. **Proto-cuneiform capacity accounts often do not write a grain sign.** The capacity system itself
   marks the commodity, so "grain sign present" is a weak proxy even with more tablets.

## What this does and does not establish

Nothing about Proto-Elamite signs. It shows that a strict arithmetic label covers too few tablets.
A workable version would need more labelled tablets, for example by letting round two's full
systems label every tablet they balance, plus a check that the label is not driven by the fitted
values. It would also need a control group defined in ATF spelling (`SZE`, `UDU`, `GAR`) and
checked against the data before the protocol is frozen.

A small parsing defect also appeared: one numeral written `1(N58)` inside a sign group was read as a
sign on one control tablet. It does not affect the counts above.

## Records and reproduction

```bash
.venv/bin/python -m experiments.proto_elamite_round_three   # refuses to overwrite results.json
```
