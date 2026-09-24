# Proto-Elamite, round four: tablets that need the grain ratio carry the proposed grain signs

Status: complete, 2026-09-24, branch `proto-elamite`. Protocol and code committed before the run
([PROTOCOL.md](PROTOCOL.md)); results in [results.json](results.json). **The gate passed; test 1
passed, test 2 failed.** Nothing here reads a word.

## What was done

- For each informative tablet, refitted round two's three-system model on all the other tablets,
  then labelled the tablet by which fitted system balances it: capacity (N14 = 6 N01) or counting
  (N14 = 10 N01). A system counted only if at least 5 of its tablets fix its N14 value.
- Compared the object signs on capacity and counting tablets with one-sided Fisher exact tests.
  The sign groups were fixed in the protocol from the literature; their ATF spellings were checked
  against the data first.

## Why

Round three's strict label covered 15 control tablets. Labels from held-out fits cover more
tablets, and a tablet never labels itself, so its label cannot come from its own fit.

## Results

| Test | Group | Capacity tablets with group | Counting tablets with group | p |
|---|---|---:|---:|---:|
| Gate, proto-cuneiform | `SZE` barley | 5 of 9 | 1 of 14 | **0.018** |
| C1, proto-cuneiform (reported) | `UDU` sheep, `SAL` woman | 1 of 9 | 4 of 14 | 0.33 (towards counting) |
| 1, Proto-Elamite | grain: M288, M036, M297 | **10 of 11** | **1 of 8** | **0.0012** |
| 2, Proto-Elamite | animals and people: M388, M124, M346, M367, M006, M362, M376 | 6 of 11 | 5 of 8 | 0.55 (towards counting) |

Labelled tablets: control 9 capacity, 14 counting, 110 unlabelled; Proto-Elamite 11 capacity,
8 counting, 91 unlabelled.

1. **The control works for barley.** Proto-cuneiform tablets that balance only with the grain
   ratio name barley five times more often (5 of 9 against 1 of 14), p = 0.018.
2. **On Proto-Elamite the grain signs follow the grain ratio.** 10 of the 11 tablets that balance
   only with N14 = 6 carry M288, M036 or M297, against 1 of 8 counting tablets, p = 0.0012. This
   supports, from arithmetic alone, the proposed readings of these signs as grain containers or
   grain products.
3. **Animal and people signs do not separate.** They appear on about half of both groups (p = 0.55),
   and the matching control check (sheep and women on counting tablets) is also not significant
   (p = 0.33). People and animals receive grain rations, so they also appear on grain accounts.
   A two-way label cannot isolate them.

## What this does and does not establish

With 19 labelled Proto-Elamite tablets, the tablets whose sums need the capacity ratio are
almost all ones that mention M288, M036 or M297. That is independent support for the textbook
view that these signs count grain. The support is independent of how those readings were first
proposed, but not wholly so: M288 was already called "capacity-counted" from its notations.
It says nothing about the language or any non-numerical sign beyond this association. The
counts are small, and one grain-group sign, M288, is on 27% of all Proto-Elamite tablets.

## Records and reproduction

```bash
.venv/bin/python -m experiments.proto_elamite_round_four   # refuses to overwrite results.json; about 7 minutes
```
