# Proto-Elamite, round two: blind arithmetic recovers both N14 ratios, on the control and on Proto-Elamite

Status: complete, 2026-09-24, branch `proto-elamite`. Protocol and code committed before the run
([PROTOCOL.md](PROTOCOL.md)); results in [results.json](results.json). **The gate passed.** This
reads numbers, not language: no Proto-Elamite word is interpreted.

## What was done

- Took every usable tablet (clean numerals, at least two entries, a reverse total) whose entries
  do not already equal the total: 133 proto-cuneiform, 110 Proto-Elamite.
- Searched for 3 number systems at once. Each assigns a value to every frequent numeral sign,
  with N01 = 1, from a fixed grid of 120 values. The score is the number of tablets whose entries
  sum to their total under some system. Hill climbing, 200 restarts.
- Null: each tablet's entries paired with another tablet's total, the same search, 50 runs.

## Why

Round one found the counting ratio N14 = 10 from two-sign tablets but could not see grain
accounts. Letting several systems compete lets each tablet pick the one that balances it.

## Results

| Corpus | Tablets balanced | Best of 50 null runs | p |
|---|---:|---:|---:|
| Proto-cuneiform (control) | 35 of 133 | 31 | 0.02 |
| Proto-Elamite | 43 of 110 | below 43 | 0.02 |

Recovered values, with the number of balanced tablets that fix each one (a value fixed by few
tablets is weak):

| System | Proto-cuneiform (known answer) | Proto-Elamite |
|---|---|---|
| Counting | N14 = **10** (16 tablets), N34 = **60** (3) | N14 = **10** (11 tablets) |
| Grain capacity | N14 = **6** (13), N45 = **60** (6), N34 = **180** (5) | N14 = **6** (9), N45 = 60 (3), N39B = 1/5 (2) |
| Third system | N14 = 9 (5), spurious | N14 = 3 (4), spurious |

Gate: one system with N14 = 10 and another with N14 = 6, each on at least 5 tablets, no null run
reaching the real score, and both found in at least 18 of 20 control subsamples of Proto-Elamite's
size (110 tablets). Observed: yes, yes, yes, 20 of 20. **Passed.**

1. **On the control every well-supported value is right.** The sexagesimal (N14 = 10, N34 = 60) and
   grain capacity (N14 = 6, N45 = 60, N34 = 180) systems come out exactly, with no values assumed.
2. **Chance can support a wrong value on up to 5 tablets.** The third control system settles on
   N14 = 9 with 5 tablets, and many values rest on one tablet. So only values fixed by more than
   5 tablets are read as results.
3. **On Proto-Elamite two values pass that bar:** N14 = 10 N01 on 11 tablets and N14 = 6 N01 on 9.
   These are the textbook counting and capacity ratios. The ambiguity Born et al. resolved by
   assuming them is here recovered from the tablets' own sums.
4. **Everything else is too thin.** N45 = 60 and N39B = 1/5 in the capacity system agree with the
   textbook but rest on 3 and 2 tablets. Values such as N39B = 2 (7 tablets) and N24 = 25 in the
   counting system, or the whole third system, look like fitting noise.
5. **The null is close.** Three free systems balance 31 control tablets by chance against 35 real.
   The count alone is weak evidence; the evidence is that the well-supported values are exactly
   the known ones on the control.

## What this does and does not establish

It establishes that the two N14/N01 ratios used in Proto-Elamite accounts, 10 and 6, can be
recovered from the arithmetic of the tablets without assuming them. The method is validated on
proto-cuneiform first. This confirms the standard reading of the number systems; it adds no new
value and no reading of any non-numerical sign. The null margin is small, and p = 0.02 is the floor
for 50 null runs.

A next round could fix the recovered ratios and ask which tablets each system claims. Then it
could test whether their object signs match the proposed commodity classes: grain signs with the
capacity system, livestock with the decimal one.

## Records and reproduction

```bash
.venv/bin/python -m experiments.proto_elamite_round_two   # refuses to overwrite results.json
```

The tablet counts per value (point 2 to 4) were computed after the run from the same seeds.
