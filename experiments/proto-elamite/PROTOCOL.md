# Proto-Elamite, round one: blind recovery of numeral ratios from balanced tablets

Status: written before any record was parsed, 2026-09-24, branch `proto-elamite`. CPU only.
Scope: [SCOPE.md](SCOPE.md). Data: CDLI search API, downloaded 2026-09-24 with the owner's
approval (`experiments/proto_elamite_fetch.py`), pinned in `sources.json`.

## Question

Can the value ratio between two numeral signs be recovered from arithmetic alone, with no
assumed number system? Answered first on proto-cuneiform, where the answer is known.

## Tablets used

- Genre "Administrative"; periods Proto-Elamite (target) and Uruk III + Uruk IV (control).
- Numerals are tokens `k(Nxx)`. A tablet is dropped if any numeral token is broken or uncertain
  (`[`, `]`, `x`, `...`, `?`); `#` (damaged but read) is kept.
- **Entries:** the obverse lines with numerals. **Total:** the first reverse line with numerals.
  A usable tablet has at least two entries and a total.

## Statistic

```text
for each usable tablet whose numerals use exactly two sign types, larger a and smaller b:
  S_a, S_b = entry sums; T_a, T_b = total
  if T_a != S_a: r = (S_b - T_b) / (T_a - S_a)
  keep r if it is an integer from 2 to 60   (one candidate ratio per tablet)
for each sign pair: support(r) = number of tablets giving r
null: 1,000 runs pairing each tablet's entries with the total of another usable tablet of the
      same sign pair (random derangement); p = share of runs whose largest support >= observed
```

"Larger" is fixed per pair as the sign that more often precedes the other inside a numeral
notation, so no ratio is assumed.

## Gate (control)

Proto-cuneiform, pair (N14, N01): the two best-supported ratios must be 10 and 6 (counting and
grain capacity), each with p < 0.01. Then the same at Proto-Elamite size: 20 random subsamples
with as many usable (N14, N01) tablets as Proto-Elamite has; the gate needs 10 and 6 as the two
best in at least 18 of 20.

## Target, only if the gate passes

Proto-Elamite: every sign pair with at least 5 candidate tablets; top ratios, support and p.
Textbook values, stated before running (Born et al. 2025; Englund): N14/N01 = 10 (sexagesimal,
decimal, bisexagesimal) or 6 (capacity); N34/N14 = 6; N45/N34 = 10; N48/N45 = 6; N23/N14 = 10
(decimal); N51/N34 = 2 (bisexagesimal); N45/N14 = 10 and N34/N45 = 3 (capacity). The fraction signs
(N39B, N24, N30C, N30D) have uncertain values; any ratio found for them is an estimate.

## Records

`experiments/proto_elamite_round_one.py`; results in `experiments/proto-elamite/results.json`.
