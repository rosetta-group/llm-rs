# Proto-Elamite, round two: joint blind recovery of number systems

Status: written before any code or result, 2026-09-24, branch `proto-elamite`. CPU only.
Round one ([report](../proto-elamite/REPORT.md)) recovered N14 = 10 N01 blind from two-sign
tablets but could not see grain accounts, which use three or more numeral signs. Same data and
parser as round one (`proto_elamite/numerals.py`, `experiments/proto-elamite/sources.json`).

## Tablets used

Usable tablets as in round one (administrative; no broken or uncertain numeral; at least two
obverse entries; a numbered reverse line as total). For each, `d = entry sums − total`, one
integer per numeral sign. A tablet is **informative** if `d` is not all zero, and all its signs
are in the **unit set**: numeral signs present in at least 5 informative tablets. Tablets with any
other numeral sign are set aside and counted.

## Model and search

```text
K = 3 systems; each system k has a value v_k(u) for every unit u; v_k(N01) = 1
grid: 2^a 3^b 5^c for a <= 5, b <= 3, c <= 3 (excluding 1), and their reciprocals down to 1/60
tablet t is balanced if d_t . v_k = 0 for some k
score = number of balanced informative tablets
search: 200 random restarts; each restart starts from random grid values and moves one value of
        one system at a time to the best grid value while the score rises (ties broken at random)
result: the highest-scoring solution; each system is reported by its tablets and values
null: 50 runs, each pairing every tablet's entry sums with the total of another informative
      tablet (random derangement), same search with 20 restarts; p = share of null best scores
      >= the real best score
```

## Gate (control: proto-cuneiform, Uruk III + IV)

1. The best solution has one system with v(N14) = 10 and another with v(N14) = 6, each
   balancing at least 5 tablets, and p < 0.02 (no null run reaches the real score).
2. At Proto-Elamite size: 20 random subsamples of control informative tablets, as many as
   Proto-Elamite has; the gate needs both 10 and 6 recovered in at least 18 of 20 (20 restarts each).

Known control values, stated before running: sexagesimal N14 = 10, N34 = 60, N45 = 600, N48 = 3,600;
grain capacity (ŠE) N14 = 6, N45 = 60, N34 = 180; bisexagesimal N34 = 60, N51 = 120. Capacity
fractions (N39, N24, N30 variants) are smaller than N01.

## Target, only if the gate passes

Proto-Elamite, the same search (K = 3, 200 restarts, 50 null runs): report each recovered system
with its values, tablets and p, next to the textbook values in round one's protocol.

## Records

`experiments/proto_elamite_round_two.py`; results in `experiments/proto-elamite-joint/results.json`.
