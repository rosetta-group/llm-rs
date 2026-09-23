# Length-aware lexicon: linear scaling fails; square-root scaling removes the long-text breakdown but levels off at ~3.2%

Development only, 2026-09-23. The protocol was committed before any run
([PROTOCOL.md](PROTOCOL.md), `745650a`). Six parallel processes, about 2 hours wall-clock. No cap was
hit. The 5,200-letter rows are reused from the [length-scaling test](../length-scaling/REPORT.md),
where the settings are identical.

## Mean polished CER over the three development texts ([summary.json](summary.json))

| Letters | Round four settings | Linear scaling (primary) | Square-root scaling (secondary) |
|---:|---:|---:|---:|
| 5,200 | 5.62% | 5.62% | 5.62% |
| 10,400 | 3.39% | 4.78% | **3.19%** |
| 20,800 | 4.18% | 5.14% | **3.25%** |

Per text at 20,800 letters (CER / WER with v3 / true, spurious and missing pieces):

| Text | Round four settings | Linear | Square root |
|---|---|---|---|
| Historical | 3.00% / 24.3% / 318, 295, 16 | 4.45% / 33.4% / 274, 49, 60 | 3.38% / 27.0% / 296, 200, 38 |
| Modern | 3.84% / 24.1% / 316, 408, 23 | 5.33% / 38.2% / 268, 44, 71 | **2.57%** / **19.6%** / 305, 181, 34 |
| Verse | 5.69% / 39.0% / 314, 400, 20 | 5.64% / 45.1% / 273, 43, 61 | 3.80% / 34.4% / 301, 178, 33 |

## Findings

1. **Decision rule: not met.** Linear scaling, the primary, gives 5.14% at 20,800 letters, above
   2.81%. No sealed long-passage round follows.
2. **Linear scaling over-corrects.** It cuts spurious pieces to 43–49, but misses 60–71 true pieces,
   against 16–23 unscaled. Rare true pieces fall under the raised count threshold, and a missing
   piece costs more than a spurious one.
3. **Square-root scaling removes the breakdown but doesn't keep improving.** It is the best arm at
   both lengths: 3.19%, then 3.25%. It sits between the extremes, with about 180–200 spurious and
   about 35 missing pieces. It is a secondary arm and is not promoted without its own declared test.
4. **A single count threshold has a built-in trade-off.** Raising it removes spurious
   concatenations, and it removes rare true pieces with them. Across these runs, missing and
   spurious pieces trade against each other. The best balance found is near 3.2% CER, still above
   the 1% gate.

## Implication

Frequency alone cannot tell a rare true piece from a frequent concatenation of two true pieces.
The next development candidate needs a discriminator that doesn't rest on counts:
- **Structural:** a spurious piece's count matches the product of its halves' rates (the
  existing concatenation ratio). A test whose power grows with length could replace the fixed θ.
- **Contextual:** the context reparse inside EM from the earlier recommendation, which scores
  alternative splits by the letters they produce.

Use square-root scaling as the length baseline, and declare either candidate against it at
10,400 and 20,800 letters. This is development evidence. No sealed result changes.
