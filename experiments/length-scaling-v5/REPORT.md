# Context admission of rare pieces: a clean negative (admits ~1,000 pieces, 2–3% true; CER triples)

Development only, 2026-09-24. Protocol committed before any run (`84bc569`). Six parallel processes,
about 2 hours each. No cap was hit. [summary.json](summary.json) and per-run JSON files.

## Result

| Setting | Baseline R mean CER | With admission | Adopted |
|---|---:|---:|---|
| RESPACING 17 | 1.79% | 6.24% | no |
| RESPACING 9 | 3.30% | 9.99% | no |

| Text, setting | Candidates | Admitted | True among admitted | Missing true units after | CER before → after admission |
|---|---:|---:|---:|---:|---|
| Historical, 17 | 3,997 | 1,061 | 24 | 61 | 1.95% → 5.11% |
| Modern, 17 | 3,636 | 832 | 28 | 61 | 1.34% → 4.57% |
| Verse, 17 | 4,128 | 1,148 | 18 | 66 | 2.49% → 9.05% |
| Historical, 9 | 4,613 | 1,484 | 33 | 72 | 3.40% → 10.42% |
| Modern, 9 | 4,439 | 1,350 | 29 | 81 | 3.51% → 8.77% |
| Verse, 9 | 4,938 | 1,597 | 31 | 76 | 3.46% → 10.78% |

## Why it fails

- **The bar is far too low for the search.** Each candidate picks its best letter from 23, and there
  are about 4,000 candidates per text. A 10-bit saving is easy to find by chance. The prior always
  prefers a more typical letter to a correct but less typical one, so "fitting the context better"
  often means overwriting a correct reading.
- **Admitted false pieces displace true ones.** Missing true units rise from 33–38 to 61–81, because
  reparse readings that use an admitted false piece replace correct parses.
- **Precision is 2–3%.** Around 20–30 true pieces are found per text, but they come with about 40 times
  as many false ones.

## Implication

Admission has to charge the search, not just the key entry. A future candidate would need at least:

- a cost that grows with the number of candidates and letters tried, such as $\log_2(\text{candidates}
  \times 23)$ per admission, which is about 17 bits here;
- a held-out check, admitting only if the saving on half of the occurrences predicts a saving on the
  other half;
- one piece per rare token, not every reading.

Until then, the reparse pipeline R remains the method: 1.79% at RESPACING 17 and 3.30% at 9 on
development text, and 1.83% on sealed historical text.
