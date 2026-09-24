# Word segmentation v4 fresh test: Sacchetti WER 8.82% to 6.87%, 0.05 points short of the threshold; modern half invalid

Evaluated 2026-09-24. Perfect letters, CPU, under a minute. Development selection is in
`bad065b` and [development.json](../word-segmentation-v4/development.json). The freeze (`8d0ed91`) was
committed before any fresh text was fetched. The Sacchetti pin and extraction record (novelle II–XL,
novella I has no page, 39 italic arguments removed) are in the next commit, before any passage.

**Record order:** the development record was committed after the freeze. It was produced before
the freeze, and the freeze holds its hash.

## Result ([results.json](results.json))

| Source | v3 WER | v4 (mix) WER | Extra / missing spaces | Passages ≤ 10% |
|---|---:|---:|---|---:|
| Sacchetti (historical, new author) | 8.82% | **6.87%** | 220 / 80 → 150 / 84 | 3 → 4 |
| ParTUT train (modern) | 0.36% | 0.36% | invalid, see below | — |

Per Sacchetti passage: 8.58 → 6.70, 7.96 → 8.11, 11.71 → 6.73, 6.90 → 5.98.

- **Transfer threshold: not met.** Historical WER fell 1.95 points; 2 were required. Under the declared
  rule v4 is not promoted, and v3 remains the segmenter.
- **The effect is real but small at this level.** Three of four passages improve and all four are now
  below 10% WER. Extra spaces fall by 32% while missing spaces barely move. Since the result sits
  0.05 points under a line fixed in advance, it is recorded as a miss.
- **The modern half is invalid.** 97.2% of its letters are UD_Italian-ISDT training sentences
  ([audit](../partut-overlap-audit.json)), which is why WER is 0.2–0.6%. The modern condition cannot be
  judged. It would not have changed the decision, which failed on the historical criterion.

## Implication

- The historical spelling mix helps, but not by the declared margin on this author. A new candidate
  would need its own protocol, and Sacchetti is now released.
- **Modern test text:** ParTUT train, like ISDT, must not be used as fresh modern text. Future passage
  construction should reject any sentence that appears verbatim in a fitting corpus, whatever its length.
