# Post-reparse pruning cleans the lexicon but barely moves CER; heavy pairing doubles the error

Development only, 2026-09-24. Protocol committed before any run (`aa042b7`). Six parallel processes,
about 100 minutes each. No cap was hit. Baseline: reparse pipeline R at 20,800 letters, mean CER 1.79%.
Records: [summary.json](summary.json) and per-run JSON files.

## P: post-reparse pruning, not adopted

| Text | R: CER / WER | P: CER / WER | Pieces, R → P (spurious, missing) |
|---|---|---|---|
| Historical | 1.77% / 20.9% | 1.65% / 20.5% | 496 (200, 38) → 299 (3, 38) |
| Modern | 1.31% / 12.8% | 1.17% / 12.3% | 486 (181, 34) → 311 (6, 34) |
| Verse | 2.28% / 29.2% | 2.09% / 28.7% | 479 (178, 33) → 306 (5, 33) |
| **Mean CER** | **1.79%** | **1.63%** | |

- **Rule not met.** P is 0.15 points below R; the rule needs 0.3. Every text improves slightly.
- **The lexicon is now almost clean, and CER barely moves.** Pruning removes 97–98% of spurious
  pieces, from 178–200 down to 3–6. Once the reparse has run, spurious pieces are nearly harmless: it
  already reads their tokens correctly.
- **The remaining bottleneck is missing pieces.** 33–38 true pieces are still absent, the same count
  before and after pruning. That points the next development step at admitting rare true pieces,
  not at removing false ones.

## H: the reparse pipeline on RESPACING-9 ciphertext (descriptive)

With 75% of letters paired, the only Naibbe setting that matches the Voynich near-duplicate rate:

| Text | CER / WER (RESPACING 17) | CER / WER (RESPACING 9) | Spurious / missing pieces (9) |
|---|---|---|---|
| Historical | 1.77% / 20.9% | 3.36% / 29.8% | 182 / 56 |
| Modern | 1.31% / 12.8% | 3.18% / 25.3% | 270 / 55 |
| Verse | 2.28% / 29.2% | 3.36% / 36.7% | 263 / 52 |
| **Mean CER** | **1.79%** | **3.30%** | |

The decoder is about twice as error-prone on heavily paired text. More tokens are two-piece, so more
piece combinations compete, and missing pieces rise to 52–56. Any Voynich-relevant claim about this
decoder has to be made at this setting, not at the published default.

## Next

1. **Admit rare true pieces.** A candidate should let a piece in when context, not frequency, supports
   it: for example, pieces whose admission lowers the reparse cost of the tokens containing them.
2. **Develop and test at RESPACING 9** as well as 17, since that is the Voynich-relevant regime.
