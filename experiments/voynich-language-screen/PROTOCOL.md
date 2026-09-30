# Language screen on the non-reserved Voynich pages: pre-registration

Pattern: **one-shot screen**. The confirmed language screen is applied once to manuscript text.
Nothing is tuned afterwards; every outcome is reported.

Declared 2026-09-30, before any fit. The user chose this over further decoder tuning.

**What this is not.** It is not the "Voynich mechanism test", which stays closed: the recovery gate
(1% CER and 10% WER on a sealed case) has never been met, and the 30 reserved test pages of the
fixed folio split are not read. No reading, word or translation of the manuscript follows from any
outcome below.

**Screen:** decoder A fits a key on one passage under each of eight language priors; the sealed key
decodes a second passage; `decide_transfer` accepts a language only if it wins both passages by
at least 0.25 bits per letter, its transfer excess is at most 0.50, coverage is at least 0.95, and
no cap was hit. Confirmed on fresh known-answer controls: 16 of 24 true languages accepted at the
published Naibbe setting, 13 of 24 at the Voynich-like setting, 0 false acceptances in about 250
control decisions ([confirmation](../key-recovery-confirmation-v2/REPORT.md),
[RESPACING 9](../respacing9-development/REPORT.md)).

## Data

- ZL transcription (Eva), `artifacts/data/zl/documents.json`, pages with split `train` or
  `validation` (177 pages). Split `test` (30 reserved pages) is excluded in code and checked.
- Tokens: split on word space, uncertain space, line and drawing marks. Tokens containing a glyph
  the Naibbe encoder cannot emit (`?` unknown glyph, `'`, `b`, `j`, `u`, rare glyphs) are dropped:
  2.7% of tokens.
- Blocks, fixed by section, Currier hand and manuscript order; offsets are token counts:

| Block | Fit passage | Transfer passage | Readability |
|---|---|---|---|
| herbal_A | herbal, Currier A, tokens 0–3,400 | herbal A, tokens 3,400–6,800 | 0.930 |
| stars_B | stars, Currier B, 0–3,400 | stars B, 3,400–6,800 | 0.951 |
| balneological_B | balneological B, 0–2,600 | balneological B, 2,600–5,200 | 0.952 |
| mixed_B | herbal B, all 2,877 | text B + cosmological B + stars B from 6,800 (3,023) | 0.914 |

**Readability:** share of transfer tokens that a key built from the fit passage's frequent pieces
could read at all. Released Naibbe controls: median 0.978. Three Voynich blocks sit at or below the
0.95 coverage gate, so "unreadable" is a distinct outcome class below.

- Inputs per block: the manuscript passages, their token shuffle, and their frequency copy (the
  same negatives as every confirmation; seeds from the system RNG, recorded).
- 4 blocks × 3 inputs × 8 priors = 96 fits. Keys are sealed before any transfer passage is decoded.

## Rules

- **Primary:** the confirmed rule: per-run transfer score, eight separate languages, ceiling 0.50.
  Only this rule is claimed.
- **Secondary, reported only:** one length code per passage and Catalan–Occitan as one group
  (`voynich/rejection_v3.py`), 19–21 of 24 on released rounds, never confirmed on fresh text.
- Decoder B (without admission) is recorded for completeness.

## Outcome classes, per input

```text
inconclusive  a cap was hit
accepted      a language passed every gate
unreadable    rejected, and coverage < 0.95 is among the reasons
rejected      rejected with coverage >= 0.95
```

## Interpretation, fixed now

1. **Manuscript input accepted for language L, and that block's shuffle and frequency copy not
   accepted:** consistent with a Naibbe-class cipher of L on that block. The only next step is a
   pre-registered repeat on the reserved pages. No reading is attempted.
2. **A shuffle or frequency copy accepted:** the screen's false-acceptance rate does not hold on
   manuscript-like statistics. Any acceptance of the manuscript input in that block is void.
3. **No manuscript input accepted, some rejected:** a bounded negative. If a block were a Naibbe
   cipher of one of the eight languages at 5,200 letters, the screen would accept it 54–67% of the
   time; the 2,600-token block is weaker than that.
4. **Unreadable:** the block's vocabulary is not composed from a small piece set the way Naibbe
   ciphertext is. This is uninformative about language and is not counted as a rejection.
5. **Always reported:** the full excess table per block, against the released reference bands
   (true model on positives: fit excess −0.40 to 0.38, transfer −0.22 to 1.13; best model on
   negatives: fit 0.32 to 1.04, transfer 1.05 to 2.37).

The recovery gate, the reserved pages and the frozen rule are unchanged by any outcome.
