# Etruscan, round two: result

Run 2026-09-24 under [PROTOCOL.md](PROTOCOL.md) (frozen fb0f522). Results:
[results.json](results.json); exploratory coverage check: [context-coverage.json](context-coverage.json).
Nothing here is a reading of Etruscan.

## A. Gate re-run on deduplicated ETP: passes

45 near-duplicate texts dropped (ETP 506 texts, 2,188 tokens). Balanced accuracy, mean of 20
replicates, chance 20%:

| | Latin M2 | Etruscan M2 | Etruscan M1 |
|---|---|---|---|
| Real | 44.8% | 45.0% | 36.3% |
| Corpus-shuffled | 19.0% | 17.6% | 20.2% |
| Permuted labels | 16.6% | 19.4% | 19.9% |
| Within-text shuffled (not gated) | 25.4% | 29.1% | 24.0% |

Round one's pass was not caused by duplicates (44.1% before, 45.0% after).

## B. Fresh test against Wiktionary: fails

M2, seeded with every ETP label, scored on corpus words that only Wiktionary glosses.

| | Primary: ETP + CIEP (49 items) | Secondary: ETP (17 items) |
|---|---|---|
| Balanced accuracy (chance 25%) | 28.1% | 15.6% |
| Accuracy | 24.5% | 17.6% |
| p vs corpus-shuffled | 0.18 | 0.62 |
| p vs permuted labels | 0.19 | 0.63 |
| Pass (≥ 45%, p < 0.01) | no | no |

1. **Names are missed.** Only 9 of 33 fresh names are called NAME; the rest are spread over
   KIN (12), LIFE (8) and NUM (4). Round one's name recall was 54%.
2. **Hits are few and scattered.** `lautni` → KIN, `machs` → NUM, `acasce` → OTHER; `apa`
   "father" → NUM, `thu` "one" → LIFE, `menrva` (Minerva) → KIN.

## Why round one and round two disagree (exploratory, after the result)

| | ETP-labelled words (round one's pool) | Fresh words |
|---|---|---|
| Words | 772 | 49 |
| Seen once | 64% | 73% |
| Neighbours that are labelled | **82%** | **24%** |

M2 predicts a word's class from the classes of its neighbours. Round one held out words
drawn from well-glossed texts, so their neighbours were mostly known. Words that scholars
have not glossed sit in texts where the neighbours are also unglossed, mostly CIEP. There M2
has almost no information. **Round one's held-out score overstated how well M2 works on the
words that matter.** This is a selection effect: a random held-out split does not look like
the real target.

## Status

Scope phase 6 (predictions for unglossed words) is not run: the fresh test failed, and the
coverage check shows why it would keep failing on this corpus. A context method would need
many more clean, word-divided texts around the unglossed words. The open CIEP extraction
does not supply them. A clean digital edition of Rix's *Etruskische Texte* would, but none
is openly licensed.
