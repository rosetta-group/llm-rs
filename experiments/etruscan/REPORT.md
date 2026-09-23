# Etruscan, round one: result

Run 2026-09-23 under [PROTOCOL.md](PROTOCOL.md) (frozen 18945f1, amended before any result).
Results: [round-one-results.json](round-one-results.json); exploratory per-class breakdown:
[round-one-classes.json](round-one-classes.json). Nothing here is a reading of Etruscan.

## Gate: M2 passes, narrowly; M1 fails

Balanced accuracy, mean of 20 replicates. Chance is 20%. Pass needs real ≥ 40% and both
controls ≤ 25%, on both languages.

| | Latin M1 | Latin M2 | Etruscan M1 | Etruscan M2 |
|---|---|---|---|---|
| Real | 40.4% | **43.9%** | 36.6% | **44.1%** |
| Corpus-shuffled (gated) | 19.6% | 15.2% | 19.9% | 19.8% |
| Permuted labels (gated) | 20.7% | 20.0% | 20.2% | 20.2% |
| Within-text shuffled (not gated) | 21.9% | 23.3% | 25.0% | 27.1% |
| Replicates ≥ 40% | 9/20 | 14/20 | 7/20 | 14/20 |
| Gate | pass | pass | fail | **pass** |

1. **Context carries some class signal.** Both nulls sit at chance, so M2's 44% comes from
   neighbours and position, not from word frequency or seed labels.
2. **Word order matters.** Shuffling words inside each text drops M2 from 44% to 27% (Etruscan)
   and 23% (Latin). Most of the signal is in the formula order, not just which words share a text.
3. **The margin is thin.** 44% against a 40% bar, with a replicate SD of 8 points; 6 of 20
   replicates fall below 40% in each language.
4. **Secondary corpus (ETP + filtered CIEP: 1,723 texts, 5,348 tokens):** M2 42.4%, M1 35.7%. Adding
   noisy CIEP text does not help.

## Anchors (not gated): M2 5 of 10, M1 0 of 10

| Word | Expected | M2 | M1 |
|---|---|---|---|
| clan "son", sec "daughter", puia "wife" | KIN | KIN, KIN, KIN | NAME, OTHER, OTHER |
| avils "years", ril "aged" | LIFE | LIFE, LIFE | NAME, NAME |
| lupu "died", svalce "lived", suthi "tomb" | LIFE | NUM, KIN, OTHER | NAME, NUM, NAME |
| ci "three", zal "two" | NUM | KIN, OTHER | OTHER, NAME |

M2 finds the core epitaph formula (name + `clan`/`sec`/`puia` + `avils` + numeral).

## Exploratory, after the result: which classes carry M2

Per-class recall on the same 20 Etruscan splits, and precision computed from the pooled confusion.

| Class | Recall | Precision | Held-out items (pooled) |
|---|---|---|---|
| NAME | 54% | 80% | 1,480 |
| OTHER | 57% | 72% | 1,280 |
| NUM | 51% | 24% | 120 |
| LIFE | 30% | 6% | 40 |
| KIN | 28% | 6% | 120 |

1. **Not only names.** NUM recall is 51%, far above chance, so the scope's negative outcome
   ("only names separate") does not hold.
2. **Rare-class predictions are mostly wrong.** A word M2 labels KIN is kin 6% of the time: 326
   of its 569 KIN calls are names. The balanced-accuracy gate hides this because it weights the
   12 LIFE types as much as the 371 NAME types.
3. **So phase 6 predictions would be weak evidence.** Most unglossed ETP words are hapax names or
   OTHER; a "KIN" or "LIFE" call for one of them would be right about 1 time in 16.

## What this means

Formula context in 2,433 tokens of Etruscan carries a measurable, order-dependent signal about
word class, equal in strength to the same test on size-matched Latin. The signal is too weak
for rare classes to support confident claims about individual unglossed words.

## Next step, not started

Scope phase 5 is the fresh test: seed with the ETP word list, score on words glossed only in
Wiktionary. The Wiktionary Etruscan extract is already pinned from the Linear A track
(`artifacts/linear-a-sources/kaikki/Etruscan.jsonl`, 491 entries), so no download is needed.
Wiktionary may cite the same sources as ETP, so it is not fully independent. The phase needs
its own frozen protocol, and it should report precision alongside balanced accuracy.
