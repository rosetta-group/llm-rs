# Etruscan, round two: deduplicated gate re-run and the fresh Wiktionary test

Status: written before any run, 2026-09-23, branch `etruscan`. CPU only. Follows
[round one](../etruscan/REPORT.md); scope phases 4 (re-run) and 5. Frozen in
[freeze.json](freeze.json). Nothing here is a reading of Etruscan.

## Why

1. **Duplicates.** Round one's ETP corpus kept near-duplicate rows of one inscription
   (`vel leinies ...` twice, `semni ramθa ...` twice). The gate is re-run without them.
2. **Fresh test.** Round one scored M2 against held-out ETP glosses only. Scope phase 5 asks
   whether M2 also recovers classes for words ETP does not gloss, scored against a second source.

## Deduplication (`etruscan.corpus.dedupe_within_id`)

Within one text ID, a text is dropped if its word multiset (Roman numerals case-folded) has
Jaccard overlap ≥ 0.8 with an earlier text of that ID. Texts with different IDs are all kept:
short ownership texts such as `mi larices` recur on different objects, and one ID can hold
several distinct parts (Liber Linteus sections). Effect, counted before any run: 45 ETP texts
dropped; ETP is now 506 texts, 2,188 tokens. No CIEP text is dropped.

## A. Gate re-run on deduplicated ETP

Round one's design and gate, unchanged, on the deduplicated ETP (Latin redrawn to the new
length distribution and labelled share). Only M2 is gated; M1 is reported. If M2 fails here,
round one's pass is attributed to duplicates and B is reported as unsupported.

## B. Fresh test against Wiktionary

- **Gold:** the Wiktionary Etruscan extract already pinned on the Linear A track
  (`artifacts/linear-a-sources/kaikki/Etruscan.jsonl`, CC BY-SA; `etruscan/wiktionary.py`).
  Old Italic headwords and forms are transliterated to the corpus spelling. Form-of entries
  ("genitive singular of X") take X's class; `name` entries are NAME unless they are gods,
  places, months or mythological figures; "slave" counts as household status (KIN), as
  "freedman" does. A spelling given two classes is dropped.
- **Seeds:** every corpus type with an ETP word-list label (all of them; nothing held out).
- **Items:** corpus types with a Wiktionary label and no ETP label.
- **Primary corpus:** deduplicated ETP + filtered CIEP (round one's filter): 49 items
  (NAME 33, OTHER 10, NUM 4, KIN 2). Chosen over ETP alone because ETP alone has 17 items.
- **Secondary:** deduplicated ETP only: 17 items (NAME 8, NUM 4, OTHER 3, KIN 2).
- **Method:** M2 only, as in round one (no parameter changed).
- **Nulls:** 1,000 corpus shuffles and 1,000 seed-label permutations; one-sided p =
  (1 + nulls ≥ real) / 1,001.

**Pass (primary corpus):** balanced accuracy over the classes present (4 here, chance 25%)
≥ 45%, and p < 0.01 against both nulls. Precision and recall per class are reported.

## Known limits, stated before the run

- **Low power.** 49 items, two-thirds NAME; KIN has 2 items, LIFE none. One KIN item moves
  balanced accuracy by 12.5 points. A pass says M2 separates names, OTHER and numerals on
  unseen words; it says little about kinship and nothing about LIFE.
- **Not independent.** Of 59 words labelled by both sources, 55 agree: Wiktionary draws on
  the same scholarship as ETP. This is a test on new words, not on new evidence.
