# Etruscan round three: source audit before modelling

2026-09-24. No round-three predictive scores inspected when this audit was written.
Raw sources and earlier experiment implementations remain unchanged.

## What was inspected

- All ETP dictionary rows for empty-first-row loss, conflicting classes, suffix and
  inference flags, damaged headwords, and missing/uncertain semantic glosses.
- Twenty ETP records and ten CIEP records, selected by SHA-256 order with the fixed
  salt in the runner. IDs and tokenisations are in `audit-initial.json`.
- Twenty changed dictionary entries selected the same way, plus the known failure
  examples `lautni`, `apa`, and frequent names that the empty-row audit exposed.
- Raw source lines in ETPWords.txt and the alternative ETP records for Cr 5.2 and
  ETP 110. This is a consistency audit of the supplied digital edition, not an
  epigraphist's verification against objects, photographs, or critical editions.

## Repairs and exclusions

1. **First-row loss.** `corpus.glossed_words` uses `setdefault`, so an empty row
   hides subsequent usable rows. Examples: `larth`, `vel`, `arnth`, `lautni`.
   The new version merges all usable rows. It removes class conflicts instead of
   resolving them by row order (`cesu`, `l`, `lavtni`, `suthic`). In the initial
   clean subset, twelve formerly unlabelled spellings acquire labels.
2. **Unknown does not mean OTHER.** The old classifier accepts a grammatical POS
   with no meaning as OTHER. The new labels require a certain gloss, or a definite
   name/numeral POS. For example, `aisece` has an unknown meaning in ETPWords.txt
   and is no longer gold OTHER. Uncertain glosses such as `alpan` are excluded.
3. **Word divisions.** Cr 5.2 contains `av | le` and `laris | al`; its alternative
   rendering still has `av le` and `laris al`. All its records are quarantined.
   No inferred re-segmentation is made. Remaining texts keep explicit separators;
   correctness of every surviving word division is not established.
4. **Spelling duplicates.** ETP 110 appears with Greek chi and with `kh`. The
   pinned source's utils.py maps chi to `kh`, while this project maps it to `ch`.
   The new text loader maps `kh` to `ch` before within-ID deduplication. This also
   removes that duplicate. Other spelling distinctions remain as before.
5. **Damaged text.** Quarantine entire ETP records with restoration/damage marks,
   invalid tokens, or fewer than two words; never join across an omitted token.
   CIEP is excluded from the evaluation. Its ten-record sample contains five
   single-token records, damage, and an apparent fused token `venzazemni`.
6. **Suspicious suffix flags remain unresolved.** `apa` and `ati` are marked
   `Is suffix` in ETP_POS.csv, although raw ETPWords.txt lists `! apa father` and
   `! ati mother`. The meaning of the source marker needs philological checking;
   this run does not override it to recover more labels.

Final pre-model audit: **260 texts, 1,154 tokens, 713 types**. There are **427**
evaluation types in **303** conservative families: NAME 250, OTHER 147, KIN 16,
LIFE 10, NUM 4. The new loader quarantines 252 candidate ETP records and removes
39 near-duplicates after filtering. These numbers are not directly comparable to
round two's 506 texts because cleaning and deduplication are applied in a different
order. All decisions and accepted dictionary rows are in `audit.json`.

## Leakage protection and limits

The 49 old fresh-test words, plus connected relatives, are excluded from both
seeds and evaluation. Eighteen labelled clean-corpus types are excluded by this
rule (including `lautni`). Their unlabelled occurrences remain in the texts.

Families join attested base-plus-suffix spellings and equal source glosses. They
are conservative blocking groups, not asserted linguistic lemmas. This can
overgroup unrelated words: the largest group includes `ara`, `suthi`, and `turce`;
another links `men` with verbs through a suffix relation. Full group membership
and splits are saved so this limitation is visible. Groups prevent several easy
forms of leakage, but cannot certify that every true morphological relation is
captured. No gloss or family identifier is a model feature.

The selection is cleaner but smaller and remains biased towards known words.
It does not reproduce the distribution of genuinely unknown vocabulary. A pass
would justify independent validation, not predictions of new meanings.
