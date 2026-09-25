# Participant-complete graphs: data audit

Prepared on 2026-09-24 before model scoring. Source files are the locally pinned
Larth `Etruscan.csv` and `ETP_POS.csv`; hashes are recorded in `freeze.json`.

The training set is all 30 monuments from the previous scaffolding experiment.
Its eight former test cases are now development/training evidence, including the
daughter and mulu families. The 16 new evaluation monuments have different IDs
and different exact normalised token sequences. They come from the same corpus:
this is a new-monument test, not an independent source or withheld-word-family
test. Selection and annotation used the published English translations. The
implementer saw those translations; only the deterministic predictor is blinded.

`manifest.json` preserves the source row, raw reading, English translation,
normalised tokens, manual entity spans, gold graphs, and per-record caveats.
All names in the selected interpretations are included. Name anchors were
manually supplied using the source: this is a favourable oracle condition.

| Cohort | Monuments | Gold relation edges |
|---|---|---:|
| Daughter | AT 1.46, ETP 181, ETP 287 | 5 |
| Transfer | Cr 3.14, ETP 284, ETP 186, ETP 128, Cr 3.20, Cr 3.12 | 7 |
| Ownership | Cr 2.15, ETP 289, ETP 344 | 3 |
| Mixed/complex | Cr 5.3, Ta 1.191, Cl 1.324 | 7 |
| Making | AV 6.1 | 1 |

There are 23 edges across 16 inscriptions, with one to four named people per
inscription. No test entity is a deity. Training includes deity anchors. All
gold graphs are connected and account for every supplied name within the five
relations. This selection favours the model's completeness assumption; it does
not establish that every Etruscan inscription should obey it.

## Transcription and interpretation limits

- AT 1.46 and AV 6.1 contain restored letters in names. Cr 3.20 uses supplied
  `mi<ni>`. The experiment tests those supplied readings, not epigraphic recovery.
- ETP 181's `papaslis a` and Cl 1.324's `rathum nasa` remain separate tokens
  grouped under a single manual name anchor. No spelling repair is inferred.
- ETP 186, Cr 5.3, Ta 1.191, Cl 1.324 and AV 6.1 exercise discontinuous names
  or compound descriptions. ETP 186's donor spans tokens 0 and 3. Cr 5.3 has
  parentage and construction, with contiguous name spans.
- ETP 128 supplies two recipients, Venel and Velkhae Rasunies. The second
  recipient's surname is anchored at token 4. Two edges share OBJECT; this is
  a coarse representation of one gift to two recipients, not two proven events.
- ETP 287 and Cl 1.324 include editorially supplied kinship descriptions.
  Parenthetical readings are accepted as the benchmark's reference, not certainty.
- Age and death clauses in AT 1.46 and Ta 1.191 lie outside the ontology.
  Completeness concerns named participants, not translation of every word.
- `mine` and `itane` are not in the inherited fixed deictic list. They remain
  ordinary anonymous stems. Unknown or conflicting name case/gender remains `?`.

The public inputs contain name type/case/gender, entity positions, generic
deictics, and opaque three-letter stem identities. They exclude translations,
gold labels, raw spellings, and cohort names. No non-name dictionary gloss is
used. The training labels do now expose mulu-family transfers and daughter
relationships from the old evaluation. This cannot demonstrate discovering
either family's previously unknown meaning.
