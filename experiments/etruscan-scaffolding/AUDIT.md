# Names as scaffolding: 30-record audit

2026-09-24. Written before any real model scores. This is a deliberately selected
small feasibility study, not a random sample or an independent expert annotation.
The annotator inspected the source texts and supplied English translations. No
claim is made that the test meanings were unknown to the annotator.

## Source and annotation

All 30 records come from the pinned Larth ETP/Zikh Rasna portion of Etruscan.csv.
For each selected inscription ID, the first source row with an English translation
is stored verbatim with its row index in [manifest.json](manifest.json). Alternative
rows with that ID are excluded from both training and test. Existing corpus
normalisation is used, plus the previously audited `kh` to `ch` equivalence.

Each record has a manual inventory of source-token spans for people and deities,
plus a relation graph transcribed from the English. Spans identify names, not roles.
The model is explicitly given favourable, audited name anchors. It does not have to
discover names or decide whether a name is a deity. This is not an end-to-end blind
translation benchmark.

Grammatical case and gender are merged from *all* usable name dictionary rows.
Only entries marked as names or deities are loaded; non-name meanings are never
loaded into model features. Conflicting values become unknown. In particular,
`larthia` has both masculine-genitive and feminine-direct entries: neither is
silently selected. A given name's attributes take precedence over those of an
adjacent family name, because the source often records the latter as genitive.

## Selected monuments

| Group | IDs | Treatment |
|---|---|---|
| Son formulas | ETP 192; Cr 1.10; Cl 1.1134; Vc 1.84; Ta 1.62; Cl 1.1006 | Training |
| Daughter formulas | Ta 1.13; Ta 1.59; Cl 1.1885 | Entire translations and graphs hidden |
| Wife contrasts | Cl 1.373; ETP 230; At 1.111 | Training |
| Object/person associations | Cr 2.20; Cr 2.18; Cr 2.2; Vs 1.86; Sp 2.36 | Training |
| mulu-family gifts | ETP 269; Vt 3.1; Cr 3.11; Cr 3.9; Cr 3.10 | Entire translations and graphs hidden |
| Other transfer wording | Cr 3.7 | Training; `aliqu`, not the hidden stem |
| Dedications | Co 3.7; ETP 339; ETP 238; Ve 3.30; ETP 189 | Training |
| Making | ETP 304; Ve 6.2 | Training |

## Decisions visible in the annotation

- **Multiple roles stay distinct.** Ta 1.13 names a woman, her father, and her
  husband; Cl 1.1885 names a woman, her husband, and her mother. A generic “some
  kinship” call cannot pass: the parent endpoint must be right.
- **Nested pedigrees stay nested.** Ta 1.59 has daughter-to-father and
  father-to-grandfather edges. Replacing both with edges from the first person
  would be wrong. Only the explicit daughter edge is the hidden target.
- **Donor and recipient are not interchangeable.** ETP 269 gives both; Cr 3.10
  gives a recipient with no named donor. Unspecified participants are retained
  as UNSPECIFIED, never invented.
- **Discontinuous names stay one entity.** In Ve 3.30 the dedication verb splits
  Thanirsiie from Fuluves. Source tokens 1 and 3 belong to one donor, not two people.
- **Some relations are editorial.** Several son/wife relations are supplied in
  parentheses in the source translations. This study learns those interpretations;
  it cannot independently establish them.
- **Two supplied readings have restoration marks.** At 1.111 has restored letters
  inside a name; Ve 6.2 restores `mi<ni>`. Both are disclosed, favourable training
  readings. No missing relation word is restored for a test item.
- **The object is coarse.** OBJECT means the inscribed object. OWNED_BY means
  “of/associated with”, including a tomb associated with a deceased person, not
  a demonstrated modern legal ownership relation. Gave and dedicated are pooled
  into TRANSFER. These abstractions limit the meaning of a successful result.

## Pre-run discovery: structural collisions

After name anonymisation and hiding non-name meanings, Cr 3.9 (“gave me”) and
Ve 6.2 (“made me”) have the same structural sequence. A local template alone
cannot distinguish the two actions from those features. This was noticed in the
input audit, before predictions, and motivated the jointly scored stem-family
model. Initial public features/manifest are retained as `audit-initial-*.json`.

The final public input retains an opaque hash of the first three letters of
non-name words. This links the `mul...` forms across separate texts without
revealing their meanings. Query positions are supplied because the task asks for
a selected missing word's role. The hashes contain neither English class names
nor the private target-family labels used in scoring. This is a crude, explicit
stemming assumption, not a validated morphological analysis.

Pooling counts identical abstract templates only once within each queried stem.
For example, Vt 3.1 and Cr 3.11 cannot manufacture extra confidence merely by
repeating one pattern. Both monuments still count separately in the descriptive
test scores. Statistical independence of the eight outcomes is not assumed.

## Information barrier

[public.json](public.json) is the predictor's complete source-side input: anonymous
entities, source grammar, deictic/object indicators, anonymous stem identities,
and queried stem IDs. Exact names, English translations, target-family names and
gold role graphs are absent. Training examples receive their own English-derived
graphs separately. All eight test graphs remain in the evaluator.

The corpus is not independently fresh: the previous experiments and this audit
have inspected these sources. The narrower protection is that the predictor does
not receive hidden translations or target word meanings and its settings are
frozen before its scores are computed.
