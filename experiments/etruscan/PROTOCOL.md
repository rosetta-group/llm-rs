# Etruscan, round one: does formula context carry meaning class?

Status: written before any run on real data, 2026-09-23, branch `etruscan`. CPU only.
Scope: [SCOPE.md](SCOPE.md). Phase 0 counts: [phase0.json](phase0.json). Frozen in
[freeze.json](freeze.json). Covers scope phases 1 to 4. Nothing here is a reading of Etruscan.

## Data

- **Etruscan, primary: the ETP part of Larth** (`daf4972`): 551 texts, 2,433 tokens.
  The owner left the corpus choice to the analyst. ETP is chosen because CIEP's word division
  is broken (phase 0) and re-segmenting it would need a lexicon built from ETP itself.
- **Etruscan, secondary (reported, no gate): ETP + filtered CIEP.** CIEP texts are kept only if
  they have 2+ tokens, no damaged token, no token of 12+ letters and no o, b, d or g.
- **Latin: LIRE v3.0 pagan epitaphs** (`etruscan/classes.latin_epitaphs`), interpretive text,
  Latin letters only, HTML residue `lt`/`gt` dropped.
- Upper-case Roman numerals become `N:<numeral>` in both languages; a lone `L` or `C` stays a
  word (usually an abbreviated name).

## Classes (`etruscan/classes.py`)

`NAME`, `KIN`, `NUM`, `LIFE`, `OTHER`, defined in the module docstring.

- **Etruscan:** from the ETP word list. POS with praenomen, nomen or cognomen → NAME; POS `num` or
  a Roman numeral → NUM; a gloss word in the kinship list → KIN; in the age/life/death/burial
  list → LIFE; any other entry → OTHER; no entry → unlabelled. ETP counts before the run:
  NAME 371, OTHER 322, NUM 31, KIN 28, LIFE 12; 764 of 1,286 types labelled (59.4%).
- **Latin:** by rule. Roman numeral (I, V, X alone, or 2+ numeral letters) → NUM; `Dis`,
  `Manibus` and a few other god/formula words → OTHER; any other capitalised word → NAME;
  closed lists → KIN and LIFE; other lower-case words → OTHER; other single letters → unlabelled.
  Known noise: capitalised place and tribe names count as NAME.

## Predictors (`etruscan/context.py`)

Both see only a type's neighbours and position (only, first, middle, last), never its letters.

```text
M1 neighbour words:   PPMI over (left word, right word, position); cosine; 5 nearest seed types vote, weighted by similarity
M2 neighbour classes: vectors over (class of left neighbour, class of right neighbour, position);
                      nearest class centroid; non-seed classes updated for up to 10 rounds
```

No parameter was tuned on real data. K = 5 and 10 rounds were fixed before any run.

## Design

```text
for r in 1..20:
    Etruscan: stratified split of labelled ETP types, 20% of each class held out (at least 1)
    Latin:    draw 551 epitaphs with exactly the ETP length distribution (no repeats within a draw)
              keep each Latin type's label with probability = ETP labelled share (about 59%)
              same stratified split
    for each method: score balanced accuracy on held-out types (mean recall over the 5 classes)
        real
        corpus-shuffled: all tokens redealt into texts of the same lengths (context destroyed, frequencies kept)
        permuted labels: seed labels shuffled among seed types
        within-text shuffled: order destroyed, text membership kept (reported, not gated)
```

Chance balanced accuracy is 20% (5 classes; a majority-class guess also scores 20%).

## Gate (scope phase 4)

A method passes if, on the mean over 20 replicates, **for both Latin and Etruscan**:

1. real ≥ 40% (chance + 20 points);
2. corpus-shuffled ≤ 25% (chance + 5);
3. permuted labels ≤ 25% (chance + 5).

If no method passes: report "formula context does not carry meaning class at this size" and
stop before any prediction for unglossed words. If a method passes: the fresh test (scope
phase 5) needs its own frozen protocol and a Wiktionary download the owner has not yet approved.

## Deviation from the scope, decided before any real run

The scope made the within-text shuffle a gated negative control ("must fall to baseline"). On a
synthetic corpus with a fully fixed formula (name name kin life numeral), M2 scored 1.00 real
and 0.95 within-text shuffled: shuffling inside a text keeps which classes co-occur, so it
cannot be a null. The gate now uses the corpus-wide shuffle, which destroys context but keeps
frequencies and text lengths (synthetic: 0.00 to 0.20). The within-text shuffle is still
reported, as a measure of what word order adds.

## Also reported, not gated

- **Anchors (scope phase 1):** leave one word out, seed with every other labelled ETP type,
  predict it. clan, sec, puia → KIN; avils, lupu, svalce, ril, suthi → LIFE; ci, zal → NUM.
  Variants such as `avil` and `clenar` stay in the seeds, so this check is lenient.
- **Secondary corpus:** the same 20 Etruscan splits on ETP + filtered CIEP.

## Known limits

- **Circularity:** ETP glosses were found mostly by reading formulae. Passing shows that a
  context method reproduces the combinatory method at this size, not that the glosses are right.
- **Small classes:** LIFE has 12 ETP types, so 2 or 3 are held out per split; single-split
  scores are noisy, which is why the gate uses the 20-split mean.
