# Qi-tu-ne: role mobility without semantic identification

The HT7 image check confirms that qi-tu-ne occurs with a numeral as well as in list headings.
This establishes a positional contrast, but does not distinguish a person's name from a
group, occupation or affiliation label.

**Count-associated use:** a word followed by a quantity; the referent and counted unit remain open.
**Role mobility:** the same sign sequence appears in more than one written position.

## What was done

- Inspected the photographs, facsimiles, transcriptions and apparatus for both HT7 faces
  in GORILA I pp.14–15; checked live SigLA and the pinned corpus. Re-inspected the edition's
  HT87 and HT117b transcriptions for comparison.
- Added an [eight-row HT7 inventory](INVENTORY.md) and [three occurrence cases](cases.json),
  retaining original signs, physical lines, erasure qualifications and unknown semantic class.
  Froze sources and descriptive driver at **9d47059** before generating the summary.
- Reused the existing source-preservation and literal-occurrence helpers. The run validates
  every HT7 row/sign and all three exact qi-tu-ne occurrence identities. All 49 focused tests
  pass; all eight follow-up freezes verify. No semantic classifier or correspondence test ran.

## Why

A heading was previously a clue toward an administrative category. A verified quantity
after the same word tests whether heading position alone can support that classification.

## Source result

| Object / face | Physical line | Observed text | Written role |
|---|---|---|---|
| HT7b | .1 | qi-tu-ne **1**; next line da-ru-A329 **2** | Count-associated use |
| HT87 | .1–2 | qi-tu-ne · ma-ka-ri-te ·; then six counted entries | List heading |
| HT117b | .1 | qi-tu-ne ·; then ku-re-ju **1**, di-ki-se **1** | List heading |

All three use **AB21f–AB69–AB24** in the source sign layer. The conventional spelling qi-tu-ne
does not require a qi/ki merger or any phonetic reconstruction. No gender is inferred from
the catalogue label AB21f.

```text
Verify the numeral on HT7b against photograph and edition
Compare the identical sign sequence in the two headings
Test both a personal-name assignment and a category assignment
If both reproduce the observed roles, leave semantic class unresolved
```

1. **The numeral is real.** GORILA's photograph and drawing show a unit stroke after the
   three signs on HT7b .1; its normalized transcription prints 1. On HT87/117 the word
   participates in a heading and has no attached numeral. The narrow claim “qi-tu-ne is
   only an unnumbered heading” therefore fails. This does not rule out a numbered heading
   or other administrative use on HT7b.
   [GORILA I, photograph p.14](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=50),
   [transcription p.15](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=51).
2. **There is a personnel context, with a boundary caveat.** HT7a starts `qe-ti` followed
   by **AB100/VIR**. Its five entries have counts 3, 4, 1, 1, 1. HT7b adds two counted
   words but does not repeat VIR. The front supports a personnel interpretation for the
   tablet; carrying that unit onto the reverse is still an assumption. No total is printed,
   and this audit does not invent one or assert how many people the tablet records.
   [SigLA HT7a](https://sigla.phis.me/document/HT%207a/index-word.html),
   [GORILA I, p.15](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=51).
3. **Two different meanings fit the same evidence.** A person could be counted once on
   HT7b and head a list of people under their responsibility on HT87/117. An occupation
   could contribute one worker on HT7b and label its members on HT87/117. A place or
   institution could likewise label contributors. These are explicit hypothetical assignments,
   not translations; the observed positions do not select among them.
4. **Parentage is not distinguished.** The shared entry `di-ki-se 1` under the HT87/117
   headings persists, but a family group and a work group produce the same association.
   There is no independently known relative designation, named-child edge or matched
   name-to-relative substitution. [Rival interpretations](rivals.json),
   [GORILA I, p.137](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=173),
   [p.199](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=235).

## Qualifications and decision

GORILA notes that HT7b .1–2 were written over erased text, and identifies overwriting at
HT7a's re and te signs. Readable extant signs are not evidence of an untouched surface.
No underlying erased words are reconstructed. HT7a/b are **one object, HM10**; the corpus's
face_count=3 also includes an object-level container record, not a third inspected face.
The comparison therefore covers **three objects**, not four independent faces/records.

The [Davis–Valério chapter, p.25](https://www.researchgate.net/publication/352343395_Names_and_designations_of_people_in_Linear_A_A_contextual_study_of_tablets_HT_85_and_117)
conditionally favours non-name interpretations of the HT87/117 headings if their entries
are personal names. HT7b does not refute that proposal: category and affiliation labels can
be associated with quantities. It also does not establish the alternative personal-name reading.

**Decision:** retain qi-tu-ne as an unresolved label with a confirmed heading/count contrast.
It is useful as a counterexample to classifying words solely by position, but is not a
parentage lead on the present evidence. Adding more unlabelled repetitions alone will not
separate the rival assignments. The next useful dependency is a known-language control
that distinguishes names from occupations/group labels when both can head lists and appear
beside quantities. Such a control needs source-labelled examples, held-out physical objects
and an option to abstain. No such classifier has been trained or validated here.

This is manual development collation, without independent epigraphic review or a significance
claim. Earlier failed gates remain failed. Previous frozen records are unchanged; this report
supersedes only their statement that HT7b was not yet image-collated.

## Records and reproduction

[Protocol](PROTOCOL.md), [source hashes and licences](sources.json), [freeze](freeze.json),
[original-sign inventory](inventory.json), [descriptive results](results.json).
SigLA-derived annotations credit Ester Salgarella and Simon Castellan under CC BY-NC-SA 4.0.
Copyrighted GORILA scans remain local reference material, not redistributed in git.
SigLA facsimiles and GORILA facsimiles are not independent witnesses.

```sh
.venv/bin/python -m experiments.linear_a_qi_tu_ne verify
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py'
# Fresh checkout at 9d47059, with pinned local sources attached:
.venv/bin/python -m experiments.linear_a_qi_tu_ne run
```

The run refuses overwrite and emits supplied-annotation counts, not meaning predictions.
The eight-row HT7 table contains seven counted words and one header; no divider or total
is supplied for HT7. The generic inventory renderer's introductory reference to corpus
dividers describes the shared format, not an additional HT7 row.
