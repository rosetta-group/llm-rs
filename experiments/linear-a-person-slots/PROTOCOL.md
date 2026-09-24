# Person-slot audit: role before relation

This is a bounded manual source audit of HT 85 and HT 117, with HT 87, HT 88 and HT 122a
as contextual comparisons. It is development work, not a blinded decipherment experiment.
All annotations were made after viewing the sources. No independent expert has reviewed them.

**Slot:** one written entry, whose referent may be a person, group or designation.
**Relation:** an interpreted connection between participants; no Linear A relation is supplied.

## What is supplied

- GORILA I photos, facsimiles and transcriptions for all four target faces; SigLA word/sign
  records; the pinned Navarre corpus; the full Davis–Valério chapter read as web-extracted text.
- All logical rows, including headers, totals, separator and single-sign logograms. Physical
  tablet lines are separately transcribed from GORILA. Two faces remain one physical object.
- Alternative readings and damage remain explicit. The HT117a preferred AB57 reading is
  an audit overlay, never a change to the frozen corpus or earlier exclusion rules.

## Procedure

```text
Inspect edition photographs, drawings, transcriptions and live SigLA readings
Annotate every logical row and map it to physical lines
Preserve word, sign, glyph, erasure and uncertainty evidence separately
Compare possible person entries with administrative, occupation, ownership and place readings
Freeze and commit annotations, source hashes, query list and descriptive code
Validate coverage and list exact source-word occurrences
Report support and limits without assigning a kinship meaning
```

## Fixed scope and decisions

1. Inventory all rows on HT85a/b and HT117a/b, not only apparent names. Quantities are
   published readings; arithmetic is not used to repair damage. One word + 1 is insufficient
   to certify a person. No gender, case or parentage labels are inferred.
2. Review attached or unnamed-relative possibilities as well as two-name formulas. A header,
   adjacent entry or subordinate logogram does not itself express a family relationship.
3. Rank leads qualitatively by layout and independent-object support. Explicit rivals are
   occupation, ownership/responsibility and place affiliation. These are hypotheses, not
   translations. The inventory is not independently labelled training gold.
4. Literal recurrence searches use exactly queries.json against the unmodified word layer.
   They retain duplicate occurrences, physical-object IDs and whole-record quality flags.
   They are a concordance, not a name detector or a clean-source sample. Proposed spelling
   variants remain separate queries. The preferred te-ja-re overlay is not added to hits.
5. No statistical score, phonetic correspondence retest, classifier fitting or target
   meaning-recovery run is authorized by this descriptive audit. Earlier failed gates stand.
   Further automatic relation recovery needs independently reviewed person/designation
   spans and positive/negative whole-object known-language controls.

## Reproduction and provenance

The run refuses overwrite. Commit freeze.json before run. Sources include access dates,
hashes, page numbers and licences. SigLA drawings and GORILA drawings can share an underlying
edition and are not independent readings. Copyrighted scans remain local reference files.
Manual source decisions are in source-review.json; comparative judgments in candidate-review.json.
