# Etruscan evidence audit — 2026-09-24

This **evidence ledger** checks nine previously exposed inscriptions against cited publications.
It improves the reference annotations; it provides no new model score or decipherment claim.

**Corroborated:** a cited passage supports the represented relationships in the dataset excerpt.
**Disputed:** a source documents alternatives that this audit cannot settle.
**Carried forward:** an old relationship remains visible without new source endorsement.

## What was done

- Built [annotations.json](annotations.json) with original token indices, grammatical spans,
  relationship claims, alternatives, pinpoint citations and source hashes.
- Checked six sources, including five locally pinned PDFs; inspected the relevant rendered pages.
- Recorded three corroborated inscriptions, one partly corroborated, three disputed and two
  unverified. Six relationship claims are source-supported. All nine await expert review.
- Preserved the original texts, name anchors, reference graphs and experiment results.

## Why it was done

The clause pilot failed to bind roles reliably, and some supplied answers or name anchors
need qualification. Another fit to these exposed cases would not provide a fresh test.

## Findings

| Inscription | Status | Source-backed finding and annotation consequence |
|---|---|---|
| Ta 1.168 | Corroborated | The wife construction supports `SPOUSE_OF`, not the model's parentage edge. [Schulze-Thulin, p. 182](https://www.studietruschi.org/wp-content/uploads/2021/06/SE58_11.pdf). Death, age and unnamed children receive separate spans; the exact age differs between readings and remains unresolved. [López Montero, p. 113](https://ddd.uab.cat/pub/faventia/faventia_a2012-14v34-36/faventia_a2012-14v34-36p111.pdf). |
| Cr 5.2 | Corroborated | Laris and Avle are sons of Laris and commissioned the tomb: two parentage and two construction relations. `MADE` here includes commissioning. The published text continues beyond our excerpt. [Rigobianco, pp. 68–69](https://iris.unive.it/bitstream/10278/5107210/1/La%20famiglia%20etrusca%20nelle%20fonti%20epigrafiche%20di%20et%C3%A0%20classica.pdf). |
| Cr 3.20 | Corroborated | The explicit active gift formula identifies Aranth as donor and Ramutha Vestiricinai as recipient. This does not settle every gift formula. [De Simone, p. 116, TLE 868](https://www.studietruschi.org/wp-content/uploads/2021/07/SE38_08.pdf). |
| Ve 3.2 | Partly corroborated | The syntax analysis treats `mene` as an object pronoun and `muluvanice` as the gift verb. Our three-letter representation conflated `mene` with training `men`; these must remain distinct. The complete two-donor graph is still carried forward. [Schulze-Thulin, p. 190, §6.2.3.1](https://www.studietruschi.org/wp-content/uploads/2021/06/SE58_11.pdf). |
| Cr 3.18 | Disputed | A historical discussion leaves agent/recipient readings of `mulu` plus `-si` formulas open. Store alternative graphs, not a forced correction to the donor reference. [De Simone, p. 116 and pp. 119–120 n. 14](https://www.studietruschi.org/wp-content/uploads/2021/07/SE38_08.pdf). The [Met's account](https://www.metmuseum.org/essays/etruscan-language-and-inscriptions) identifies the gift and person but does not formally decide the role. |
| Vt 1.58 | Disputed | A 2025 study explicitly flags the name formula as ambiguous. [Rigobianco, p. 74 n. 44](https://iris.unive.it/bitstream/10278/5107210/1/La%20famiglia%20etrusca%20nelle%20fonti%20epigrafiche%20di%20et%C3%A0%20classica.pdf). [CIE I, p. 10, entry 11](https://www.studietruschi.org/wp-content/uploads/2025/01/CIE-I_tit.1_474.pdf) records competing readings; the existing transcription stays versioned, not certified. |
| AV 6.1 | Disputed | `titenas` may be another object or the subject's family name. The supplied single-person anchor is therefore uncertain. [Schulze-Thulin, p. 190 n. 6](https://www.studietruschi.org/wp-content/uploads/2021/06/SE58_11.pdf). |
| Ru 5.1 | Unverified | This bounded search located no independent passage settling the complete role graph. Keep the old interpretation explicitly unverified. |
| Vs 1.28 | Unverified | Same source gap; absence of corroboration does not establish that the old interpretation is wrong. |

## How to use the ledger

```text
Read the original excerpt and its cited passage
Keep disputed alternatives separate
Obtain expert adjudication before promoting a disputed relationship
Use these exposed cases for development only
Reserve different, independently annotated monuments for a future frozen test
```

1. **Preserve provenance.** `original` pins the historical manifest and reference graph.
   Token indices are zero-based in that excerpt; `e0`, `e1`, etc. retain its supplied people.
   Relation `tokens` identify supporting cues, not every argument. An empty span means no
   cue alignment is asserted. Entity anchors themselves may be disputed, as in AV 6.1.
2. **Keep uncertainty local.** `relations` is a claim list, not an executable gold graph.
   Cr 3.18's alternatives must not be unioned into two gifts. A corroborated record refers
   only to its represented relationships; Ta 1.168's age remains uncertain.
3. **Separate source support from independence.** These publications share earlier editions
   and scholarship. Their agreement is not six independent archaeological observations.
   Span-to-token and relation mappings were made by Codex, not supplied by the authors or
   checked by an external expert. All 66 prior monuments remain development material.
4. **Make review concrete.** First adjudicate Cr 3.18's role and Vt 1.58/AV 6.1's name
   anchors using the cited passages. Seek full-role sources for Ve 3.2, Ru 5.1 and Vs 1.28.
   No new experiment is justified by this audit alone, and no old score was recalculated.

## Reproduction and integrity

Run from the repository root:

```sh
.venv/bin/python -m experiments.etruscan_evidence verify
.venv/bin/python -m unittest discover -s tests -p 'test_etruscan*.py'
```

`verify` checks annotation bytes, pinned PDFs, historical manifests, spans, participants and
citations. The tests also run without research PDFs; they check integrity, not linguistic truth.
The PDF copies live under ignored `artifacts/etruscan-sources/evidence-audit/`; their URLs,
byte counts and SHA-256 hashes are in the source catalog inside `annotations.json`.
The Met page is web-only and has no pinned local snapshot. To rebuild in a clean output
directory, obtain the catalogued PDF bytes at those paths and run the module with `build`.
The builder refuses to overwrite an existing annotation file.

[validation.json](validation.json) records the counts and annotation hash.
[provenance.json](provenance.json) is a **post-audit snapshot**, not a preregistration or a
new evaluation freeze. Earlier experiment freezes remain authoritative for their results.
