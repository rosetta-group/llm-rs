# Word segmentation v3 groundwork: rubric-free extraction and extra-space diagnosis

Development diagnosis, 2026-09-23. CPU only, under ten seconds. No fresh text,
downloads, cipher decoding, Voynich text or setting selection. The frozen v2 files
are unchanged. Code: `experiments/word_segmentation_v3.py`.

## 1. Rubric-free extractor

`extract_book_v3` removes `<p>` paragraphs whose first non-empty line is a bare Roman
numeral (Wikisource chapter rubrics), then applies the unchanged v2 extractor.
On the released Villani Book I source ([check](extraction-check.json)):

| | v2 | v3 |
|---|---:|---:|
| Extracted paragraphs | 73 | 37 |
| Rubric paragraphs removed | — | 37 (one was already below the 8-word minimum) |
| Body paragraphs changed or reordered | — | 0 |

The 36 dropped paragraphs equal the post-grading audit's rubric count. Rubrics have
at most 24 words; the shortest body paragraph has 60. The rule is specific to this
Wikisource layout; a new source needs its own check before any passage is built.

## 2. Where the extra spaces come from

Input: the saved v2 baseline predictions (`dev-weight-0.json`) on the three
5,200-letter development streams, with perfect letters. Each extra space lies inside
exactly one reference word; each missing space lies inside exactly one predicted
token. "Missing form" means the reference word is absent from the segmenter's
407,341-form lexicon.

| Stream | WER | Extra spaces | …inside missing forms | Missing spaces | Missing forms split |
|---|---:|---:|---:|---:|---:|
| Historical prose | 15.27% | 92 | 83 (90%) | 22 | 57 / 64 |
| Modern ISDT | 6.89% | 31 | 27 (87%) | 16 | 19 / 22 |
| Petrarca verse | 28.49% | 161 | 133 (83%) | 98 | 103 / 123 |

Released records, descriptive only and never for selection: Villani 670 of 762 extra
spaces (88%) fall inside missing forms; VIT 94 of 134 (70%).

**Oracle ceiling (not a method).** Adding each stream's own missing forms to the
lexicon, with no count and no other change, gives WER 2.29% historical, 2.76% modern,
6.89% verse (6.73% with true counts). The scorer and search already handle known words;
the missing forms cause most of the error.

**Mechanism.** An unknown word costs a flat $15 + 3\ell$ nats. Any split into
known pieces is cheaper. Unigram approximation, $\alpha = 1$:

| Reference | Unknown cost | Predicted split | Split cost |
|---|---:|---|---:|
| melano | 33.0 | me la no | 19.4 |
| merlino | 36.0 | merli no | 21.9 |
| linnocenti | 45.0 | l innocenti | 18.3 |
| favolatore | 45.0 | favola tor e | 27.9 |

So the segmenter rarely emits an unknown word: 7 of 1,203 historical tokens.
It splits 89% of missing historical forms instead. The missing forms are mostly archaic spellings and names
(`melano`, `merlino`, `burchiello`). A smaller group comes from elisions that
`normalize` merges when it drops the apostrophe: 6 of 64 historical, 7 of 22
modern and 29 of 123 verse missing forms are an elided prefix plus a known word
(`linnocenti`, `lincendio`, `sagghiaccia`).

## Implication

The next candidate should change the unknown-word model, not the vocabulary weights.
One example: score unknown words with a letter n-gram spelling model trained on
training word types, plus one unknown-rate constant, replacing the flat $15 + 3\ell$.
Elision could be a second declared component. Declare the candidates and the selection
rule before scoring the development streams, and set any constant on training text
only. A fresh test then needs a new historical author and a new modern corpus. Villani
and VIT are released and excluded.

Reproduce:

```sh
.venv/bin/python -m experiments.word_segmentation_v3 check
.venv/bin/python -m experiments.word_segmentation_v3 diagnose
```
