# Word segmentation v3: rubric-free extraction, extra-space diagnosis, unknown-word model

Development work, 2026-09-23. CPU only. No fresh text,
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

## 3. Development selection of a letter-level unknown-word model

Declared in [PROTOCOL.md](PROTOCOL.md) and committed (`5d58648`) before scoring. Each
candidate replaces the flat cost with $-\log p_{\text{unk}} - \log P_{\text{spell}}(w)$.
The training-only Good–Turing rate is $p_{\text{unk}} = 2.34\%$, about 3.8 nats.
Perfect letters, same three streams, WER ([development.json](development.json)):

| Candidate | Historical prose | Petrarca verse | Modern ISDT | Extra / missing spaces (prose) |
|---|---:|---:|---:|---:|
| Baseline, flat $15+3\ell$ | 15.27% | 28.49% | 6.89% | 92 / 22 |
| Order 3 | 10.86% | 24.58% | 6.30% | 40 / 36 |
| Order 3 + elision | 10.50% | 23.92% | 5.71% | 38 / 36 |
| Order 5 | 9.27% | 22.09% | 6.40% | 43 / 24 |
| **Order 5 + elision (selected)** | **8.65%** | **20.68%** | **5.61%** | 39 / 24 |

All four candidates are eligible. The selected one improves the historical mean by 7.2
points (the rule needs 3), and no stream worsens. Extra spaces in prose fall from 92 to
39. Missing spaces stay about level at 24. Verse remains far from the 10% gate: 100 extra
and 76 missing spaces. All candidates together took about 3 seconds of segmentation;
fitting took 24 seconds and 0.5 GB of memory. The selection is frozen in
[freeze.json](freeze.json). `verify` refits the rate and the spelling model and
checks their digests.

These are development numbers on the streams that motivated the diagnosis. They are
not a transfer result. The fresh test in the protocol decides whether the gain holds
on an unseen historical author.

## Implication

The diagnosis pointed to the unknown-word model, and section 3 confirms it on development
text. The next step is the protocol's fresh test: a new historical prose author and a new
modern corpus, named and pinned before download. Villani and VIT are released and excluded.

Reproduce:

```sh
.venv/bin/python -m experiments.word_segmentation_v3 check
.venv/bin/python -m experiments.word_segmentation_v3 diagnose
.venv/bin/python -m experiments.word_segmentation_v3 verify
```

`develop` and `freeze` refuse to overwrite their records.
