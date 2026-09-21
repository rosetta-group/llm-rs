# Fresh-passage lexicon segmentation

The frozen lexicon method improves both corpora, but historical word recovery remains poor.
The declared improvement gate passed; the “solved” gate failed.

**Word error rate (WER):** inserted, deleted, or substituted words divided by reference words.
**Boundary F1:** precision/recall balance for inserted spaces; it is not word accuracy.
**Freeze:** source hashes, code, model, and parameters recorded before challenge preparation.

```text
Fit word counts/transitions on modern ISDT train and Morph-it! surface forms
Tune 18 configurations on 200 non-Dante ISDT development sentences
Freeze the winning method
Generate 24 fresh passages excluding all earlier challenge sentences
Save predictions without reading originals
Grade once; preserve historical failures
```

![Word recovery](../method-benchmark/figures/segmentation.png)

| Corpus | Previous WER | Lexicon WER | Relative reduction | 95% interval for improvement | Boundary F1 |
|---|---:|---:|---:|---|---:|
| Modern | 15.92% | 6.15% | 61.4% | 4.73–14.36 points | 0.980 |
| Historical | 60.93% | 39.90% | 34.5% | 16.00–25.30 points | 0.835 |

1. **Method.** A lexicon trie and beam-Viterbi decoder combine word counts, smoothed
   word transitions, and unknown-word costs. Morph-it! plus training words supplies
   404,346 normalized forms. The selected pseudocount is 1, bigram weight 0.5, unknown
   cost 15+3×length, beam 8, maximum word length 32. Development WER was 6.14%.
2. **Fresh evaluation.** Twelve modern ISDT test passages and twelve historical
   Italian-Old passages, 400–800 letters each. There is no source-sentence overlap with
   the old six-passage benchmark or exact train/development sentences. The modern
   result is 66 errors / 1,074 words; historical is 533 / 1,336. One modern passage
   was exact; no historical passage was exact. Characters were preserved throughout.
3. **Declared gates.** Both corpora exceed 20% relative improvement. Historical WER
   39.90% fails the predeclared requirement of <10% on both corpora. The method improved
   segmentation; it did not fix historical segmentation. The modern prior is a plausible
   mismatch, but this experiment does not isolate vocabulary, spelling, syntax, or genre.
4. **Limits.** The 2,000-draw paired bootstrap samples passages, not new authors.
   Dante passages are fresh but share author/work with the prior benchmark. Gold remains
   evaluator-only in software; this is not an OS security boundary. No tuning followed
   this evaluation. No Voynich final-test text was scored.

Sources: [Morph-it! documentation](https://docs.sslmit.unibo.it/doku.php?id=resources:morph-it),
Marco Baroni and Eros Zanchetta, version 0.48, under the upstream CC BY-SA 2.0 option;
[pinned mirror](https://github.com/giodegas/morphit-lemmatizer/tree/b99d75d774367e4bedc5c5f339dec384777488ff),
[license](https://creativecommons.org/licenses/by-sa/2.0/).
Normalization changes forms; the derived lexicon/model remain local with source attribution.
Language corpus revisions and licenses: [source manifest](../language-sources.json).

Audit: [frozen plan](../METHOD_BENCHMARK_PLAN.md), [lexicon source/hash](../segmentation-sources.json),
[freeze and development grid](freeze.json), [per-case metrics](results.json).
Commands are in the [README](../../README.md#current-benchmark-commands).
