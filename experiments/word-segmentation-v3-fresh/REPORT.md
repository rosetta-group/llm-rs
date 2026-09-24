# Fresh test of the v3 segmenter: transfer threshold passed, word gate not met

Evaluated 2026-09-23. Perfect letters, CPU only, about 40 seconds. No cipher decoding,
Voynich text or paid compute. Method frozen in `85531e7` ([freeze](../word-segmentation-v3/freeze.json)).
Sources pinned in `bf5dc9a` before any passage was built ([notes](NOTES.md)).

## What was compared

The v3 segmenter keeps the frozen known-word scorer. It replaces the flat unknown-word
cost $15 + 3\ell$ with $-\log p_{\text{unk}} - \log P_{\text{spell}}(w)$, using an
order-5 letter model and the elision join ([protocol](../word-segmentation-v3/PROTOCOL.md)).
The baseline is the round-four segmenter. Both methods segmented the same dense
letters. Predictions were saved before the references were opened.

- **Historical:** four passages from Dino Compagni's *Cronica*, a new author, taken in page order.
- **Modern:** four passages from UD Italian ParTUT test then dev. This is a recorded deviation: the test split alone was too short.
- Passages are 5,222–5,887 letters each. Any passage sharing a 20-word sequence with fitting,
  development or previously released text was rejected; 14 historical and 15 modern source rows were excluded.

## Results

| Source | Words | Baseline WER | v3 WER | Change | Extra spaces | Missing spaces | Passages ≤ 10% |
|---|---:|---:|---:|---:|---|---|---:|
| Compagni (historical) | 4,702 | 24.01% | **17.46%** | −6.55 | 626 → 391 | 132 → 124 | 0 → 0 |
| ParTUT (modern) | 3,802 | 7.23% | **6.05%** | −1.18 | 145 → 117 | 20 → 26 | 3 → 3 |

Per passage, historical: 21.56 → 15.64, 22.95 → 16.75, 24.00 → 17.59, 27.82 → 20.07.
Modern: 5.01 → 4.03, 4.91 → 4.27, 3.35 → 3.14, 15.21 → 12.39. All eight passages improve.

- **Transfer threshold: passed.** Historical pooled WER must fall by at least 3 points and
  modern must not worsen by more than 1. Historical fell 6.55 points and modern fell 1.18.
- **Word gate: not met.** No Compagni passage is at or below 10% WER.
- **Transfer is partial.** On development text the gain was 6.6 points from a 15.3% base.
  On Compagni it is 6.55 points from a 24.0% base, so the relative reduction is smaller:
  27% here against 43% on development prose.

## What remains ([error-attribution.json](error-attribution.json), post-grading, descriptive)

Missing forms still dominate. In Compagni, 344 of the 391 remaining extra spaces fall inside
words absent from the lexicon, and 286 of the 411 missing-form tokens are still split,
down from 381. Many are forms that repeat through the text: the name `giano` is split 27
times, the archaic `erono` 15 times, and `fusse` 9 times, as are the verb endings
in `-orono` / `-ono`. The spelling model is fitted on lexicon types, and about 377k of
those are modern Morph-it forms. So it scores these archaic spellings as unlikely words.

Missing spaces are few and stable (132 → 124). In ParTUT, some errors come from source
artefacts such as `attribuzionecondividi`, a Wikipedia footer glued into the text.

## Implication

This is the first word-segmentation candidate to pass a declared fresh-transfer test. It
licenses the protocol's only follow-up: a separately declared, paired Naibbe comparison
that keeps the round-four decoder and changes only the segmenter. That comparison costs
real CPU time, since each cipher stage runs for minutes.

Segmentation-only candidates the diagnosis points to, each needing its own declared
protocol:
1. A spelling model weighted towards historical training types.
2. Document-level adaptation, so a form that recurs in a passage becomes cheaper.

Compagni and ParTUT (test and dev) are now released and are excluded from future hidden tests.

## Post-grading caveat, 2026-09-24

A later audit ([partut-overlap-audit.json](../partut-overlap-audit.json)) found that 54% of the
ParTUT modern letters here are sentences also in UD_Italian-ISDT: 33.0% in ISDT test and 21.3% in ISDT
dev. None are in ISDT train, which fits the models, so this is not training leakage. But ISDT dev was
the modern development stream, and ISDT test was released by earlier rounds. So the modern half is not
fully fresh. The transfer decision rests on the historical Compagni result (−6.55 points), which is
unaffected. The modern condition (no worse than +1 point) passed with a 1.18-point improvement.

## Records and reproduction

`sources.json`, `extraction.json`, `NOTES.md`, `results.json`, `evaluated-records.json`,
`error-attribution.json`, and `fresh-sources.tar.gz` with its `archive.json`.

```sh
.venv/bin/python -m experiments.word_segmentation_v3_fresh restore
.venv/bin/python -m experiments.word_segmentation_v3_fresh verify
```

`verify` checks the archive, sources, sealed challenge and freeze order, and regrades all
eight cases. Licences: Compagni transcription CC BY-SA (Wikisource); ParTUT CC BY-NC-SA 4.0.
Its README and licence are in the archive.
