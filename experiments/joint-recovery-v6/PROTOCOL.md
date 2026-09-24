# Round six: sealed long-passage Naibbe recovery with context reparse, paired

Declared 2026-09-24, before any round-six passage exists and before the modern source is
downloaded. CPU only. No Voynich text. Licensed by the
[development result](../length-scaling-v3/REPORT.md): mean CER 1.79% at 20,800 letters against
3.25% for the square-root baseline.

## Arms

Each case is decoded once through the shared stages: round four's settings with square-root-scaled
count thresholds and linearly scaled caps for $n$ letters, then joint EM, pruning, joint EM,
lexicon repair and refinement. Then:

- **S, square-root baseline:** polish, then v3 segmentation.
- **R, reparse:** two rounds of full-context reparse (width 128) with key refit by refinement,
  then polish and v3 segmentation.

Everything is fixed at the development values. No setting is changed after any case is decoded.

## Cases

Six Naibbe cases of 20,800–22,000 letters, each with a fresh random letter key and Naibbe seed:

- **Dante (2):** consecutive fresh UD_Italian-Old sentences. Sentences used by any earlier round,
  or identical to ISDT text, are excluded.
- **Compagni (2):** consecutive *Cronica* paragraphs in page order. Paragraphs in the v3 fresh
  test, round five or the language-ID control are excluded.
- **Modern (2):** consecutive sentences of UD Italian ParTUT **train**, downloaded at the ParTUT
  commit already pinned (`6ae975a…`). No model has used any ParTUT split; test and dev are
  released and are not used here.

A passage is rejected if it shares a 20-word sequence with fitting or development text or with
any released passage. The solver sees ciphertext only. References stay evaluator-only until both
arms are saved.

## Endpoints, fixed now

- **Primary:** pooled CER of R at least 0.5 points below S over the six cases.
- **Gate:** 1% CER and 10% WER per case for R, reported.
- **Secondary:** per-source CER and WER, parse agreement, reparse changes and cap hits.
