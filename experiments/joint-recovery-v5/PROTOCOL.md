# Round five: the round-four Naibbe decoder with the v3 segmenter, paired

Declared 2026-09-23, before any round-five passage exists. CPU only, no download,
no Voynich text. The round-four and v3 modules stay frozen. This is the follow-up
licensed by the [v3 fresh test](../word-segmentation-v3-fresh/REPORT.md), and it changes
only the segmenter.

## Arms

Each case is decoded once through every shared stage of round four, with its frozen
settings: joint EM, usage pruning, joint EM, lexicon repair and refinement. That gives
one refined key per case. Then:

- **A, round four exactly.** Polish with the round-four segmenter cost; segment with the round-four segmenter.
- **B, segmentation only.** A's letters, segmented with the frozen v3 segmenter.
- **C, v3 throughout.** Polish from the same refined key with the v3 segmenter cost
  (unknown words cost $-\log p_{\text{unk}} - \log P_{\text{spell}}$). Segment with v3.

Polish settings, caps and the character prior are the same in A and C. Any cap hit is reported.

## Cases

There are eight Naibbe cases, each with a fresh random letter key and a fresh Naibbe
encryption seed, as in round four.

- **Four Dante passages.** Consecutive fresh UD_Italian-Old sentences in corpus order,
  5,200–6,000 letters. Sentences used by any earlier round are excluded, and so are
  sentences identical to ISDT text.
- **Four Compagni passages.** Built by the v2 packer from *Cronica* paragraphs that were
  not in the released v3 fresh passages, in page order. A passage is rejected if it shares
  a 20-word sequence with fitting or development text or with any released passage. Nothing
  in the decoder or the segmenter was tuned on Compagni. The v3 test only graded four other
  passages of it.

There is no modern half: ISDT test is exhausted, and ParTUT test and dev are released. The solver
sees the ciphertext only. References and keys stay evaluator-only until all three arms are saved.

## Endpoints, fixed now

- **Primary, C against A, pooled over the eight cases.** WER must fall by at least 3 points,
  and CER must not rise by more than 0.5 points. The WER measure is the same as round four:
  word edit distance of the segmented recovery against the reference.
- **Secondary.** B against A isolates the segmenter; C against B isolates the polish cost. All
  are reported per source.
- **Recovery gate.** 1% CER and 10% WER per case, reported. Passing the primary endpoint does
  not open any Voynich test.

No setting is changed after any case is decoded, and no case is dropped or rerun except to
resume a checkpoint.
