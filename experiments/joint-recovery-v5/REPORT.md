# Round five: the v3 segmenter in Naibbe recovery. Primary endpoint passed; the gain is segmentation only

Evaluated 2026-09-23. Eight sealed cases, ciphertext only, CPU, 71 minutes. Protocol
committed before any passage existed ([PROTOCOL.md](PROTOCOL.md)). The round-four decoder
and settings are unchanged. No Voynich text.

## Arms

Every case ran round four's shared stages once: joint EM, pruning, joint EM, lexicon repair
and refinement. Then:

- **A**: round four exactly.
- **B**: A's letters, segmented with the frozen v3 segmenter.
- **C**: polish with the v3 cost from the same refined key, then v3 segmentation.

## Results ([results.json](results.json))

| Pooled | A: CER | A: WER | B: CER | B: WER | C: CER | C: WER |
|---|---:|---:|---:|---:|---:|---:|
| All 8 cases | 5.63% | 45.81% | 5.63% | **41.46%** | 5.81% | 41.69% |
| Dante (4) | 5.74% | 45.50% | 5.74% | 40.49% | 5.91% | 40.82% |
| Compagni (4) | 5.52% | 46.15% | 5.52% | 42.53% | 5.71% | 42.65% |

- **Primary endpoint (C against A): passed.** WER fell 4.12 points; the threshold was 3.
  CER rose 0.18 points; the limit was 0.5.
- **The segmenter is the source (B against A).** With identical letters, v3 segmentation
  lowers WER by 4.35 points, and it improves all 8 cases.
- **The v3 polish cost does not help (C against B).** C has worse CER than A in 5 of 8
  cases, and WER within 0.2 points of B. The v3 cost accepts a few more key changes (24–41
  against 19–41). Making unknown words cheaper also makes some wrong letters cheaper.
- **Recovery gate: not met.** No case reaches 1% CER and 10% WER. No cap was hit.

Round four's 5.7% CER is reproduced on new text: 5.63% here, against 5.7% on its own sealed
Dante passages. So the decoder is stable across fresh draws and across the prose/verse split.

## Implication

Adopt the v3 segmenter for final segmentation (arm B). Keep round four's polish cost. B was
a declared secondary arm, not the primary, so this choice rests on the paired secondary
comparison. It needs no further confirmatory test, because B's letters are identical to A's by construction.

Word error stays above 40% because of letter errors. At 5.6% CER a five-letter word
has about a one-in-four chance of a wrong letter ($1 - 0.944^5 \approx 0.25$). With perfect letters, v3 reaches 17–20% WER on the same kind of
text (see the [v3 fresh test](../word-segmentation-v3-fresh/REPORT.md)). The next gain has
to come from letter recovery, which means the candidate lexicon (oracle 0.5% CER in round
four). The segmenter has already delivered most of its share.

Dante sentences and Compagni paragraphs used here are released and are excluded from future tests.
