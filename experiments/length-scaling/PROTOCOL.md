# Length scaling of codebook-free Naibbe recovery (development only)

Declared 2026-09-23, before any run. CPU only. No download, no sealed passage, no Voynich text.

## Question

Does round four's decoder recover letters better from longer ciphertext? With true splits,
error fell from 77% at 1,300 letters to 4.6% at 2,600 and 3.2% at 5,200, and it hadn't levelled
off. The Voynich text is long, so passage length is a realistic lever, and it needs no new method.

## Design

- **Texts:** the three existing development texts: historical prose, modern ISDT and Petrarca verse.
- **Lengths:** nested prefixes of 5,200, 10,400 and 20,800 letters. Each text keeps one fixed key
  and Naibbe seed (7000) at every length, so the lengths are paired.
- **Decoder:** round four's frozen settings and stages. Joint EM, pruning, joint EM, lexicon
  repair, refinement, then polish with the round-four segmenter cost. The prose+verse prior is
  unchanged. The whole-case cap scales with length: $3600 \times n / 5200$ seconds. Stage caps
  are derived as in round four.
- **Recorded:** refined and polished CER, parse agreement, letter error inside correct parses,
  the share of letters in mis-parsed tokens, lexicon true/spurious/missing counts against the
  oracle-only trace, cap hits and seconds. Also WER with the v3 segmenter.

## Decision rule, fixed now

Take mean polished CER over the three texts. If the 20,800-letter mean is at most half the
5,200-letter mean, declare a sealed long-passage round. Otherwise record the length curve as a
negative for this route, and move to parse fixes (context reparse, single-letter protection)
at 5,200 letters. This is development evidence. It does not change any sealed result.
