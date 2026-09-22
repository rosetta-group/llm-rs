# Codebook-free Naibbe recovery, round three: fresh evaluation

Frozen method commit: `2221e8312b9e8afc3ce53970bbf20f2c64b28684` ([freeze](freeze.json), [protocol](../joint-development-v3/PROTOCOL.md)).
Four fresh passages of 5,211–5,469 letters, none overlapping any earlier challenge, decoded from ciphertext only; references opened after predictions were saved.

**The polish buys about one point and the gate still fails.** Character error is 8.8% on modern
Italian and 10.5% on Dante, against 9.5% and 10.3% in round two. Word error is 54% and 56%. The
only change was a word-level polish after the round-two decoder; its paired effect is 0.5–1.5
points on every case. The development attribution holds on fresh text: letters inside correctly
parsed tokens are 99–99.5% right, and the roughly 10% of mis-parsed tokens carry the rest.

**CER / WER / segmentation agreement / gate:** as in [round one](../joint-recovery/REPORT.md).
**Paired comparison:** the same decoder output graded before and after the polish, so the
difference is the polish alone.

## What was done

- Froze the round-three decoder (round two pipeline plus lexical polish at weight 1), committed,
  prepared two ISDT-test and two Dante passages excluding all source IDs used by rounds one and two.
- Decoded the four cases in 20 CPU minutes total; graded both the polished and the pre-polish
  letters once; measured letter error inside correctly parsed tokens; archived all records.

## Why

Development showed that the character prior tolerates a wrong letter when it leaves a
pronounceable string. A lexicon does not, so re-scoring shortlisted letters against the frozen
segmenter's word cost should fix letters inside correctly parsed tokens. The same development
run predicted the ceiling: the polish cannot touch a wrong parse, and wrong parses hold about
12% of the letters. This run checks both predictions on unseen text.

## Results

| Case | Letters | Words | CER polished | CER before polish | WER polished | WER before | Segmentation agreement | Letter error inside correct parses | Letters in mis-parsed tokens | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| modern | 5,469 | 963 | **8.2%** | 8.7% | 53.9% | 56.2% | 91.4% | 0.5% | 9.9% | fail |
| modern | 5,234 | 926 | 9.3% | 9.9% | 55.0% | 58.4% | 89.7% | 0.6% | 11.9% | fail |
| historical | 5,211 | 1,223 | 10.1% | 10.7% | 58.1% | 62.3% | 89.7% | 1.0% | 11.9% | fail |
| historical | 5,355 | 1,271 | 10.8% | 12.3% | 53.6% | 59.2% | 88.5% | 1.4% | 13.9% | fail |

| Dataset | Round three CER | Round two CER | Round one CER | Round three WER | Round two WER |
|---|---:|---:|---:|---:|---:|
| modern | **8.8%** | 9.5% | 12.5% | 54.4% | 54.1% |
| historical (Dante) | **10.5%** | 10.3% | 33.5% | 55.8% | 60.2% |

Before the polish this run's decoder gave 9.3% modern and 11.5% Dante, so the round-two decoder
itself landed 0.2 and 1.2 points worse on these passages than on round two's; the polish then
recovered 0.5 and 1.0 points. No case hit a cap. Development predicted 8.6–9.6%; the fresh cases
land at 8.2–10.8%.

1. **The polish helps every case, paired.** 0.5, 0.6, 0.6 and 1.5 points of character error and
   2–6 points of word error, all in the same direction. Fifty to eighty-four letter changes per
   passage.
2. **The attribution transfers.** Letter error inside correctly parsed tokens is 0.5–1.4%, in the
   development band of 0.5–1.0%; mis-parsed tokens hold 10–14% of the letters and account for
   nearly all the remaining error. The letter mapping is solved where the parse is right.
3. **Letters are still not words.** Sample, modern, best case (true above, recovered below):

> itorepossaadempierelobbliga**zi**onenelleobbliga**zi**oniche**ha**nnoperoggettounasommadidanaro
> itorepossaadempierelobbliga**cui**onenelleobbliga**ti**oniche**gra**nnoperoggettounasommadidanaro

## What this establishes and what it does not

- Three frozen rounds have taken codebook-free Naibbe from over 300% to about 9–11% character
  error on fresh 5,200-letter passages. The gate (CER ≤ 1%, WER ≤ 10%) is not met, and the
  polish is the last cheap gain: the remaining error sits in parses, not letters.
- The next gain, if any, needs a segmentation model that sees more than one token when choosing
  a split, or a different candidate-piece rule. Development estimated about 4% of tokens are
  ambiguous even with the true lexicon; the other 6% are the target. No such model exists here.
- Four passages; no significance claim. Procedural blinding on one machine. These passages are
  now disclosed; exclude their source IDs from future fresh tests.
- The Voynich mechanism test stays closed. Nothing here touches Voynich text or the final test.

## Records and reproduction

- [Frozen settings and hashes](freeze.json); [all per-case metrics](results.json), including the
  pre-polish grades and the attribution; protocol and development record in `experiments/joint-development-v3/`.
- `evaluated-records.tar.gz` holds the public challenge, predictions (polished and pre-polish
  letters, parses), challenge hashes and evaluator answers with seeds and traces, released only after grading.

```sh
.venv/bin/python -m experiments.verse_sources restore          # unpack the pinned Canzoniere pages
.venv/bin/python -m experiments.joint_recovery_v3 verify
.venv/bin/python -m experiments.joint_recovery_v3 evaluate     # regrades the archived predictions
```

The prior file is the round-two prose+verse prior under `artifacts/verse-prior/` (about 50 MB),
hashed in `freeze.json` and refitted deterministically by `experiments.joint_development_v2.fit_priors`.
The segmenter used by the polish is the round-one frozen lexicon model, untouched.
CPU only; no downloads in this round; no paid compute.
