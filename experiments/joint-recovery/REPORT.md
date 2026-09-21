# Codebook-free Naibbe recovery: fresh evaluation

Frozen method commit: `64d37f4d083b0ca477621fbccba4a0ed76f0fb12` ([freeze](freeze.json), [protocol](../joint-development/PROTOCOL.md)).
Four fresh passages of 5,201–5,529 letters, ciphertext-only decoding, references opened after predictions were saved.

**The gate fails.** Character error is 12.5% on modern Italian and 33.5% on Dante; word error 58% and 87%.
The previous frozen result on this cipher, with 1,200–1,800-letter passages, was 300%+ character error.
This is partial recovery of a codebook-free verbose homophonic cipher, not a solved one, and not a Voynich result.

**CER:** character edits divided by reference letters. **WER:** word edits after the frozen segmenter, divided by reference words.
**Segmentation agreement:** share of tokens whose decoded parse matches the encoder trace, measured evaluator-side only.
**Gate:** CER ≤ 1% and WER ≤ 10% on every case.

## What was done

- Froze the joint segmentation-and-EM decoder and its development settings, then committed.
- Prepared two ISDT-test and two Dante passages excluding every earlier source ID and any training/development sentence.
- Encoded each with the pinned Naibbe encoder under a hidden letter permutation; verified round trips.
- Decoded from ciphertext alone in 14.5 CPU minutes; graded once; archived all records.

## Why

Development showed this cipher class is unrecoverable below about 2,600 letters and partly recoverable at 5,200,
so the fresh protocol changed passage length. The question was whether the development numbers hold on unseen passages.

## Results

| Case | Letters | Words | CER | WER | Segmentation agreement | Recovered letters | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| modern | 5,201 | 1,051 | **9.5%** | 48.9% | 90.9% | 5,111 | fail |
| modern | 5,208 | 1,003 | **15.5%** | 67.6% | 85.1% | 5,046 | fail |
| historical | 5,217 | 1,260 | **30.1%** | 85.6% | 85.2% | 5,059 | fail |
| historical | 5,529 | 1,443 | **36.7%** | 88.1% | 89.1% | 5,348 | fail |

| Dataset | Pooled CER | Pooled WER | Previous frozen method (1,200–1,800 letters) |
|---|---:|---:|---:|
| modern | 12.5% | 58.0% | 318.8% |
| historical | 33.5% | 87.0% | 303.7% |

No run hit a cap. Joint EM restart likelihoods lie within 0.3% of each other in every case.

1. **Modern text matches development.** Development gave 12.2% at 5,200 letters; fresh cases give 9.5% and 15.5%.
2. **Historical text is worse than development.** Development historical text was Novellino/Decameron prose (14.0%); the
   fresh cases are Dante verse at 30–37%. The prior is trained on prose; the segmentation agreement is as high as on
   modern text, so the loss is in the letter mapping, consistent with the prior mismatch seen in earlier reports.
3. **Words are far from recovered.** Even the best case has 49% word error: letter errors break the lexicon segmenter,
   and the segmenter itself was 6% WER on exact modern letters.
4. **Recovered text is readable in places and wrong in others.** Modern, best case:

> quisolodueanni**ugadomingvailcopriguoco**impostoda**holdathsraelianzeenite**alprimo**cibo**nellastoriadi**giaza**
> quisolodueanni**fadominavailcoprifuoco**impostoda**isoldatiisraelianivenite**alprimo**circo**nellastoriadi**gaza**

## What this establishes and what it does not

- A decoder given only ciphertext, the language, the alphabet and a declared token class recovers most letters of a
  5,000-letter Naibbe passage in modern Italian. The declared gate is not met; the method is not validated for the
  benchmark's stated purpose.
- The historical gap is a prior problem as much as a cipher problem. A verse-aware or word-level prior is the obvious
  next lever; it must be tuned on non-Dante material and frozen again with new passages.
- Four passages; no significance claim. Procedural blinding on one machine, not an independent evaluator.
- The Voynich mechanism test stays closed. Nothing here touches Voynich text or the final test.

## Records and reproduction

- [Frozen settings and hashes](freeze.json); [all per-case metrics](results.json); protocol and development record in
  `experiments/joint-development/`.
- `evaluated-records.tar.gz` holds the public challenge, predictions, challenge hashes and the evaluator answers with
  seeds and traces, released only after grading. These four passages are now disclosed; exclude their source IDs
  from every future fresh test.

```sh
.venv/bin/python -m experiments.joint_recovery verify
.venv/bin/python -m experiments.joint_recovery evaluate    # regrades the archived predictions
```

A new replication needs its own `artifacts/joint-recovery/` and will draw new random keys. CPU only; no downloads; no paid compute.
