# Decipherment-method benchmark: 21–28 September 2026

The deliverable is a validated recovery benchmark with positive controls and explicit
limits on Voynich inference. A Voynich translation is not supported by current evidence.
Publication suitability is an aim, not a guaranteed outcome.

## Prediction track: closed

No additional BPC sweeps, larger-model comparisons, or longer training runs.
A future prediction experiment needs a named mechanism, a falsifiable contrast,
a fixed budget, and an explanation of which interpretation its result could reject.
A lower BPC alone is not a reason to reopen this track. Existing results stay archived.

## CPU work, bounded to one week

```text
Build a lexicon segmenter and tune only on non-Dante Italian
Freeze code, parameters, source hashes, and split rules
Generate new hidden passages excluding every earlier challenge sentence
Freeze predictions before grading originals
Run the frozen recovery pipeline without a supplied cipher family or codebook
Report failure as failure of the declared method and budget
Start a separate image-annotation association pilot with grouped controls
Prepare an auditable benchmark report, including negatives and limitations
```

The week is a work budget, not a reason to keep computing once checks finish.
CPU only; no cloud services or model downloads. Preserve 20 GiB free disk.
Do not launch unattended new experiments after evaluation exposes the answers.

## Segmentation protocol, fixed before the new evaluation

- Lexicon: pinned Morph-it! 0.48 surface forms; retain author/license attribution.
- Frequencies and word transitions: modern Italian UD ISDT train only.
- Tune on up to 200 ISDT development sentences of 80–300 normalized letters.
  The prior boundary diagnostic used this development corpus; it is development data.
- Candidate model: lexicon Viterbi segmentation with smoothed word frequencies,
  optional word-bigram interpolation, and an explicit unknown-word cost.
- Fixed grid: lexicon pseudocount [0.01, 0.1, 1], bigram weight [0, 0.5, 0.9],
  unknown per-letter cost [2, 3]. Beam width 8; maximum word length 32.
- Select minimum development word error; deterministic tie-breaking. Compare the
  previous calibrated dictionary method (penalty 2) on the same new cases.
- Fresh evaluation: 12 nonoverlapping passages from ISDT test and 12 from Italian-Old,
  each 400–800 normalized letters; exclude all previous Dante challenge sentences
  and any exact training/development sentence. Group uncertainty by passage.
- Primary: pooled word error rate, reported separately for modern and historical text.
  Also boundary precision/recall/F1, exact-passage recovery, character preservation,
  and paired passage-bootstrap improvement. No reranking after seeing test answers.
- Practical target: at least 20% relative word-error reduction over the fixed old
  method on each corpus; do not call the problem solved unless error is below 10%
  on each corpus. These thresholds are declared engineering gates, not universal laws.
- Gold source, document/canto IDs, exclusions, and random keys remain evaluator-only.
  Public IDs are opaque. Separation is auditable software isolation, not a security
  boundary against an adversarial solver with filesystem access.

## Remove cipher hints

Use fresh synthetic challenges with opaque case IDs. The same decoder sees every case;
no family field, codebook, encoder trace, plaintext length, or true key is supplied.
Known target language Italian and the unpaired Italian prior remain declared hints.
Positive controls include simple substitution; Naibbe is the harder condition.
Do not claim a negative establishes that every possible unknown-cipher method must fail.
The first implementation must declare its representational assumptions and fixed CPU cap
before its predictions are graded. A solver limited to letter substitution is a lower
baseline, not an adequate test of arbitrary variable-length encodings.

## Independent text–image evidence: start with annotation audit

Grove/Stolfi's 1998 catalogue provides visual object descriptions and nearby EVA labels,
mostly in pharmaceutical pages. It is a potential pilot source, not validated species
identification or exhaustive botanical-part annotation. Audit duplicate readings,
uncertain labels, physical folio aliases, scribal hand, license, and usable sample size.
Use only existing Voynich training folio groups; keep the final test sealed.

Before scoring associations, freeze endpoints and code. Use held-out folio groups,
length/layout/hand controls, and within-page label permutations. Require a text model
improvement over the controls, an adjusted permutation result, and independent visual
review before calling any association confirmed. Missing feature mentions are unknown,
not automatically negative labels. Report an infeasible or underpowered endpoint honestly.
