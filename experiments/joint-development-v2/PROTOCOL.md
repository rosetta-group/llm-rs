# Codebook-free Naibbe recovery, round two: usage pruning and a verse-aware prior

Status: frozen by `experiments/joint-recovery-v2/freeze.json` at the commit recorded there; the
fresh evaluation runs under `experiments/joint_recovery_v2.py`. Round one
([protocol](../joint-development/PROTOCOL.md), [fresh report](../joint-recovery/REPORT.md)) is unchanged.
CPU only; no cloud, no model downloads, no Voynich text, final test sealed.

## What changes and why

Round one ended at 12.5% character error on modern Italian and 33.5% on Dante, with
segmentation agreement of 85–91% and the gate (CER ≤ 1%, WER ≤ 10%) failed. Two levers
were identified there and are the only changes here:

1. **Usage pruning of the candidate lexicon.** After the first joint EM, candidate pieces whose
   expected usage under the parse posteriors is below `prune_usage` are dropped, parses are
   rebuilt, and EM runs again. Tokens left without a split keep their whole-token fallback.
   Development: 12.2% → 9.4% (modern) and 14.2% → 10.4% (historical prose) at 5,200 letters;
   a second pruning round adds nothing, so exactly one is run.
2. **A prior with non-Dante verse.** Petrarca's Canzoniere (366 poems, pinned from Wikisource in
   `experiments/verse-prior/sources.json`) is added to the modern and prose training text at
   weight `verse_weight`; every fifth poem is development and never enters fitting. Dante is
   never used for fitting or tuning. The prior chosen for the freeze is named in `freeze.json`;
   the development comparison of both priors on prose, modern and Petrarca development text is
   in `results.json`.

Everything else is inherited from round one: the token class assumption (one or two pieces
per token, one letter per piece, role-specific emissions), the frequency threshold `minimum`,
EM restarts and iterations, refinement budget, and the passage length of 5,200–6,000 letters.

## Method

```text
pieces = candidate_pieces(tokens, minimum)
first  = joint EM over latent parses with role emissions
kept   = pieces with expected usage >= prune_usage under first's posteriors
second = joint EM on kept
units  = role-tagged pieces of second's decoded segmentation
key    = majority letter per unit; refine by iterated local search under the description length
report the refined letters
```

## Fresh evaluation

- Two modern ISDT test passages and two Dante passages, 5,200–6,000 letters, excluding every
  source ID used by any earlier challenge including round one, and any training or development
  sentence. Passages concatenate consecutive fresh sentences.
- Pinned Naibbe encoder, hidden letter permutation, system-random seed, verified round trip.
  Public files hold opaque IDs and ciphertext; answers, keys and traces are evaluator-only.
- Grade CER, WER after the round-one frozen segmenter, and segmentation agreement from the trace.
- Gate: CER ≤ 1% and WER ≤ 10% on every case. Secondary: change from round one per dataset.
- Four passages are the units; no significance claim. Procedural blinding on one machine.

## Limits declared in advance

- The word segmenter is the round-one model; it was not retrained on verse, so WER on Dante
  remains limited by it as well as by letter errors.
- Petrarca is one author of verse; it is a step toward, not a solution of, the historical-verse mismatch.
- Nothing here concerns Voynich text. The reserved Voynich mechanism test stays closed.
