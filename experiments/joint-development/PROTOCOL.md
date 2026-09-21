# Codebook-free Naibbe recovery by joint segmentation and EM

Status: frozen by `experiments/joint-recovery/freeze.json` at the commit recorded there; the fresh evaluation runs under `experiments/joint_recovery.py`.
The development record is [REPORT.md](REPORT.md) with numbers in `results.json`.
CPU only; no cloud, no model downloads, no Voynich text, final test sealed.

## Question and gate

Can a fixed decoder recover Naibbe-encrypted Italian when it receives only the
space-delimited ciphertext and an unpaired Italian character prior: no codebook, no
family label, no key, no encoder trace, no plaintext length?

```text
Fix the prior, candidate-piece rule, EM settings, refinement budget and seeds
Commit code, protocol and development results
Generate fresh passages of at least 5,200 letters, excluding every earlier source ID
Save ciphertext-only predictions before opening evaluator references
Grade letters (CER), words (WER after the frozen segmenter) and segmentation agreement
Gate: CER <= 1% and WER <= 10% on every case; report the shortfall otherwise
```

The earlier gate stays. Development reaches 12–15% CER at 5,200 letters, so the gate is expected to fail.
The declared secondary result is the reduction from the previous 300%+ CER, per case,
with segmentation agreement measured evaluator-side from the encoder trace.

## Declared representational assumptions

1. Each ciphertext token is a concatenation of **one or two pieces**, and each piece stands
   for **one plaintext letter**. This matches Naibbe's unigram and prefix+suffix tokens; it
   is an assumption about the cipher class, declared here, not learned.
2. A piece's letter may depend on its **role**: whole token, first part or second part.
   Emission tables are separate per role.
3. Candidate pieces are strings that occur at least `minimum` times as a whole token, a
   proper prefix or a proper suffix of tokens in the ciphertext. A token keeps a whole-token
   parse only if it is itself a candidate piece or has no valid split; otherwise rare whole
   tokens act as free wildcards and every token degenerates to one letter.
4. The plaintext prior is the frozen order-5 interpolated character model from
   `artifacts/standard-decipherment/prior.npz`; the joint EM uses its trigram marginal.

The artificial variable-homophonic control (atomic four-letter tokens for one or two
letters) is **out of class** for this method: its tokens share no sub-token statistics.
It remains a documented negative from the standard-method report.

## Method

```text
pieces = candidate_pieces(tokens, minimum)
joint EM: trigram HMM over letters, latent parse per token, role emissions; R restarts, I iterations
decode: best parse per token, argmax letters
units = role-tagged pieces of the decoded segmentation
key = majority letter per unit; refine by iterated local search under the description length
warm round: one more joint EM initialized from the refined key; decode; refine again
report the last refined letters
```

Settings fixed at freeze time (development values): `minimum` 6, joint restarts 4,
iterations 60, refinement 30 kicks of 6 units with a 300 s cap, one warm round of 30
iterations, seed 3. Per-case CPU cap 40 minutes; capped runs are reported as capped.
Greedy re-segmentation under a fixed key was tried and rejected in development: it lowered
segmentation agreement from 88% to 80% (modern) and 85% to 75% (historical) and raised error.

## Fresh evaluation

- Two modern ISDT test passages and two Dante passages, each **5,200–6,000 letters**.
  Exclude source IDs from every earlier decipherment, segmentation, codebook-free and
  standard-method challenge, and exact overlap with any training or development sentence.
- Encode each passage with the pinned Naibbe encoder under a hidden global letter
  permutation and a system-random seed. Verify the round trip. Public files hold opaque IDs
  and ciphertext only; answers, keys and traces stay evaluator-only.
- Word error uses the frozen segmenter from the standard-method study on the recovered letters.
- Four passages are the independent units. No significance claim.
- Development text (Novellino/Decameron dev tales, ISDT dev) must not appear in evaluation.

## Why passage length changed

Development shows the mapping search is length-limited: with the true segmentation
supplied, EM fails at 1,300 letters (77% CER with 24 restarts) and recovers at 2,600
letters (4.6% after refinement) and 5,200 letters (3.2%). About 300 to 400 piece types
against 1,300 letters is near the unicity distance of this cipher class under a character
prior. The earlier benchmarks used 1,200–1,800 letters; that length, not only the decoder,
made codebook-free Naibbe unrecoverable. The manuscript itself is far longer.
