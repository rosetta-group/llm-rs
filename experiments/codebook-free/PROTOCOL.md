# Codebook-free recovery: fixed CPU baseline

Frozen before generating or grading this challenge, 2026-09-21.

## Question

Can one decoder select a useful representation and recover new Italian passages
without receiving a cipher-family label, structural codebook, key, encoder trace,
plaintext length, or plaintext word boundaries? Italian and its normalized alphabet
remain known. This is a bounded baseline, not unrestricted cipher discovery.

## Cases

Four fresh passages: two modern ISDT test and two historical Italian-Old passages,
1,200–1,800 letters each. Exclude all source sentences in both previous recovery and
segmentation challenges, and exact training/development sentences. Each passage has
three encodings: simple substitution, an artificial homophonic variable-length
control, and published Naibbe with a hidden global letter permutation. Use one new
random key per case; opaque IDs and ciphertext are the only public case fields.

The artificial control has one or two output letters per space-delimited cipher
unit and two random cipher spellings per plaintext chunk. It tests the decoder's
broader representation, not just monoalphabetic recovery. Cipher-unit spaces are
observable; they are never original word boundaries. Paired cases share a passage,
so report four independent passages, not twelve independent texts.

## Decoder and fixed budget

Try both observed characters (discarding spaces) and space-delimited units. For each
representation, try a bijection if its inventory fits the Italian alphabet, and a
many-to-one mapping with one or two output letters per unit. This latter model can
represent homophones and variable expansion. It cannot represent arbitrary
context-dependent mappings, unknown word languages, or every transposition scheme.

Use only the unpaired ISDT training character prior. Select candidates by mean Italian
conditional four-gram log score minus the KL divergence of their unigram distribution
from the training prior. This is a fixed heuristic, not calibrated posterior evidence.
For bijections use six annealing restarts of 12,000 proposals; for variable mappings
use six restarts of 12,000 proposals, seeded 42 plus public case index. Initial and
replacement chunks are sampled from training single-letter and digram counts with
equal prior probability of length one or two. No evaluation-driven hyperparameter
changes. Restore spaces with the already frozen lexicon segmenter.

CPU only. Maximum 120 seconds per candidate search; stop that search and record its
budget status if reached. Maximum four candidate searches per case. A cap-hit result
is budget-limited, not a converged solver. No cloud costs. Save predictions and hashes
before opening evaluator answers. Expose no reference passages in the public report.

## Grading and interpretation

Report character edit error (CER), word error, exact passage recovery, actual search
steps, selected representation, runtime, and the best candidate CER as an explicitly
post-hoc oracle diagnostic. The oracle diagnostic cannot count as decoder success.
Recovery gate: CER <= 1% and word error <= 10% per passage. Summarize modern/historical
conditions separately. Do not hide a failed positive control.

If Naibbe fails while substitution succeeds, the earlier codebook-assisted result does
not transfer to this codebook-free method. If the artificial variable-length control
also fails, state that search/selection is not validated for the broader representation.
Neither outcome proves all codebook-free methods impossible or identifies Voynich's
cipher family. No Voynich final-test text is scored; no translation is claimed.
