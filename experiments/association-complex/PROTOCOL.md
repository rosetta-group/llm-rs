# Complex text–image associations: exploratory extension

Frozen before scoring this extension, 2026-09-21. The earlier root-color pilot is
preserved. This study reuses its annotation source and some examples, so it is
exploratory even though this analysis is frozen before evaluation. Final Voynich
test folios remain sealed. CPU only; no downloads of models or cloud jobs.

## Questions

1. Do nonlinear combinations of EVA label features predict the earlier explicit
   root-color endpoint better than nonlinear length/layout controls?
2. Can label text predict a joint profile of visual descriptions, including
   part/color combinations, shape, texture, size, and arrangement?
3. Can a held-out label retrieve its paired visual description among alternatives
   on the same page, rather than exploit differences between sections or folios?
4. Do relationships between predicted profiles track relationships between actual
   descriptions within held-out pages?

These are associations with human image descriptions, not direct pixel models or
biological absence labels. Original annotators could see the writing. No word
meaning, species identification, or causal image/text link follows automatically.

## Eligibility and targets

Use the pinned Grove/Stolfi catalogue and original deduplication and folio aliases.
For the binary endpoint, retain the original pilot's exact eligibility rules.
For richer profiles, use confident `plant` objects, clear EVA labels, existing training
folios, hand 1, and pharmaceutical section only. Exclude uncertain clauses containing
`?`, editorial clauses about letters/labels, and cross-references. Keep other clear
clauses. Use an explicit botanical vocabulary; ignore proposed species identities.

The fixed descriptor vocabulary covers roots/leaves/flowers/stems/twigs/bulbs;
explicit light/dark color attached to a part; round/triangular/conical/square/arrow
shapes; hairy/fuzzy, stripes/spots/edges; large/long; multiple/split roots and explicit
one/two/three/four counts. Zero means **not mentioned**, never biologically absent.
Add all pairwise co-mentions as joint targets. Retain a target dimension only when it
has at least three mentions and three non-mentions in that fold's training data.
No target vocabulary or parameter is selected by held-out association scores.

Require at least 60 usable whole-plant descriptions across four physical folios and
40 objects on pages with at least three usable descriptions. Otherwise record that
part of the extension as infeasible. Never relax gates after inspecting associations.

## Models and evaluation

Leave one physical folio out. All panels and page sides travel together.

Controls: label length, word count, relative label index, location group, transcriber.
Standardize numeric/categorical controls using the training fold and use a radial
kernel with bandwidth equal to its median positive training squared distance.

Text: normalized fixed-hash EVA 1–4-grams with boundary markers plus skip-bigrams
(one intervening character), 512 bins. Compare three fixed kernels:

- additive: cosine similarity;
- interactions: squared cosine similarity;
- nonlinear: radial kernel on the normalized feature vectors, median positive
  training-distance bandwidth.

Control model uses twice the control kernel. Each text model uses control + text
kernel, keeping diagonal scale equal. Kernel ridge penalty is 1 with an unpenalized
intercept; binary threshold 0.5. No hyperparameter sweep or test-driven reranking.
For profile targets, standardize with training-fold means and standard deviations.
Cache linear prediction operators: they exactly refit the fixed ridge model for
permuted targets. Training target support/scale is invariant to within-page permutations.

Four endpoints, each for three text models (12 planned tests):

1. Root-color balanced-accuracy improvement over controls.
2. Joint-description profile mean-square-error reduction versus controls, standardized
   by training-fold target variability (positive means lower error).
3. Within-page matching rank improvement. Compare each predicted profile with every
   description on that same held-out page using cosine similarity. Normalize ranks
   to 0–1; chance is 0.5. Give tied descriptions half credit, not arbitrary wins.
4. Within-page relational alignment improvement: Pearson correlation between pairwise
   predicted-profile cosine similarities and observed-profile cosine similarities.
   Compute separately per page, then average over eligible pages. Constant similarities
   contribute zero; pages need at least three objects. This measures descriptions'
   relationships, not a recovered dictionary.

Use 999 within-page permutations, moving the whole profile (and all co-mentions)
together. Refit every model and compute all four improvements. One-sided p-values
are (1 + null >= observed) / 1000. Apply Holm correction across all 12 planned tests,
including non-significant outcomes; infeasible tests count as p=1. Do not select a
winning metric/model after the fact. Also show 2,000 paired physical-folio bootstrap
intervals for each improvement; these are descriptive, not simultaneous intervals.

A signal requires a positive gain, Holm p<=0.05, and a folio interval above zero.
Even a passing result needs independent image annotations and fresh held-out material.
Archive all outcomes, including negatives and unstable gains.

## Sanity checks and budget

Test the same grouped kernel/permutation machinery on a synthetic interaction-only
(XOR) positive control. A squared/radial kernel should recover the planted interaction
while an additive kernel cannot. This checks implementation, not Voynich meanings.
Check held-out targets have zero influence, same-page permutations preserve joint
profiles, target construction ignores uncertain/editorial clauses, and matching ties
receive half credit. Full suite budget: one CPU hour and 100 MiB additional local
outputs; preserve 20 GiB free disk. Record incomplete work as incomplete.
