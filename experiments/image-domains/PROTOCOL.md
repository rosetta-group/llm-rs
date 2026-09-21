# Broad illustration domains and manuscript text

Frozen before domain classification, 2026-09-21. This extends the plant-only studies.
It is an exploratory association study on the existing training split, not a translation
test, new pixel annotation, or BPC experiment. CPU only; final test remains sealed.

## Labels and scope

Use IVTFF `$I` illustration categories already supplied with the GC2a transcription.
The format specification describes these as illustration types; they are conventional
visual classifications, not verified topics of the unread text.

- Botanical: herbal H and pharmaceutical P (the latter includes containers/plant parts).
- People/bathing: balneological B, the conventional biological section.
- Celestial/diagrams: astronomical A, cosmological C, zodiac Z.

Keep all eight original category counts in the coverage report. Stars S are marginal-star
text pages, not automatically astronomical diagrams; text-only T is also excluded from
this three-domain comparison. These are dominant domains, not mutually exclusive object
presence labels. Zodiac pages also contain people/animals. A separate presence annotation
would be needed to distinguish each co-occurring object type.

## Data unit and controls

Use only folios assigned to training in the existing folio-42 split. Include paragraph,
label, circular, and other transcribed loci; paragraph-only documents omit some diagrams.
Skip loci explicitly excluded by the IVTFF locator `!`. Normalize with the existing
parser; strip annotations, split written forms at boundaries, exclude uncertain forms.
Do not use `$I`, page identifiers, or section names as text features.

Aggregate text by physical folio, combining sides and panels. The Rosettes foldout is one
group 85-86, including fRos. Require a single broad domain per aggregate; report mixed
aggregates and exclude them rather than choose a label by its text. Use folios as both
prediction and uncertainty units. Coverage audit currently yields 63 independent groups:
50 botanical, 7 people/bathing, 6 celestial/diagrams.

Nuisance controls: log character/word/locus/page counts; mean/SD written-form length;
locus-type proportions; unreadable-form fraction; proportions of inherited hand labels.
No Currier class: it is a text-derived grouping, not independent image evidence.
A strict descriptive control additionally includes quire identity and folio position.
Record the domain-by-hand and domain-by-quire contingency tables before scoring.

## Fixed models

Compare word-frequency and within-form character 1–4-gram representations separately.
Hash each view into 2,048 bins, use log count × training-fold inverse document frequency,
then L2 normalize. No n-grams cross word, line, or page boundaries.

Use centered kernel ridge, penalty 1, one-hot class targets, argmax prediction.
Controls use twice a radial kernel with median positive training-distance bandwidth;
control-plus-text uses control kernel plus squared cosine text kernel. Fit feature
scaling, IDF, bandwidths, and models only on the training folds. Class-prior and text-only
models are descriptive comparators. No model tuning on held-out results.

## Two tests of transfer, four planned comparisons

1. **Leave one physical folio out**, three domains. Primary statistic: improvement in
   macro recall (balanced accuracy) over hand/layout controls, one comparison per text view.
2. **Leave one quire out**, only domains represented in at least two training quires and
   at least five folios. Botanical and celestial qualify; people/bathing does not.
   Report that exclusion explicitly; do not score an unseen class as a model failure or
   claim cross-quire human-domain validation. Same two text views and primary statistic.

For each design, use 999 permutations of whole-folio domain labels within exact inherited
hand-signature strata (including mixed-hand signatures). Refit all models. This preserves
hand/domain composition; it does not exactly condition on every continuous layout feature.
Inference is exploratory and conditional on this declared exchangeability assumption.
Use one-sided p=(1+null>=observed)/1000 and Holm correction across four planned comparisons.
Report per-class recall, confusion matrices, accuracy, macro recall, and every result.
If no label can move within hand strata, mark that conditional test unidentifiable;
reserve p=1 only for its place in the four-test correction, not as evidence of no relation.

Use 2,000 paired bootstrap draws: physical folios for the first design, held-out quires
for the second. Draws missing any evaluated class are omitted and their count reported.
Intervals are descriptive, not simultaneous. A supported incremental signal requires
gain>0, Holm p<=0.05, and interval above zero. Descriptive high text-only accuracy alone
is not evidence of image semantics independent of hand, layout, or manuscript position.

## Book-structure identifiability audit

Also enumerate which domain labels can move within exact (hand signature, quire) strata.
If every stratum has one domain, a fully book-structure-conditioned permutation test is
infeasible. Record it as unidentifiable, not a significant result or proof of no relation.
Show the strict hand/layout/quire/position baseline under folio holdout as a descriptive
check. A predictive domain association cannot establish any individual word's meaning.

## Validation and budget

Test annotation exclusion, final-test filtering before normalization, Rosettes grouping,
no cross-word n-grams, fold isolation, grouped permutations, macro-recall handling, and
a synthetic domain signal with matched nuisance features. Preserve old frozen studies.
Budget: one CPU hour, 100 MiB outputs, at least 20 GiB free disk. Store source/code hashes,
all predictions and corrected tests, and graphs. No new model downloads or paid compute.
