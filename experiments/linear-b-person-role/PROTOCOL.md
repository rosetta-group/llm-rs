# Name versus designation: a Linear B control

This experiment tests whether anonymous layout features distinguish personal names from
occupational, title or work-group designations in known-language Linear B. It follows the
qi-tu-ne audit, where both readings fit the observed heading/count contrast.

**PERSON:** a source-identified personal name, including an inflected owner's name.
**DESIGNATION:** an occupation, title or human-group/status term; its precise gloss may be uncertain.
**Abstention:** no class assigned when independent training support is absent or conflicting.

## Scope and supplied information

This is a curated development challenge, not an independent sealed evaluation or a random
sample of Linear B. Selection and annotations are exposed to the investigator before freeze.
The primary label source is Ventris & Chadwick, Documents in Mycenaean Greek (1956, 1959
reprint), cross-checked against pinned DĀMOS transcriptions. Old sigla, changed readings and
ambiguous cases are recorded. Source attribution is not independent modern expert adjudication.

Include explicitly identified personal names, occupations, titles and work-group/status
designations with readable target strings. Exclude places, ethnic labels, deity names, generic
children/kinship terms and unresolved personal-name versus title readings. Do not infer gold
labels from position or quantity. A target can be readable in an otherwise damaged record.
Select a small set including counterexamples, not every name in a long roster. All selections
are listed in cases.json before scoring; no additions or exclusions after seeing predictions.

Inputs supplied to the learner are only manual written role (heading/entry/footer), the
quantity bin (none/one/many/unknown) for the annotated entry, initial/noninitial word position,
and whether that exact intact non-erased word repeats on its object. A multiword entry's
quantity is shared context, not an assertion that every word independently counts people.
Entry boundaries and target spans are supplied hints, not recovered by the method.
No Greek spelling, endings, lexicon, translation, tablet series, site, source page, gender,
personal identity or gold class is available as a feature. Object IDs are used only for folds
and weighting; token IDs are used only for within-object equality. No Linear A is scored.

## Frozen procedure

```text
Source-annotate target spans, roles, quantities and independent labels
Validate each target and its literal repetition against the pinned object text
Export anonymous structural features separately from the label key
Freeze and commit code, annotations, settings, tests and source hashes
For each physical object, train on all other objects and predict its cases
Repeat the full validation with whole-object swapped labels as a negative control
Report abstentions, errors, feature collisions and the fixed gate
Keep Linear A unlabelled regardless of this small pilot's outcome
```

Three fixed feature sets are evaluated: **role**, **role_quantity**, and **layout** (role,
quantity, initial/noninitial position, within-object repetition). Layout is primary; the two
simpler sets are diagnostics, never substitutes selected after observing outcomes.

For an exact feature signature, each training object contributes one vote distributed among
the classes occurring with that signature on that object. Predict only with at least two
supporting objects and >=90% of object-normalized votes for one class. Otherwise abstain.
No feature backoff, threshold tuning, lexical transfer, morphology or guessed meanings.
A forced-majority diagnostic uses the same signatures, abstaining only on no support or ties;
it does not determine acceptance. Leaving one object out removes all its faces and rows.

Metrics give equal weight to the two gold classes, then equal weight to objects containing
each class, then equal weight to that object's cases of the class. Balanced recall counts
abstentions as failures. Also report coverage, conditional accuracy, raw counts and every
held-out prediction with its training object support. A single long roster must not dominate.

An empirical feature ceiling sums the larger class weight in each exact signature. It is an
optimistic in-sample upper bound for deterministic classifiers using ONLY that feature set,
not a held-out result or a limit on all possible linguistic methods. Mixed signatures retain
source IDs for concrete counterexamples.

## Gate fixed before scoring

- At least 10 physical objects, each class on >=5 objects, and each class represented on
  >=2 heading objects and >=2 entry objects. If insufficient, report not_evaluable.
- Primary layout: object/class-balanced recall >=90%, coverage >=90%, conditional accuracy
  >=95%. Abstention cannot manufacture a pass from a handful of easy calls.
- In 199 seeded negative runs (seed 20260924), independently swap PERSON/DESIGNATION for
  all cases on each object with probability 1/2, refit every held-out fold, and require <=5%
  of evaluable negative runs to pass the same performance thresholds. This preserves each
  object's internal grouping; it is a sanity control, not a linguistic significance test.
  Report performance distributions as well as pass counts. Spelling-shuffle invariance is
  checked in software: this structural method never sees syllables, so their rearrangement
  cannot produce a new language match.

This gate is necessary for considering an expanded control, not sufficient for assigning
Linear A meanings. Even a pass requires independent label review, a larger source-balanced
benchmark and a separate fresh validation. A failure closes this particular feature/method
combination; it does not prove names and occupations indistinguishable in richer evidence.

## Reproduction

`prepare` is not needed: the manual cases and derived public views are committed inputs.
The driver validates them, freezes hashes and refuses to overwrite result files. All source
snapshots and licences are in sources.json. Earlier frozen experiments remain unchanged.
