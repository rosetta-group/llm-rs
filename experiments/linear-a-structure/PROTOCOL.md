# Entry endings: one structural question

Declared 2026-09-24 before any model fit or score. CPU only, fixed existing corpora.

**Question:** Do word-final sign sequences predict whether an unseen word immediately precedes
a number (possibly after a commodity sign) on inscriptions withheld from training?

**Entry role:** the observable numeric-entry position, not an inferred personal name or meaning.
**Balanced accuracy (BA):** average of entry and non-entry recall; an always-one-class model gets 50%.

## What will be done

- Use Linear A sign identities, including signs without assumed sound values. Retain intact
  sign runs of length >=2. Gaps/editorial marks invalidate their run; do not turn fragments into
  complete words. Logogram boundaries keep round two's declared 80% role threshold. Use the
  same adjacent-number labels as before. DĀMOS syllables serve as opaque sign IDs for Linear B.
- Group lettered faces of each tablet. SHA256 of `entry-endings-v1:` plus its group ID modulo
  five selects 20% of documents for testing. Exclude every test word type found anywhere in
  training, including ambiguous training types. Collapse remaining occurrences to one type;
  discard types with conflicting role labels within their partition. No random split search.
- Fixed baseline: length bucket (6+ pooled) and bag of sign counts. Fixed candidate: baseline
  plus final sign and final two signs. Both are multinomial naive Bayes, Laplace smoothing 1,
  equal class priors, training vocabulary only; no tuning or full-word feature.
- Compare on Linear B first, including 20 samples matched exactly to Linear A's training/test
  class counts. Shuffle signs within each sampled word for a negative control; bags and lengths
  remain identical. Report full-size Linear B descriptively, even if the size-matched gate fails.
- Test 199 within-site/length-bucket permutations of training role labels with fixed test data.
  Their p-values are diagnostics under that exchangeability assumption, not archaeological
  significance or independent replication. Report raw null gains.

## Why

The profile audit concerns whole-corpus averages and cannot establish whether endings transfer
to unseen forms in context. This test asks about one positional relationship while retaining
unknown signs and excluding memorisation of complete words.

## Fixed decisions

```text
Freeze code, source hashes and protocol; commit before fitting
Score Linear B at the target's training/test size
Require BA >= 0.60 and gain over bag/length baseline >= 0.05 in at least 18/20 samples
Require the same criterion in at most 1/20 within-word shuffled controls
If the gate fails: report the failure and do not score Linear A
If the gate passes: score the fixed Linear A partition once
Call a structural lead only if the same BA/gain criterion passes,
the label-shuffle p <= 0.05, and the within-word shuffled candidate fails
```

Scoping inspected parser output and class counts, but no model scores: Linear A has 857 retained
tokens on 336 documents, 511 training types (303 entry / 208 non-entry) and 85 unseen test types
(44 entry / 41 non-entry), including 10 types containing signs without known readings. Linear B
has 3,193 training and 544 unseen test types. These counts and the small test size are disclosed
before freezing. Tests include an artificial order-only positive example (same bags, opposite
endings) to verify implementation, not to replace the real Linear B control.

## Limits

- All corpora were available in earlier work; this is procedural model holdout, not a newly
  discovered or independently sealed corpus. The 20 samples overlap and are not 20 independent
  replications. No statistical claim is made from their win count.
- Faces are grouped by identifier; words common to training and test are excluded. Distinct
  inscriptions sharing partial formulas may remain related. Site and document genres may
  confound associations. The model does not see site, but the diagnostic label shuffle conditions
  on site and length.
- A success would establish predictable numeric-entry position in this parse, not morphology,
  a language family, a translation, or that all entry words are names. A failure retires only
  this simple ending model and representation.
- Damage exclusions and type-role consistency filters define a subset, not all Linear A.
  Source/licence records remain in rounds one/two; derived results are CC BY-NC-SA 4.0.

## Reproduction

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a_structure.py' -v
.venv/bin/python -m experiments.linear_a_structure freeze
# Commit code, protocol and freeze before fitting.
.venv/bin/python -m experiments.linear_a_structure verify
.venv/bin/python -m experiments.linear_a_structure run
```
