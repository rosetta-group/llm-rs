# Participant-complete graphs: useful recovery, failed recall gate

This experiment transfers known relationship graphs to 16 different Etruscan
inscriptions using supplied names and their grammar.
The redesigned model recovers 10 complete interpretations without a wrong
accepted edge, but fails the frozen recall requirement.

**Edge:** one relationship with its participants, such as CHILD_OF(Vel,Laris).
**Graph:** all reference relationships within the five-relation ontology for one inscription.
**Precision:** correct accepted edges divided by all accepted edges (11/11 here).
**Recall:** correct accepted edges divided by all reference edges (11/23 here).

## What was done

- Added exhaustive injective name mappings and bounded composition of connected
  graphs that account for all supplied participants.
- Added training-only anonymous-stem evidence, three controlled model variants,
  two inherited baselines and 99 shuffled-translation controls.
- Froze code, tests, data and protocol before the first evaluation run on the
  16 inscriptions. The previous eight tests became training/development evidence.
- Ran 26 unit tests successfully. Verified 50 hashes across the three earlier
  frozen experiments and 17 hashes for this experiment. Earlier evidence is unchanged.
- Saved post-run exhaustive-search diagnostics separately; no scoring code,
  threshold or annotation changed after the freeze.

## Why it was done

The preceding model lost some correct child-parent pairings during monotonic
alignment, including Ta 1.13 and Cl 1.1885. Keeping alternative mappings and
requiring every name tests whether those representation failures can be repaired.

## Frozen execution

```text
Use the previous 30 monuments as labelled training evidence
Predict the 16 public test views with the frozen models
Score their accepted graphs against 23 reference relationships
Shuffle training translations 99 times and rerun the full primary model
Apply the unchanged continuation gate
Diagnose failure without retuning
```

See [protocol](PROTOCOL.md), [data audit](AUDIT.md), [freeze](freeze.json),
[results](results.json), and [post-run diagnostics](diagnostics.json).

## Results

| Method | Correct / called edges | Precision | Recall | Edge F1 | Documents with calls | Exact graphs |
|---|---:|---:|---:|---:|---:|---:|
| Complete graphs, primary | 11 / 11 | 100.0% | 47.8% | 64.7% | 10 / 16 | 10 / 16 |
| Single mapping | 10 / 12 | 83.3% | 43.5% | 57.1% | 11 / 16 | 8 / 16 |
| Partial graphs allowed | 8 / 10 | 80.0% | 34.8% | 48.5% | 10 / 16 | 8 / 16 |
| No lexical evidence | 9 / 9 | 100.0% | 39.1% | 56.2% | 8 / 16 | 7 / 16 |
| Old local alignment | 8 / 11 | 72.7% | 34.8% | 47.1% | 10 / 16 | 5 / 16 |
| Entity-count majority | 8 / 18 | 44.4% | 34.8% | 39.0% | 15 / 16 | 4 / 16 |

The primary gains 17.6 F1 percentage points over old local alignment and 25.7
over the count baseline. The shuffled-translation F1 mean is 9.7%, with a maximum
of 38.7%. None of 99 shuffles reaches the observed 64.7%; the specified p-value
is 0.01. This is a conditional permutation result for this selected dataset,
not evidence that the historical reference translations are independently true.

| Frozen criterion | Result | Outcome |
|---|---:|---|
| Precision >=80% | 100.0% | Pass |
| Edge recall >=60% | 47.8% | **Fail** |
| Document coverage >=50% | 62.5% | Pass |
| Exact-graph accuracy >=50% | 62.5% | Pass |
| F1 improvement over old local >=15 points | +17.6 | Pass |
| F1 improvement over count baseline >=15 points | +25.7 | Pass |
| Translation-null p <=0.05 | 0.01 | Pass |

The overall gate **fails**. No unknown-word predictions follow from this run.

## What the comparisons establish

1. **The missing-candidate problem is repaired.** All 23 test edges survive
   candidate generation and the edge budget. Both previously missing development
   targets, Ta 1.13 and Cl 1.1885, are now present; all eight old target edges are
   available. That is representational coverage, not successful interpretation:
   on the old eight, trained only on their original 22 examples, the new model
   still accepts only one target, Cr 3.10.

2. **All-person coverage helps on gifts.** The primary recovers all seven edges
   across six gift inscriptions, all three ownership inscriptions and the maker
   in AV 6.1. ETP 128 is the useful complex success: both Venel and Velkhae are
   correctly recipients. The single-mapping variant wrongly makes Venel the
   donor. Allowing partial graphs reduces F1 from 64.7% to 48.5% and permits
   unsupported calls in ETP 287 and Cr 5.3.

3. **Lexical evidence contributes, so mapping changes do not deserve all the
   credit.** Without anonymous-stem associations, F1 falls to 56.2% and exact
   graphs to 7/16. The extra supervision includes mulu-family translations from
   the old eight. The new gift successes are transfer of known examples to
   different monuments, not rediscovery of a withheld lexical family.

4. **The remaining barrier is ranking, not search width.** Exact enumeration of
   valid graphs using the frozen retained edges gives the same best graph as
   beam search for all 16 inscriptions. Every reference graph is representable
   within those edges. All six abstentions have an incorrect best graph; simply
   lowering the confidence margin would introduce errors or incomplete answers.

## The Participant-Coverage Trap

Accounting for every person does not guarantee accounting for every statement.
Cr 5.3 makes this concrete: CHILD_OF(Vel Matunas,Laris) covers both names but
omits MADE(Vel Matunas,OBJECT), the tomb-construction statement. The reference
two-edge graph ranks fifth; its cost is 0.295 versus 0.130 for parentage alone.
The complexity penalty favours the shorter interpretation, while weak stem
associations do not force a second predicate to be explained.

| Abstained inscription | Reference graph rank among all retained-edge graphs | Preferred interpretation's problem | Best/runner-up margin |
|---|---:|---|---:|
| AT 1.46 | 2 | One parent becomes a spouse | 0.0250 |
| ETP 181 | 2 | One parent becomes a spouse | 0.0083 |
| ETP 287 | 6 | Reversed parentage ties with spouse | 0.0000 |
| Cr 5.3 | 5 | Tomb construction omitted | 0.0757 |
| Ta 1.191 | 2 | One parent becomes a spouse | 0.0062 |
| Cl 1.324 | 4 | Husband placed in a parent-child chain | 0.0023 |

The model recovers zero of five daughter-cohort edges and zero of seven
mixed-cohort edges. Even two successes are close to the fixed 0.10 margin:
ETP 128 at 0.1033 and AV 6.1 at 0.1040. The observed 11/11 precision is a small
sample result, not a general accuracy guarantee.

## Interpretation and boundary

This is evidence for a narrower useful capability: recovering some known gift,
ownership and making formulas on different inscriptions with manually supplied
names. It does not support reliable complex kinship parsing or new Etruscan
glosses. The corpus is shared with training, translations guided case selection,
restored/editorial readings are accepted, and the implementer saw reference
translations while the deterministic model did not.

A distinct future hypothesis would require candidate predicates to explain
non-name clauses as well as people. Cr 5.3 provides a development example of that
requirement; these 16 inscriptions must then remain development evidence, with
another preselected test needed. This report does not change the frozen model
or launch another round on its test cases.

## Reproduction

Existing manifests and results are immutable outputs: build/freeze/run refuse
to overwrite their artifacts. To reproduce, copy the code and pinned source
files to a clean experiment directory and use the following commands there.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_etruscan*.py' -v
.venv/bin/python -m experiments.etruscan_graphs build
.venv/bin/python -m experiments.etruscan_graphs freeze
.venv/bin/python -m experiments.etruscan_graphs run
.venv/bin/python -m experiments.etruscan_graph_diagnostics
```

`PROTOCOL.md` and `AUDIT.md` must be copied unchanged before freezing. A new
timestamp changes the freeze hash, so the expected comparison is identical
predictions, metrics and null results rather than byte-identical `results.json`.
