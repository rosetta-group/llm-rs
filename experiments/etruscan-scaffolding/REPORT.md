# Names as scaffolding: pilot result

Run 2026-09-24 under the unchanged [protocol](PROTOCOL.md), with code, annotations,
public inputs and sources pinned in [freeze.json](freeze.json).

**The joint model abstains on every hidden case. The simpler local model recovers
one of eight target relationships. The continuation gate fails.** This result is
about the implemented alignment models, not a limit on all relational approaches.

## What was done

- Audited 30 translated inscriptions and encoded people, deities, the inscribed
  object and directed relationships. Names and relation annotations remain
  separately inspectable in [manifest.json](manifest.json).
- Withheld the complete translations of three daughter-family and five mulu-family
  inscriptions. The 22 remaining translations supply training graphs. The predictor
  receives neither the hidden translations nor the target words' dictionary meanings.
- Implemented whole-inscription alignment and joint evidence across recurring
  anonymous stems, with abstention and exact participant scoring. Ran 999 shuffled
  translation controls and ten focused tests, including synthetic positive controls.

## Why

The prior classifier only used local classes and word endings. This pilot tests
whether recognised names and complete inscription patterns can constrain a hidden
word's relationship, and whether repeated occurrences disambiguate one another.

## Frozen test results

The eight target edges are specific: correct child **and parent**, or correct
donor **and recipient**, including explicitly unspecified participants.

| Model | Correct target edges | Documents with accepted calls | Correct / all accepted edges |
|---|---:|---:|---:|
| Joint stem-family constraints — primary | 0 / 8 | 0 / 8 | undefined: no calls |
| Local name/morphology alignment | 1 / 8 | 6 / 8 | 1 / 6 |
| Local alignment without case/gender | 1 / 8 | 5 / 8 | 1 / 5 |
| Entity-count-only baseline | 0 / 8 | 8 / 8 | 0 / 11 |

The primary translation-permutation p-value is 1.0: zero recovered targets cannot
beat the shuffled controls. It is not a probability that the approach is false.
Mean shuffled target recall was 3.5%; 110 of 999 shuffled runs recovered at least
one target. None of the primary continuation requirements passed.

The descriptive leave-one-monument-out check on the 22 training inscriptions
recovered 12 of 23 annotated relationships, making 16 calls with 75% precision.
It exactly recovered 12 complete graphs. Thus the local machinery can transfer
some familiar formulas, but this competence did not extend to the hidden families.
No setting or threshold was chosen using that check.

## Where it failed

1. **The daughter cases need the right parent.** Local alignment gets the explicit
   daughter edge in Ta 1.59 right. In Cl 1.1885 it assigns the husband as parent,
   rather than the mother. In Ta 1.13 it abstains. The joint family's support splits
   between CHILD_OF (55.8%) and SPOUSE_OF (40.6%), below the frozen 60% threshold.
2. **Gift formulas resemble other object formulas.** Local alignment calls Vt 3.1,
   Cr 3.11 and Cr 3.10 ownership, and calls Cr 3.9 making. In the anonymised input,
   Cr 3.9 has exactly the same structural pattern as a training “made me” text.
   Pooling the mulu family does not resolve this: OWNED_BY scores 46.2%, MADE 38.1%,
   and TRANSFER 15.6%. These are heuristic supports, not calibrated probabilities.
3. **A richer hypothesis search is a separate issue from more data.** In an
   explicitly post-result diagnostic of the saved candidates, the correct target
   edge is absent for Ta 1.13 and Cl 1.1885. Both contain several people, and the
   single best monotonic alignment discards the needed pairing. Even a perfect
   reranker of these frozen candidates could recover only one of three daughter
   targets, so lowering the acceptance threshold cannot repair that family.

All five correct gift frames occur somewhere in the saved joint candidates. Their
failure is ranking/identification, whereas two daughter failures already occur
during candidate generation. These are different engineering limitations.

## What is and is not learned

The experiment gives no validated new Etruscan meaning. It identifies two concrete
weaknesses to address in a separately declared design: retain alternative participant
alignments, and require proposed actions to account for the other named participants
instead of accepting a good partial match. Those changes were **not** fitted or
tested against these eight answers after the result.

The exact “gave/made” collision shows that one local abstract pattern is insufficient
for that distinction; it does not show that the complete inscriptions, additional
grammar, object evidence, or a better joint model are insufficient. Conversely,
the presence of a correct frame among candidates is not a successful prediction.

## Decision

```text
Retain the audited benchmark and its error cases
Stop this frozen template-alignment implementation
Keep the eight revealed answers as development evidence for any future redesign
Require a separate test before claiming improved recovery
```

This is a small selected pilot with two hidden lexical families. Name spans and
person/deity distinctions were manually supplied; some training relationships and
two readings are editorial reconstructions. Relations deliberately collapse finer
distinctions, such as giving versus dedication. Several abstract formulas repeat.
The three-entity training stratum contains only ETP 189, so that graph cannot move
under the entity-count-stratified shuffle. All these limits were stated before
scoring in the [audit](AUDIT.md) and [protocol](PROTOCOL.md).

## Verification and artifacts

- Ten tests pass: hidden-label separation, ambiguous name grammar, directed roles,
  synthetic recovery, abstention, discontinuous names, monument separation and
  whole-graph permutation integrity.
- The run checked all pre-run hashes. Earlier frozen Etruscan experiments remain
  unchanged. Full predictions, transferred candidates and controls are in
  [results.json](results.json); the exact predictor inputs are [public.json](public.json).
- Build, freeze and run refuse to overwrite outputs. For a reproduction, use a
  separate copy and move only its saved results out of the output path before
  running the frozen command in [PROTOCOL.md](PROTOCOL.md).
