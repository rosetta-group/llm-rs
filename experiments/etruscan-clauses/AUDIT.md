# Clause Coverage: data audit

Prepared 2026-09-24 before scoring the new evaluation. The local Larth
`Etruscan.csv` and `ETP_POS.csv` remain the source; `freeze.json` pins their hashes.

## What was done

- Reused all 46 audited monuments from the scaffolding and participant-graph
  pilots as labelled training evidence. The preceding 16 tests are development.
- Selected and manually annotated 20 different monuments with 36 reference edges.
  IDs and exact normalised texts are disjoint from all 46 training records.
- Retained source indices, raw readings, translations, name spans and caveats in
  `manifest.json`; regenerated model inputs in `public.json`.

## Why it was done

The test concentrates on interpretations that must explain multiple predicates:
Ru 5.1 combines two parents with construction, and Vs 1.28 combines tomb ownership
with parentage. These distinguish statement coverage from merely including names.

| Cohort | Monuments | Edges |
|---|---|---:|
| Complex | Ru 5.1, Cr 5.2, Ta 1.182, Cr 3.17, Vs 1.28, AT 1.34, ETP 335, Vt 1.58 | 20 |
| Kinship | Ta 1.15, Ta 1.96, Cl 1.2261, Ta 1.168 | 7 |
| Transfer | Cr 3.18, ETP 303, Ve 3.2 | 4 |
| Making | ETP 120, Vc 6.6 | 2 |
| Ownership | Cm 2.32, Cm 2.65, ETP 331 | 3 |

All query entities are manually supplied people; no deity discrimination is tested.
There are at most three names and four reference edges per new inscription.
The inherited model searches at most n edges for n names. Cr 5.2 has four edges
for three names, so the new model permits four edges. An extended-search baseline
controls that extra capacity separately from the new span evidence.

## Reference interpretation limits

1. **Manual anchors remain favourable.** Split `pes nalisa` in Ru 5.1, `av le`
   and `laris al` in Cr 5.2, and `tiscusn al` in Cl 1.2261 are retained as separate
   tokens grouped into names. Cr 3.18's donor is discontinuous. These are supplied
   readings, not outputs of an automatic name recogniser.
2. **Restoration and implicit predicates are retained.** Ta 1.182, AT 1.34,
   Ta 1.168, Ve 3.2 and ETP 120 include supplied/restored letters. Several son,
   tomb and object readings are parenthetical. Their uncertainty is not resolved
   by this experiment. Vt 1.58's `klan` remains distinct from `clan`.
3. **Coarse ontology choices are explicit.** Both named brothers in Cr 5.2 get
   MADE edges for jointly having the tomb constructed. Both donors in Ve 3.2 get
   TRANSFER edges sharing OBJECT. Vc 6.6's “work of” is encoded as MADE.
   Cr 3.17's “born from” is encoded as CHILD_OF.
4. **The reference controls role direction.** Cr 3.18 explicitly says given by
   Licene Hirsunaie, despite the -si name forms. Its gold donor is not changed to
   a recipient to agree with morphology. ETP 303 assigns Aranth donor and
   Thankhvil Prasanai recipient. Potential scholarly disagreements remain source
   limits, not reasons to edit reference labels after scoring.
5. **Completeness is limited to the five relations and named people.** Death,
   age, family/living modifiers, Ta 1.96's uninterpreted `camthi eterau`, and
   Ta 1.168's unnamed children are outside the ontology. Tokens remain public
   even when their meanings are outside the ontology; the model must cope with
   that background text rather than receiving manual clause masks.

This is a same-source, hand-selected evaluation of supplied scholarship. The
implementer read translations to annotate the test. Only the deterministic
predictor is blinded. No independent-source, unknown-word or full-grammar claim
is warranted. The opaque stem hashes are an implementation barrier, not secrecy.

## Development evidence and known risks before freezing

`development.json` evaluates the unchanged first clause implementation on the old
16 cases with only the original 30 training examples: 8/16 exact graphs, 8/8
accepted edges correct, 8/23 recall. This is below the preceding model's 10/16
exact graphs and is not fresh test evidence. There was no threshold sweep.

The full 46-case training lexicon associates the `cer` stem with MADE at heuristic
support 0.714 from one monument, and `clan` with CHILD_OF at 0.719 from four.
These are not calibrated probabilities. `sec` remains confounded with spouse
contexts and fails the 0.60 anchor threshold. The same learner incorrectly treats
age/death stems (`avi`, `lup`) as CHILD_OF evidence because its few examples
co-occur with parentage. No hand-coded semantic exclusions repair that confound.
The new test checks whether the proposed mechanism survives these weaknesses.
