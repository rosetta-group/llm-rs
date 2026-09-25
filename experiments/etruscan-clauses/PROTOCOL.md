# Clause Coverage: frozen evaluation protocol

Written 2026-09-24 before scoring the 20 new inscriptions.
This tests weakly learned predicate spans and local participant roles, with an
explicit penalty for unexplained evidence. It is a token-span pilot, not a full
grammar or a claim that every word must express one of five known relations.

**Predicate:** one CHILD_OF, SPOUSE_OF, OWNED_BY, TRANSFER or MADE relationship.
**Span anchor:** one non-name token occurrence assigned a relation from training.
**Coverage:** how well distinct candidate edges explain repeated anchor occurrences.

## Data and model boundary

Use all 46 prior audited monuments as labelled training. The old 16 test cases
are development evidence, stored separately. Evaluate on the fixed 20 monuments
and 36 edges in `manifest.json`; eight complex cases contain 20 of those edges.
The source, manual anchors and reference limitations are in [AUDIT.md](AUDIT.md).

Public inputs are unchanged: ordered anonymous entity anchors, name/deity
type and dictionary case/gender, a fixed generic deictic marker, and opaque
three-letter stem IDs for other tokens. No test translation, cohort, predicate
span annotation, gold edge or non-name dictionary meaning enters the predictor.
Translations annotate training graphs only; the learner receives those graphs,
not English prose or supplied word-to-predicate alignments.

## Fixed mechanics

```text
Count training stem/relation associations, once per stem per monument
Select confident predicate anchors while allowing background text
Learn relative role positions from those anchors and training graph edges
Generate candidate edges using every eligible participant mapping
Compose complete graphs and penalise unexplained predicate anchors
Accept only a sufficiently separated low-cost best graph; otherwise abstain
```

1. **Contrastive anchor learning.** For each stem s, let n(s) be its training
   document count, n(s,r) its count with relation r, N the training count and n(r)
   the document count containing r. Relations with n(s,r)=0 receive no score.
   Otherwise use lift `((n(s,r)+0.5)/(n(s)+1))/((n(r)+0.5)/(N+1))`.
   Add a background option of weight 1 and normalise these weights. Select the
   largest relation only if its normalised weight is >=0.60; ties use label order.
   Every occurrence of a selected stem is an anchor. Unseen and ambiguous stems
   remain background. These heuristic weights are not probabilities of truth.

2. **Local role learning.** For each selected training anchor, retain the signed
   distance from that token to the nearest occurrence of each named role in
   every training edge of the selected relation. Ties choose the earlier name
   occurrence. Clip distances to [-4,4] and divide by 4. OBJECT and UNSPECIFIED
   retain their literal role identities. All positions use the inherited public
   sequence, which collapses adjacent tokens within a name. A query edge's local
   attachment cost is its smallest mean role-offset difference against templates
   of the same anonymous stem, with each difference capped at 1. Incompatible
   literal roles cost 1. A different relation or missing template costs 1.
   SPOUSE_OF allows either role order.

3. **Span accounting.** Group query anchors by anonymous stem. Within a group,
   each occurrence can use a distinct graph edge, costing its local attachment
   distance, or remain unexplained at cost 1. Find the minimum assignment by
   dynamic programming over edge subsets. Average the cost over all anchors.
   Different stems may explain the same edge; one occurrence does not forbid
   multiple coordinated graph edges. A graph edge's local cost is its smallest
   attachment cost over all anchors (zero when there are no anchors). Thus an
   unsupported extra predicate also carries a cost when anchors are present.

4. **Graph selection.** Reuse all-mapping candidate generation, mapping costs,
   provenance and validity constraints from the frozen participant-graph model.
   New graph cost is its old graph cost +0.70*unexplained-span cost
   +0.25*mean local edge cost. Old cost retains mean mapping distance,
   0.18*(edge count-1), and the -0.30 bag-of-stems reward. Retain the best 18
   singleton edges by the new score, plus the best incident edge for each name
   and the best attachment for each anchor. Enumerate every valid connected
   graph of one through four retained edges. Every query name must be covered.
   This replaces beam search; it is exhaustive only over the retained edges.
   The best graph needs cost <=0.65 and margin >=0.10 over the second best;
   a sole graph requires only the cost check. Thresholds match the earlier model.

## Fixed comparisons

- **Primary:** clause coverage, local role offsets, four-edge exhaustive search.
- **No coverage:** set the unexplained-span term to zero; keep local edge costs.
- **No local roles:** matching relation costs zero regardless of role offsets;
  different relations still cost one. Keep coverage and all other terms.
- **Extended graph:** remove span anchors entirely but retain four-edge search.
  This controls for search/capacity changes separately from predicate grounding.
- **Previous graph:** unchanged preceding model, trained on the same 46 records.
- **Count-only:** inherited participant-count majority, trained on the same 46.

All are fixed before evaluation. Development testing on the old 16 cases was
performed once; no parameter sweep or test-dependent semantic exceptions follow.

## Null control and success criteria

Shuffle complete training graphs within entity-count strata 99 times, with seed
240927. Refit stem associations, anchor selection, local templates and old lexical
evidence for every shuffle. Reuse only label-independent geometric mappings.
Run graph selection and abstention again. The one-sided F1 p-value is
`(1 + number of null F1 values >= primary F1)/100`.

All criteria must pass:

- Overall accepted-edge precision >=80% and no lower than the previous model.
- Overall edge recall >=60%, and document coverage >=50%.
- Complex-case edge recall >=60% and exact graphs >=4/8.
- At least one more exact complex graph than the previous model.
- Overall edge F1 at least 10 percentage points above the previous model.
- Overall edge F1 at least 5 percentage points above the extended-graph control.
- Translation-shuffle p <=0.05.

Every accepted edge is scored by relation AND participant roles; missing calls
count as missed reference edges. Report complete-graph accuracy and cohort
metrics as well as precision, recall and F1. Secondary diagnostics may inspect
candidate loss, anchor errors and omitted predicates but cannot alter predictions.

## Stop rule

```text
Freeze code, tests, data, prior evidence, and protocol hashes
Score the new 20 cases once with every fixed method
Run the 99 translation-shuffle controls
Apply all criteria without changing annotations or thresholds
If the gate fails, stop tuning this dataset
Prioritise independently checked translations and grammatical span annotations
```

A pass would justify separate source-controlled validation, not unknown-word
glosses. A failure ends this local tuning sequence. All earlier frozen evidence
must remain unchanged; development and new results remain separate artifacts.
