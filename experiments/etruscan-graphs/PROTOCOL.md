# Participant-complete graphs: frozen protocol

Written before scoring on 2026-09-24. CPU only. The question is whether retaining
multiple participant mappings and requiring all named participants improves
recovery of supplied relationship graphs on different monuments.

## Scope and information barrier

Train on all 30 previously audited monuments; evaluate on the 16 disjoint
monuments in `manifest.json` (23 reference edges). The old eight tests are now
development evidence. Source, selection, restoration and manual-anchor limits
are in [AUDIT.md](AUDIT.md). Test translations were visible to the implementer
for annotation but are absent from model inputs. This is an exploratory test
of known scholarship, not new meanings or independent historical validation.

Use the inherited five relations CHILD_OF(child,parent), SPOUSE_OF(a,b),
OWNED_BY(OBJECT,owner), TRANSFER(donor,OBJECT,recipient), MADE(maker,OBJECT).
SPOUSE_OF is symmetric; the others preserve role direction. UNSPECIFIED means
the participant is not given. Gold includes all in-scope relations, not ages,
death, or a complete translation. Public views retain only manually anchored
names and their dictionary grammar, generic deictics, and anonymous stem IDs.

## Fixed model

**Mappings.** Enumerate every injection from a training inscription's entities
to a query's entities, up to four query entities. If the training text has more
entities than the query it contributes no mapping. Complexity is at most
`n!/(n-k)!` assignments for k training and n query entities (24 at n=k=4).
No order constraint is imposed. Average mapped-pair costs: kind mismatch 0.5,
known case mismatch 0.45, known gender mismatch 0.15, plus 0.2 times the absolute
difference of mean normalised entity positions. Unknown morphology incurs no
mismatch cost. Add 0.1 if deictic counts differ. Keep costs <=1.0.

**Edge transfer.** Transfer training edges through each eligible mapping.
Reject missing endpoints and self-relations using the inherited frame validator.
Each distinct resulting edge keeps its cheapest mapping and source provenance.
Keep the full inventory for diagnostics, even when graph search prunes it.

**Lexical evidence.** This is an additional change beyond alignment/completeness.
From training only, count each opaque stem once per inscription, alongside the
set of relation types in that inscription. For relation r, its support is
`log((count(r)+1)/(count(total)-count(r)+1))`. No word-to-predicate alignment is
given. In a candidate graph, take the maximum positive support over its relation
types for each query stem, then average over distinct query stems. Unseen stems
contribute zero. Repeated edges of a relation do not multiply its lexical support.

**Composition.** Graph cost is mean edge mapping cost +0.18*(edge count-1)
-0.30*lexical support. Search retains the best 18 singleton edges plus the cheapest
incident edge for each entity. A beam of 128 graphs adds edges for at most n
rounds, where n is the query entity count. Search priority adds 0.4 per uncovered
entity; this penalty is absent from the final graph cost. Deduplicate graphs and
break ties lexicographically. Beam search is bounded, not exhaustive graph search.

Reject child cycles, more than two parents per child, a child/spouse conflict
for the same pair, and mixtures of object-event relation types. Multiple edges
of the same object relation are allowed, such as two gift recipients. Kinship
can coexist with one object-event type. Acceptable graphs must mention all and
only the supplied names and connect them through relation edges or OBJECT.
UNSPECIFIED is excluded from connectivity. These are pilot assumptions.

**Abstention.** Accept the best complete graph only when its cost <=0.65 and it
beats the second-best complete graph by >=0.10. A sole candidate needs only the
cost check. Otherwise return no edges. These scores are not probabilities.

## Comparisons and controls

- Primary: all mappings, participant completeness, and training lexical evidence.
- Single mapping: inherited monotonic sequence alignment, same new cost cutoff
  and composer. Distances differ in definition; this is a method comparison,
  not an isolated measurement of one parameter.
- Partial allowed: same composer without connectivity/name completeness checks.
- No lexical evidence: primary with lexical support set to zero.
- Old local alignment and entity-count majority: inherited predictors, each
  trained on the same 30 records as the primary.
- Development diagnostic: retrain new model on the original 22 records and
  inspect candidate coverage/predictions for the old eight. Not test evidence.
- Translation null: 99 fixed-seed (240926) shuffles of entire training graphs
  within participant-count strata. Refit lexical evidence and rerun composition
  and abstention. Cache only label-independent entity mappings. Compare primary
  edge F1 to null F1 with `(1 + count(null >= observed))/100`.

## Frozen evaluation and continuation rule

```text
Audit the 16 inscriptions and test the implementation with synthetic cases
Freeze code, tests, protocol, manifests, prior evidence, and source hashes
Run every fixed model on public test inputs
Score accepted edges against the 23 reference edges and 16 complete graphs
Run the 99 shuffled-translation controls
Apply every continuation criterion without tuning on these outcomes
Preserve the earlier frozen experiments
```

Report edge precision/recall/F1, document coverage, exact-graph accuracy and
cohort recalls. Abstentions count as misses. A correct relation with incorrect
participants is wrong. Candidate presence alone is not a correct prediction.

All continuation criteria must pass: precision >=80%, edge recall >=60%,
coverage >=50%, exact-graph accuracy >=50%, edge F1 >=15 percentage points above
both old-local and count-only baselines, and translation-null p <=0.05. Passing
would justify another source-controlled validation, not publishing unknown-word
glosses. Failing ends this frozen run; diagnostics do not authorise retuning it.

The test is small, hand-selected and from one source. No confidence interval or
p-value removes those limits. Earlier frozen code and evidence remain unchanged.
