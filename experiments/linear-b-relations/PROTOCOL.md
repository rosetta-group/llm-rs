# Relational frames: family references versus administrative rivals

This curated Linear B development control asks whether source-identified family expressions
can be distinguished from service/dependency, occupational affiliation and ordinary co-listing.
It does not equate family-reference recognition with recovering a directed CHILD_OF edge.

**FAMILY:** an explicit kin term or source-supported patronymic expression; not proof of a named parent-child edge.
**OTHER:** the selected expression asserts service, occupation or co-listing, not an explicit family relation; people may still be related.
**Onomastic slot:** a supplied name or name-derived expression, with spelling and grammatical role hidden.
**Unresolved:** competing readings or doubtful relation-marker text; retained outside scoring.

## Source and annotation scope

Use the pinned DĀMOS corpus and published interpretations (Hiller 1989; Killen 1966;
Duhoux 2007; Ventris & Chadwick 1956; Godart 2024). Preserve all original text. Include
source-supported clauses even with qualified surrounding name readings; require an intact
relation marker for the scoring set. Record uncertainty about direction separately from
recognition. MY Au102 i-jo-qe, KN Vs1523 i-jo, and damaged i-*65 cases are review-only.
Oe106's daughter reference has secure family content but disputed argument binding. Patronymics
support family affiliation but need not identify an immediate parent rather than wider lineage.

The source review and selected cases are visible to the investigator. This is not a sealed,
independently adjudicated or representative benchmark. All earlier experiments remain frozen.
No Linear A or Etruscan file is scored or modified. No hypothetical parent name is generated.

## Inputs and fixed comparison

Supply manually selected relational fragments, onomastic spans, cross-line joins and a
literal terminal -qe segmentation. These are substantial known-language hints. Keep unknown
and damaged quantities out of the classifier. Mask every onomastic expression as N. Represent
all other words by opaque equality IDs, with -qe represented by Q. The spelling-to-ID mapping
carries no Greek gloss, case, gender, site, series or relation label. Parent/child roles and
source interpretations remain evaluator-only. Name replacement therefore cannot affect a fit.
Do not merge i-jo, i-*65, i-je-we or tu-ka-te-re/tu-ka-te; no supplied kinship lemma dictionary.

Three feature sets are fixed before scoring:
- frame: ordered N/W/Q pattern; no word identities.
- marker: sorted set of opaque non-name word IDs, ignoring order and Q. Diagnostic of form reuse.
- frame_marker (primary): ordered N/opaque-word/Q pattern.

For each exact signature, training objects contribute one vote each, divided among their
cases' classes. Require >=2 supporting objects and >=90% agreement, otherwise abstain.
No fallback or threshold tuning. Hold out whole physical objects; merge parallel records of
the same transaction into one leakage group when present. Record all training support. Only
one of the parallel Eb/Ep records for a named servant is included here.

```text
Validate source spans and source-backed labels, retaining unresolved cases outside scoring
Export opaque public features and a separate evaluator key
Freeze and commit code, inputs, source hashes and settings
Hold out each object, fit on other objects, predict the held-out fragments
Measure weighted recall, coverage, accuracy, family recall and false-family calls
Repeat with 199 seeded whole-object label swaps
Report source direction limits separately; export no inferred family edges
```

Equal weight to the two classes, then objects containing each class, then their cases of that
class. Abstentions count as recall failures. Report empirical deterministic ceilings for each
exact signature, explicitly in-sample rather than validation accuracy. Audit all three arms;
only frame_marker determines the gate.

Preflight: >=10 objects, >=5 objects per binary class, and OTHER covers >=3 service objects,
>=3 occupation objects and >=2 ordinary-pair objects. Performance: balanced recall >=90%,
coverage >=90%, conditional accuracy >=95%, family recall >=90%, false FAMILY calls among
OTHER <=5%. In 199 negatives (seed 20260924), swap all labels on each object with probability
1/2; <=5% of runs with >=5 objects per class may pass these performance thresholds. Other
source subclasses are not reinterpreted in the label-swap null. No p-value claim.

A gate pass would justify a larger independently reviewed control, not Linear A transfer.
Even correct FAMILY calls do not license directed parentage unless argument roles and named/
unnamed participants are independently resolved. This round exports zero predicted edges.

## Coverage audit

Also inventory fixed exact forms i-*65, i-*65-qe, i-jo, i-jo-qe, i-je-we, tu-ka-te-qe,
tu-ka-te-re, tu-ka-ta-si and attached tails -u-jo / -ko-wo / -*65 in every pinned item.
Separate intact token matches from damaged-token candidates. Never strip uncertainty to
make a strict hit. Attached tails are candidates, not validated kin terms; e.g. an ending may
be part of a name. PY An654's line-final i-jo belongs to a cross-line patronymic interpretation.
No absence claim about unsearched forms. Snapshot object IDs do not by themselves resolve
all possible catalogue aliases; reviewed scoring objects have explicit identities.
