# Etruscan: closed research record

This track tested word-class prediction and recovery of known relationships from Etruscan inscriptions.
**Closed on 2026-09-24:** the tested methods do not justify new word meanings or reliable complex interpretations; no further fitting is scheduled.

**Balanced accuracy:** mean recall across word classes, giving rare and common classes equal weight.
**Precision / recall:** correct accepted relationships divided by all accepted / all reference relationships.
**Exact graph:** every reference relationship for one inscription, with no additions or omissions.
**Frozen gate:** success criteria fixed before an evaluation, rather than adjusted to its outcome.

## What was done

- Completed six modeling stages, from formula context to clause-level relationship recovery.
- Repaired a dictionary-loading defect and separated candidate-generation errors from ranking errors.
- Preserved protocols, inputs, predictions, controls, failed gates and later diagnostics in separate records.
- Audited nine exposed inscriptions against six publications, with cited spans and explicit uncertainty.
- Closed the track at the user's request. Unresolved research questions below are recorded conditions for reopening, not queued work.

## Why it was done

The last model recovered only 4/20 complete interpretations and none of eight mixed cases, despite having all 36 reference relationships among its generated candidates. Another fit to the same exposed examples would not establish generalisation, and the source audit found reference ambiguities that need review first.

## Results in one place

These stages use different samples and metrics; their percentages are not a common leaderboard.

| Stage | Evidence | Decision and record |
|---|---|---|
| 1. Formula context | Etruscan balanced accuracy 44.1%, against 20% chance; size-matched Latin 43.9%. Etruscan KIN and LIFE precision only 6% each. | Narrow internal gate pass, insufficient for individual rare-word claims. [Report](../experiments/etruscan/REPORT.md), [protocol](../experiments/etruscan/PROTOCOL.md). |
| 2. Fresh-word test | Deduplicated internal score 45.0%; nominally fresh 49-word score 28.1%, against 25% chance. Neither shuffle comparison was significant under its gate. | Failed. Later loader audit qualifies which words were genuinely fresh. [Report](../experiments/etruscan-fresh/REPORT.md), [results](../experiments/etruscan-fresh/results.json). |
| 3. Repair and abstention | Context plus endings: 164/192 accepted calls correct, 85.4% precision at 45.0% coverage. Accepted labels: 177 NAME and 15 OTHER; no KIN, NUM or LIFE calls. | Both methods failed their continuation gate. [Audit](../experiments/etruscan-repair/AUDIT.md), [report](../experiments/etruscan-repair/REPORT.md), [results](../experiments/etruscan-repair/results.json). |
| 4. Names as scaffolding | 22 training inscriptions, eight hidden cases. Joint model accepted none; local alignment recovered one target relationship. | Failed. Some correct parent pairings were absent from candidate generation. [Report](../experiments/etruscan-scaffolding/REPORT.md), [results](../experiments/etruscan-scaffolding/results.json). |
| 5. Complete participant graphs | 30 training inscriptions, 16 different tests. 10/16 exact graphs; 11/11 accepted relationships correct, but only 11/23 recovered (47.8%). All reference graphs representable. | Useful limited recovery; failed the 60% recall gate. [Report](../experiments/etruscan-graphs/REPORT.md), [diagnostics](../experiments/etruscan-graphs/diagnostics.json). |
| 6. Clause coverage | 46 training inscriptions, 20 different tests. 4/20 exact graphs; 4/8 accepted relationships correct; 4/36 recovered (11.1%); 0/8 mixed graphs. Name-count baseline: 19/26 correct, 19/36 recovered. | Failed nine of ten criteria. Stop local tuning. [Report](../experiments/etruscan-clauses/REPORT.md), [results](../experiments/etruscan-clauses/results.json), [diagnostics](../experiments/etruscan-clauses/diagnostics.json). |
| Source evidence audit | Nine exposed inscriptions: three corroborated, one partly corroborated, three disputed, two unverified. Six relationship claims have source support; all nine await expert review. | Data qualification only; no new score or revised historical score. [Report](../experiments/etruscan-evidence/REPORT.md), [annotations](../experiments/etruscan-evidence/annotations.json). |

The participant-graph and clause models both beat all 99 shuffled-translation controls under their respective statistics (p=0.01). This does not override their failed accuracy gates. The earlier 10/16 result and later 4/20 result concern different samples; the later report also evaluates the previous model on the same harder cases, where it obtains 4/20.

## What the failures taught us

1. **Missing labels were partly a software defect.** The old loader retained the first spelling row even when its gloss was empty. Later usable entries, including `lautni`, were lost. Round two therefore did not establish that all its nominally fresh words lacked ETP glosses, or that low labelled-neighbour coverage was entirely a scholarly coverage problem. The repair is separately versioned; the original experiment remains unchanged.
2. **The Participant-Coverage Trap.** Explaining every named person does not explain every statement. In Cr 5.3, parentage accounts for both people while omitting tomb construction. Retaining alternative name mappings fixed missing pairings, but did not solve this omission.
3. **Confound amplification.** Requiring learned token associations to be explained can reinforce the wrong interpretation. In Ta 1.168, age/death context acquires parentage associations and outweighs the wife construction. In Cr 5.2, recognising construction does not distinguish the builders from their father. These are role-binding failures, not merely insufficient search width.
4. **Coarse representations and uncertain references limit interpretation.** The three-letter representation merges `mene` with `men`. Supplied name anchors and reference relationships also need qualification in AV 6.1, Vt 1.58 and Cr 3.18. The evidence audit records those problems separately rather than changing old answers to improve scores.

## Evidence audit and open questions

The [audit's cited case table](../experiments/etruscan-evidence/REPORT.md#findings) gives the publication, page and annotation consequence for each inscription. The machine-readable ledger preserves original tokens, name anchors and historical graphs alongside new claims.

| Case | State at closure | Remaining question |
|---|---|---|
| Ta 1.168 | Spouse relationship corroborated; age/death/children separated | Exact age reading remains uncertain. |
| Cr 5.2 | Two parentage and two construction relationships corroborated | `MADE` includes commissioning; the source extends beyond the dataset excerpt. |
| Cr 3.20 | Donor and recipient corroborated | This active formula does not settle every gift formula. |
| Ve 3.2 | Pronoun and gift construction partly corroborated | Complete two-donor role assignment still needs review. |
| Cr 3.18 | Alternative donor/recipient graphs retained | Historical discussion records ambiguity; it does not itself settle current consensus. |
| Vt 1.58 | Name formula disputed | Name readings and participant assignments remain unresolved. |
| AV 6.1 | Participant anchor disputed | Family name versus another participant remains unresolved. |
| Ru 5.1; Vs 1.28 | Unverified | No passage settling the complete role graphs was found in this bounded search. |

Separate publications can inherit the same editions and analyses. Their agreement is not independent archaeological verification. Codex made the span and role mappings; no external expert adjudicated this ledger. Source corroboration therefore does not certify every detail or validate a new model.

## Closure boundary

```text
Keep existing results, source hashes and disclosed cases
Stop model tuning and unknown-word predictions
Leave the nine-record audit pending expert review
Reopen only with reviewed evidence, a distinct hypothesis and a separately reserved test
Freeze the new protocol before scoring that test
```

1. **What is established.** Some familiar gift, ownership and making relationships can be transferred between selected inscriptions with manually supplied names: the participant-graph pilot recovered 10/16 complete graphs. The subsequent complex-case test limits how far that result generalises.
2. **What is not established.** No new Etruscan meaning, general translator, reliable complex parser or unknown-word prediction is validated. Failure of these implementations does not show that all computational Etruscan research is exhausted. Neither more model capacity nor another threshold sweep has been justified by these results.
3. **What could justify reopening.** Reviewed word divisions, predicate spans, role bindings and explicit restoration uncertainty must support a concrete new hypothesis. Reserve different monuments with independently checked answers. All 66 already exposed monuments remain development material; the audit creates zero fresh evaluation cases.
4. **What closure changes.** The scope and repository indexes now mark Etruscan closed. Historical reports retain their original prospective language because they are dated evidence. No further run, expert outreach, publication, cloud job or unknown-word prediction is scheduled by this closure.

## Preservation and verification

The closure record describes the state on 2026-09-24, when the later work was still uncommitted.
On 2026-09-25, that complete working state was preserved in commit `c9d787e` on `etruscan`
and integrated into the canonical `master` branch at the user’s request. The historical
closure JSON and experiment results remain unchanged. Ignored source downloads stay local;
Git contains their recorded URLs and hashes, not the research PDF copies.

**Round-one reproduction caveat:** its frozen `etruscan/corpus.py` is the version in
commit `18945f1f32`. Later work changed that shared module before this integration.
The five later freezes verify against the integrated tree; reproducing round one
requires its historical checkout. This is an existing versioning limitation, not a
new mismatch introduced by the merge. The original closure document hashes can be
checked against `c9d787e`, rather than against the merged and updated index documents.

| Material | Location |
|---|---|
| Original scope and source catalog | [SCOPE.md](../experiments/etruscan/SCOPE.md), [sources.json](../experiments/etruscan/sources.json) |
| Modeling code and tests | `etruscan/`, `experiments/etruscan_*.py`, `tests/test_etruscan*.py` |
| Stage protocols, manifests, outputs and freezes | The six linked experiment folders above |
| Evidence catalog and PDF byte hashes | [annotations.json](../experiments/etruscan-evidence/annotations.json) |
| Audit code/data snapshot | [provenance.json](../experiments/etruscan-evidence/provenance.json); post-audit, not preregistration |
| Source downloads, ignored by Git | `artifacts/etruscan-sources/`; other source paths are recorded in the experiment freezes |
| Closure verification | [closure.json](../experiments/etruscan-closure/closure.json) |

At closure, 44 Etruscan unit tests pass. All 91 file/source hash entries across the five historical freezes match; the six audit snapshot files and five historical-freeze hashes also match. These checks establish implementation integrity, not linguistic truth. No model was retrained or regraded for closure.

From the repository root, read-only checks are:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_etruscan*.py'
.venv/bin/python -m experiments.etruscan_evidence verify
git diff --check
```

Full reproduction instructions remain in each stage's protocol/report. Generated outputs refuse overwrite. Reproduce in a separate copy with the pinned sources, following that stage's instructions; do not delete or replace the archived results. The five audit PDFs are ignored local research copies, so a Git checkout alone cannot run their byte verification. Their URLs and hashes are retained; the Met web page has no pinned local snapshot.
