# Next: source-checked account benchmark

Status: proposed follow-on, not run. The integer pilot returned not_evaluable; it did not
reject the accounting approach. No enlargement of the current frozen experiment is implied.

## Concrete deliverable

A versioned set of whole physical accounts with row-level quantities, unit systems, source
references, uncertainty and evaluator-only arithmetic-function annotations. The first seven
review candidates are listed in REPORT.md, but inclusion must be based on legibility and
source-established grouping, never whether a candidate equation balances.

```text
Review development objects against editions and available images
Record numeric certainty separately from damaged names
Document unit conversions and row/face joins with primary citations
Freeze source inclusion and an annotation schema
Acquire separate control objects; keep inspected objects in development
Freeze the revised parser and arithmetic grammar
Check control coverage before generating predictions
```

## Annotation fields

- Stable physical-object ID and all joined/face/source IDs; never split one object across folds.
- Edition/page, DĀMOS permalink, image/drawing reference and retrieval hash where available.
- Raw line and token/region positions; boundaries of names, commodities, units and quantities.
- Separate certainty flags for lexical reading, numeral reading, unit and record completeness.
  A damaged name must not automatically erase a certain quantity; missing numbers stay missing.
- Each quantity as an exact rational value in an explicitly named unit. Record the published
  conversion and alternatives. Never select a conversion because it makes this account balance.
- Block boundaries and any cross-face/tablet joins, with editorial support; alternatives stay
  separate. Blank lines alone must not imply that an account ended or continued.
- Evaluator function label: source-supported sum, subtotal, balance, allocation, ordinary
  entry or unresolved. `to-so`/`to-sa` alone is insufficient. Record the justification and avoid
  selecting only numerically successful cases. Unresolved functions are not negative labels.

## Development and evaluation requirements

First establish whether enough natural known-answer cases exist. The present 33-object target
would need >=33 eligible control objects for identical-size sampling, including >=10 verified
function-bearing objects for the full control. If those do not exist, declare a smaller bounded
question and matched target subset *before* scoring; never duplicate cases to meet a gate.

Keep all examples whose numeric relationships inform implementation in development. Newly
acquired evaluation objects must have disjoint physical/source IDs. Published accounts whose
complete quantities are split over lines or expressed in units are priorities for source review,
not automatically eligible positives. KN Fp 1 is a known mixed-unit discussion example already
seen during scoping, so it belongs in development if it can be annotated, never fresh evaluation.

For a revised model, compare word-triggered predictions against identical arithmetic programs
without words, copy-last, and training-only typical quantities. Preserve commodity/unit
inventories in shuffled controls and repeat the full rule search inside every null draw.
Reserve an allocation-ratio grammar for a separately declared extension; do not infer it from
the current review candidates and grade on those same records.

Primary starting sources: [DĀMOS](https://damos.hf.uio.no/), its per-item bibliography and
[CaLiBRA](https://calibra.classics.cam.ac.uk/) for edition/image references. Linear A fraction
values remain proposals to be explicitly compared, not silently adopted:
[Corazza et al.](https://cris.unibo.it/handle/11585/789546).
