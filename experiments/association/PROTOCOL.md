# Text–image pilot: explicit root coloration

Frozen before fitting or scoring text associations, 2026-09-21.
This starts the plan's independent-evidence track. It is a limited pharmaceutical-label
pilot, not the larger herbal-page annotation study or a translation result.

## Source and audit

Grove/Stolfi, 1998, [catalogue and format](https://www.ic.unicamp.br/~stolfi/EXPORT/voynich/98-02-01-lotsa-labels/).
Visual descriptions predate our models. They are not plant identities or decoded words.
The original annotators could see the text; their work is not a blinded annotation study.
No explicit reuse license was found in the catalogue pages; keep downloaded raw data
local and publish retrieval code, hashes, aggregate results, and attribution only.

Deduplicate by page/group/label location. Prefer Grove's V reading, then C, L, F;
use stable file order for other ties. Require confident plant/root object class,
a clear EVA label (letters and boundary dots only), and existing training-folio membership.
Map historical f101v1/f101v2 subdivisions to f101v for metadata and folio grouping.
All eligible pharmaceutical pages have hand label 1; section and hand are thus fixed.

The audit found root-versus-whole-plant class confounded with folio 99. That endpoint
is ineligible: require at least 20 examples and four physical folios in each class.
Leaf and flower light/dark endpoints also fail that sample-size gate.
Root coloration passes the preliminary coverage check. This choice used annotation
counts only, without inspecting text associations.

## Primary endpoint

Explicit adjacent phrase `light [coloured] root(s)` versus `dark [coloured] root(s)`
(case insensitive, `colored` also allowed). Exclude descriptions containing both
classes or a question mark, and missing mentions. Absence of a mention is not absence
of a part or a negative colour label. No species or medical-use interpretation.

## Model and controls

- Leave one physical folio out; every fold fits independently.
- Control features: label length, word count, normalized label index within its group,
  location-group indicator, and transcriber indicator. Same section and hand by design.
- Text model adds normalized counts of fixed-hash EVA character 1–3-grams (256 bins).
- Fixed ridge penalty 10, intercept, threshold 0.5; no tuning on this small sample.
- Primary statistic: out-of-fold balanced-accuracy improvement over controls.
- 999 permutations of annotation labels within each page; refit the same fixed model
  for every permutation. Keep page composition and the original held-out folios fixed.
- One eligible primary endpoint. One-sided permutation p=(1+null>=observed)/1000.
- Also report raw balanced accuracy, 2,000 paired folio-bootstrap draws, counts, and
  per-folio scores. Small numbers of physical folios limit generalization.

A promising result requires positive improvement, p<=0.05, and a folio interval above
zero. Even that is only a pilot association: independent visual re-annotation and a
new held-out sample are required before confirmation. Record a negative honestly.
Keep original final-test Voynich pages sealed. No GPU or pretrained image model.
