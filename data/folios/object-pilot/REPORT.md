# A 24-panel object-and-relation pilot

**What was done**

- Inspected 24 panel crops covering plant-like forms, people, animals, vessels,
  circular diagrams, star-like marks, dense writing and sparse marks. Enlarged
  f76v and f57v to check details; checked the evidence overlay on f81r.
- Recorded object groups, evidence rectangles, conservative count bounds,
  uncertainty and spatial/drawn relations in `observations.json`.
- Added a deterministic description script with multi-label domains and compound
  relation rules. It writes a CSV, JSONL, evidence gallery and provenance registry.
- Added blank independent-review forms and agreement scoring. No human annotations,
  reviewed text masks or agreement results exist yet.

**Why**

The pixel baseline cannot distinguish a round leaf from a diagram, or green plants
from green basins. Object descriptions and relations provide testable image-side
features without inventing word meanings.

## Outputs

[Gallery](../analyses/object_relation_v1_b9cbf0c3a6dc/index.html) ·
[CSV](../analyses/object_relation_v1_b9cbf0c3a6dc/descriptions.csv) ·
[JSONL](../analyses/object_relation_v1_b9cbf0c3a6dc/descriptions.jsonl) ·
[Registry](../analyses/object_relation_v1_b9cbf0c3a6dc/analysis.json) ·
[Rubric](RUBRIC.md)

Analysis ID: `object_relation_v1_b9cbf0c3a6dc`.
All rows are `single_ai_development_review`, `text_masked=false`. Every table row
has the analysis ID, image checksum, panel crop and physical-group identifier.
The semantic observations are supplied by one AI reviewer; the script processes
those observations. It is **not** an automatic image-to-object detector.

## What the rules record

| Domain tag | Panels | Interpretation |
|---|---:|---|
| Botanical | 9 | Whole plants or detached botanical forms |
| Human figures | 7 | Includes one tentative assignment on f57v |
| Animal figures | 2 | Central quadrupeds; no species inferred |
| Containers | 7 | Includes tentative tub-like forms on f76v |
| Basins/channels | 4 | Includes tentative paths on the Rosettes |
| Diagrammatic | 7 | Circular or radial structures |
| Celestial-like | 5 | A visual combination of diagrams and stars/faces |
| Text-dominant | 3 | Illustrated marks may still be present |
| Sparse marks | 1 | f116v |

Tags overlap. Counts describe this purposive pilot, not manuscript prevalence.
`tentative_domains` and `tentative_patterns` identify outputs supported only by
uncertain observations. Absence of an annotation is not a verified object absence.

| Compound pattern | Panels | Concrete example |
|---|---:|---|
| Figures in linked basins | 2 | f78r, f81r: occupied upper/lower basins share a channel |
| Vessels beside botanical groups | 5 | f88r: three left-column vessels beside roots/leaves |
| Animal in circular diagram | 2 | f70v1, f72r1: a central quadruped within rings |
| Linked circular structures | 1, tentative | Rosettes: major circular nodes joined by drawn paths |

## Checks against the pixel baseline

- **f2v:** the round green form is annotated as a leaf in a whole plant; no
  diagrammatic domain is emitted just because it is round.
- **f57v:** visually evident faint rings are recorded despite the pixel detector's
  miss. Four figure-like forms remain marked tentative in this development record.
- **f58r / f103r:** marginal star-like marks do not imply a celestial diagram.
- **f1r:** faint green show-through is uncertain_form, not foreground botanical.
- **fRos:** nine major diagram nodes are recorded by inspection; the pixel method
  found three circle candidates. These count different objects, so this is not an
  accuracy comparison or evidence of an automated counting improvement.

## Reproduce and review

```sh
.venv/bin/python -m experiments.describe_folio_objects
.venv/bin/python -m unittest tests.test_folio_objects tests.test_folio_description
```

A repeat run verifies identical outputs; changed code, observations or rubric
create a new analysis ID. The original 228-panel pixel archive is unchanged.
Images shown in the gallery include evidence-box overlays for checking coordinates.

Before independent annotation, a coordinator must mask writing and visually check
that masks do not erase the objects being scored. The two template files have
separate panel orders and no pilot predictions, but the current output directory
also contains a coordinator key and the unmasked gallery: **do not send this whole
folder as a blinded packet**. No packet is declared ready yet.

After two independently completed human forms are available:

```sh
.venv/bin/python -m experiments.folio_agreement reviewer-A.json reviewer-B.json --out agreement.json
```

The scorer checks declared independence, distinct reviewer IDs, complete rubric
fields and a shared reviewed-mask manifest hash. It reports raw, positive and
chance-adjusted agreement per feature, retaining missingness; constant-label
kappa is undefined. Metadata cannot prove that reviewers worked independently.
The scaffold currently scores domain and relation-pattern presence; object boxes
and count bounds still require separate review. No accuracy, agreement or text
association has been estimated from this single-observer pilot.
