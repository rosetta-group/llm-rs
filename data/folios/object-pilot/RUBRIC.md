# Object–Relation v1

A description method for visible objects, their layout and their relations.
This first pilot is **one AI observer's development annotation**, not ground truth.

**Object:** a visible form or group (for example, figures, roots or vessels).
**Relation:** an observed spatial or drawn connection between two annotated groups.
**Evidence box:** a normalized rectangle on the panel crop, not an exact object mask.
**Count bound:** conservative lower and optional upper count; null upper means not counted exhaustively.

```text
Inspect a panel without consulting catalogue descriptions or transcriptions
Mark visible groups, evidence boxes, count bounds and uncertainty
Record drawn connections, containment, parts and relative positions
Derive multiple domain tags and graph patterns with versioned rules
Review independently before testing associations with words
```

## Recording rules

- Vocabulary: whole_plant, plant_part, leaf, root, flower_like, human_figure,
  animal, vessel, basin, channel, circular_structure, radial_motif, face,
  star_like_mark, decorative_mark, uncertain_form.
- Relations: inside, part_of, connected_to, above, left_of, overlaps.
  `inside` refers to depicted containment, not necessarily one evidence rectangle
  fitting inside another. A group box can span several repeated instances.
- Certainty is `clear` or `tentative`, an observer judgement, not a calibrated
  probability. Leave identity/function unknown: no species, organs, medicines,
  named stars, constellations, mythology or translations.
- Whole plants need a visibly integrated plant-like form; roots/leaves among
  detached specimen groups use plant_part. Faint reverse-side show-through is
  uncertain_form, not a foreground botanical detection.
- A vessel suggests a bounded container shape, not a pharmaceutical function.
  Channels are drawn connecting forms; do not infer direction or actual liquid flow.
- Use multiple domain tags. Circular diagrams containing people, animals and stars
  retain all supported tags. Stars in a text margin alone do not imply astronomy.
- Circle counts must declare their unit: major diagram nodes vs nested rings.
  The Rosettes record counts nine major nodes, not every small circle.
- Evidence boxes are deliberately coarse. Object identity, boxes and count bounds
  were manually supplied by this AI review. The script validates and summarizes
  these observations; it **does not automatically recognize objects in pixels**.

## Derived rules

- Plant-like object → botanical; human figure → human_figures; animal → animal_figures.
- Vessel → containers; basin or channel → basins_channels.
- Circular structure or radial motif → diagrammatic.
- Diagrammatic plus a star-like mark or face → celestial_like. This is a visual
  domain tag, not a claim of astronomical content.
- Annotated text-dominant/sparse layout → text_dominant/sparse_marks.
- A channel connected to at least two distinct basins → linked_basins.
- Figures inside such basins → figures_in_linked_basins.
- Vessels left of plant groups → vessels_beside_botanical_groups.
- Animal inside circular structures → animal_in_circular_diagram.
- Circular structures linked to one another → linked_circular_structures.

## Independent review: not completed

Two human annotators must work independently, with manuscript writing masked and
without these predictions or catalogue descriptions. Do not treat two AI passes as
independent human evidence. The generated blank templates are schema scaffolds,
**not a ready-to-release blind packet**: text masks still need manual review, since
writing overlaps figures and diagrams. Record masks separately and archive their
hashes. If masking removes evidence, mark the affected object unassessable rather
than absent. Include the full panel context where feasible.

After mask approval, use anonymous panel IDs in independently shuffled orders;
hide the mapping and pilot predictions. Treat shared physical folios/foldouts as
one group for splitting. f67–f68 belong to one physical foldout, even though the
individual panel names differ; grouping is conservative in the pilot registry.

Before expanding: calculate per-domain raw agreement, Cohen's kappa (undefined
when the marginals are constant), positive agreement, and relation-pattern
agreement. Report prevalence and sample sizes; 24 purposively selected panels
are a pilot, not a precision claim. Inspect disagreements before revising the
rubric, and validate the revised rubric on a separate sample. No text association
or multiple-testing analysis is authorized by merely completing these descriptions.
