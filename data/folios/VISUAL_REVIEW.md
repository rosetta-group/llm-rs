# Direct image review — 2026-09-21

Codex inspected contact sheets covering all 213 downloaded Yale images, a separate
sheet of all 39 proposed foldout crops, and enlarged scans of f1r, f75r, f81r, f88r
and the Rosettes foldout. This was a visual development review, not an independent
annotation or a reading of the manuscript text.

## Observations

| Example | Visible content | Consequence for descriptions |
|---|---|---|
| f1r | Dense writing, red marks, faint green forms showing through. | Colour alone can describe the reverse side rather than the intended illustration. |
| f1v, f2v | Large plant-like forms; f2v has a broad green round leaf-like form. | A rounded green form is not enough to infer a celestial diagram. |
| f57v | Faint concentric circular arrangement with writing. | Weak, irregular rings can evade a conservative circle detector. |
| f67r1, f68v1 | Radial coloured motifs surrounded by circular writing. | Geometry and colour offer separable description axes. |
| f72r1 | Central red animal-like figure within concentric arrangements. | One panel can mix figurative content and circular structure. |
| f75r | Human figures within green channels/pools; text surrounds the scene. | Green does not mean botanical. Connectivity, figures and containment need richer annotations. |
| f81r | Two horizontal groups of figures in green basins, joined by a curved channel. | Relations such as connected-to and inside are more informative than colour alone; v1 does not recover them. |
| f88r | Decorated vessel-like forms down the left; detached roots/leaves arranged in rows beside text. | A future description should distinguish objects and their relative placement. |
| fRos | Several large circular structures, connecting paths and smaller motifs across the foldout. | Preserve the complete foldout and identify scale limits before counting fine details. |
| f103r–f116r | Mainly dense writing with repeated marginal marks. | Regular horizontal marks need a separate layout category. |
| f116v | Sparse writing and small sketches/marks. | Sparse content should not force a plant/people/cosmos label. |

These examples establish breadth of the inspected imagery, not botanical species,
astronomical identities, functions of vessels, or the meanings of any words.

## Development checks

The frozen `pixel_layout_v1_16d1ceb438d2` run contains 228 rows: 95 localised-colour,
59 distributed-colour, 56 horizontal-mark and 18 circle-candidate layouts. Eight
panel crops are under 500 pixels on their shorter side and carry a resolution flag.
These counts describe rule outputs, not counts of botanical or celestial subjects.

Known limitations in this run: f57v's faint rings are missed and its layout becomes
`horizontal_marks_dominant`; f2v's round plant form receives a circle candidate.
The Rosettes foldout produces only three circle candidates, despite visibly having
more circular structures. The table therefore cannot replace semantic annotation
or provide reliable counts of illustrated objects.

The first unfiltered Hough detector found ten false circle candidates on f1r and
six on f103r. Radial-edge support was added after this failure. f103r is now a
regression fixture. Synthetic checks cover blank parchment, separated colours,
spatial direction, a drawn circle and straight text-like rows. These are mechanism
checks; they are not a manuscript-wide precision/recall estimate.

All 39 registered panel crops were visually checked against the foldout contact
sheet. Registration found 35–913 supporting feature matches per crop. The matching
thumbnails and match counts are archived. This resolves panel identity, not exact
parchment boundaries; small overlaps at folds remain possible.

## Next description method

Use an explicit object-and-relation rubric: visible people, whole plants, detached
plant parts, containers, star-like marks, circular structures and uncertain forms;
then containment, connection, repetition and relative position. Attach evidence
regions and uncertainty to every claim. Compare it with this pixel-only baseline
under a new `analysis_id`. Before testing text/meaning associations, obtain masked,
independent annotations and agreement measurements as the research plan requires.
