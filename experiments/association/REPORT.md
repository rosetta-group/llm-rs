# Text–image association pilot

Existing visual descriptions provide a narrow test outside text-only statistics.
This pilot did not establish a text–image association.

**Endpoint:** explicit light versus dark root descriptions; silence about a part is unknown.
**Balanced accuracy:** mean accuracy across light and dark classes; 50% is the trivial baseline.
**Grouped evaluation:** each fold holds out an entire physical folio, including its page sides/panels.

```text
Audit existing descriptions before inspecting text associations
Exclude uncertain labels, absent mentions, duplicate readings, and non-training folios
Freeze endpoint, controls, model, and tests
Hold out each folio; compare controls with controls plus EVA label text
Shuffle annotation labels within pages 999 times; refit the same models
Report the observed gain against the controlled null
```

![Image association](../method-benchmark/figures/image-association.png)

1. **Source and eligibility.** Grove/Stolfi's 1998 catalogue has 249 deduplicated plant-related
   objects. Root-versus-plant class is concentrated on folio 99 and cannot support the
   proposed comparison. Leaf/flower coloration has too few eligible examples. The
   root-color endpoint retains 59 confident plant/root labels: 27 light and 32 dark,
   across folios 88, 89, 99, 100, 101, and 102. All have the same pharmaceutical section
   and inherited hand label 1. Historical f101v subdivisions share one folio group.
2. **Controls.** Label length, word count, relative label index, location group, and
   transcriber. The text model adds fixed-hash EVA character 1–3-grams. Ridge penalty 10
   and decision threshold 0.5 were fixed before evaluation. Standardization fits only
   each training fold. The cached linear prediction operator is mathematically equivalent
   to refitting ridge for every permutation; held-out target values have zero influence.
3. **Result.** Controls score 59.43%; adding text scores
   60.88%. The gain is 1.45 percentage points,
   p=0.348, paired folio-bootstrap interval
   [-6.81, 20.83] points. The predeclared promising-result gate fails.
   This is a negative pilot, not evidence that images and writing are unrelated.
4. **Limits and remaining study.** Six folios are few; folio 99 supplies only one eligible
   example. Descriptions are selective and the original annotators could see the text.
   Within-page permutations preserve page composition, not every layout effect. This is
   pharmaceutical-label coverage, not the planned larger herbal-page study. Independent
   visual annotation of crops with text hidden, explicit absent/unknown labels, annotation
   agreement, and a fresh held-out sample are still required. Do not infer plant species
   or word meanings from these results.

Source: [Grove/Stolfi catalogue and format](https://www.ic.unicamp.br/~stolfi/EXPORT/voynich/98-02-01-lotsa-labels/),
[annotation index](https://www.ic.unicamp.br/~stolfi/EXPORT/voynich/98-02-01-lotsa-labels/labtit-98-07-20.idx).
No explicit reuse license was found; raw annotations remain local. The repository stores
retrieval code, source hash, aggregate results, and attribution, not the annotation text.

Audit: [fixed protocol](PROTOCOL.md), [freeze/source hash](freeze.json),
[aggregate and per-folio results](results.json). No Voynich final-test scores were opened.
