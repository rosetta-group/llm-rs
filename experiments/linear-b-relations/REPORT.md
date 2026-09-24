# Relational control: more son evidence, no validated parentage recovery

This control compares family references with service, occupation and ordinary co-listing in Linear B.
Repeated non-name forms carry some transferable information, but neither the source labels nor the classifier justify automatic parent-child edges.

**Recognition–binding split:** recognizing a family expression and assigning its child/parent are different tasks.
**Onomastic slot:** a supplied personal name or name-derived expression, with spelling and grammatical role hidden.
**Balanced recall:** equal weight to classes, then objects within class, then their cases; abstentions count as failures.

## What was done

- Expanded the source review with Hiller (1989), Duhoux (2007) and Godart (2024), retaining
  competing interpretations and original DĀMOS text. [Sources and access limits](sources.json).
- Annotated **19 scored expressions on 16 physical objects**: seven FAMILY cases on six objects,
  plus 12 OTHER cases covering five service, four occupation and three ordinary-pair objects.
  Seven unresolved review cases on seven objects remain outside scoring; one of those objects
  also contributes a separately supported ordinary-pair fragment. [Inventory](INVENTORY.md).
- Separated [anonymous inputs](public.json) and [gold labels](labels.json). Supplied name spans,
  selected fragments, cross-line joins and terminal `-qe` segmentation are substantial hints.
  No Greek gloss, grammatical case, gender or parent/child roles reach the classifier.
- Froze code, inputs and all **5,932 corpus snapshots** at **`095dbdc`** before scoring.
  Ran whole-object validation, 199 block-label negatives and an expanded literal inventory.
  Passed **66 focused tests**; all **ten follow-up freezes** verify.

## Why it was done

The preceding layout control could not separate names from designations. Relational expressions
add repeated words and linked slots, but service clauses and family clauses can share those forms
of arrangement, so they require explicit rivals and separate checks on argument direction.

```text
Retain source-supported fragments and quarantine disputed labels
Mask names; preserve non-name equality and supplied coordination
Hold out all cases from one tablet, including both family and rival examples
Require two other tablets and 90% agreement for an exact feature signature
Score recognition and abstentions; report source direction limits separately
Export no predicted family edges
```

## Results

| Fixed input | Balanced recall | Coverage | Family recall | Accuracy among calls | Correct / wrong / abstained |
|---|---:|---:|---:|---:|---:|
| Anonymous ordered frame | 0.0% | 0.0% | 0.0% | undefined | 0 / 0 / 19 |
| Repeated non-name forms, order ignored (diagnostic) | 41.7% | 50.0% | 50.0% | 83.3% | 7 / 1 / 11 |
| **Ordered frame + form identity (primary)** | **0.0%** | **0.0%** | **0.0%** | **undefined** | **0 / 0 / 19** |
| Primary requirement | >=90% | >=90% | >=90% | >=95% | false-family rate <=5% |

All percentages use class/object weighting; raw diagnostic accuracy is 7/8 = 87.5%.
[Full results](results.json) retain every prediction, supporting object and negative assignment.
[Readable predictions](PREDICTIONS.md) include source fragments and all three arms.

1. **Repeated forms help within one marker family.** With name spelling hidden, the diagnostic
   identifies the `i-*65` expressions on PY Ae344, Aq218 and Jn725 from the other two objects.
   It also recognizes four `do-e-ro` service fragments as OTHER. The two daughter forms have
   no supporting objects with the same form, and the patronymic becomes a false OTHER call.
   These are supervised within-language transfers, not discovery of a word's meaning.
2. **Exact frames lack independent support.** The primary arm has 11 unseen signatures and
   eight cases with only one supporting object; none meets the fixed two-object minimum.
   This is a coverage failure. The fixture meets the broad class/object preflight, but those
   counts do not guarantee repetition of each exact construction. Do not lower the support
   threshold after observing this result.
3. **The negative gate is uninformative here.** Zero of 196 evaluable label-swap runs pass;
   three of 199 lack five objects per class. Every evaluable primary negative also abstains
   everywhere, because label swaps cannot create missing structural support. This result
   provides no independent reassurance about semantic discrimination.
4. **Recognition leaves binding unresolved.** Among seven supplied FAMILY cases, four record
   an unnamed child (on three objects), two have disputed argument binding, and one is a
   patronymic with an immediate-parent versus wider-lineage qualification. These are source
   annotations, not extraction results. **Zero predicted CHILD_OF edges were exported.**

Empirical in-sample ceilings are **75.0%** for anonymous frames, **91.7%** for form identity
alone and **95.8%** for the combined frame/form signatures. The last number does not measure
transfer: a memorizer could separate most annotated examples, while the held-out model finds
no independently repeated exact pattern. This differs from the previous layout experiment's
84.9% ceiling, which concerned a different task and fixture.

Masking whole name-derived spans also removes patronymic morphology. The patronymic miss
is therefore a limitation of this public input, not a test of a morphology-preserving method.
That richer method would need fresh feature definitions and validation.

## What the source audit changed

The earlier exact-marker inventory omitted the literal forms `i-*65` and `i-*65-qe`. The new
inventory finds three intact occurrences on three objects and two doubtful-text counterparts
on two more objects (Aq64 and Jn431). Those two remain excluded from scoring. Hiller's pp.60–61
supply the five-object comparison; [Duhoux](https://www.researchgate.net/publication/327727893_Le_nom_du_fils_en_lineaire_B_dans_F_LANG_-_C_REINHOLDT_-_J_WEILHARTNER_ed_STEPHANOS_ARISTEIOS_Archaologische_Forschungen_zwischen_Nil_und_Istros_Festschrift_fur_Stefan_Hiller_zum_65_Geburtstag_Vienne_)
also discusses attached forms missed by a bare-word search.

The expanded retrieval yields **59 occurrences: 12 intact exact-form hits, 22 intact attached-tail
candidates and 25 uncertain-text candidates**. These are string-search counts, not a census of
kinship. For example, `si-to-ko-wo` is an occupational label already reviewed in the preceding
control; its `-ko-wo` ending does not license a child relation. `ra-]ke-da-mo-ni-jo-u-jo` on TH Gp227
remains a damaged-name attached-form review case. [Literal inventory](marker-inventory.json).

Three qualifications matter:

- **MY Au102:** `i-jo-qe` can be read as a kin term or a personal name. It is unresolved here.
- **KN Vs1523:** Hiller accepts a motion-participle reading of `i-jo`; Duhoux argues for son.
  The three occurrences stay outside gold and remain one physical object.
- **MY Oe106:** Hiller pp.54–55 takes `o-te-ra` as the daughter in apposition; Martínez Fernández's
  university teaching translation takes it as the parent. Family-reference recognition does
  not resolve that disagreement. The teaching translation documents the rival, not independent
  expert adjudication. [One-page translation](https://campusvirtual.ull.es/ocw/pluginfile.php/6168/mod_resource/content/0/Actividades_y_practicas/Epigrafia_griega/3.1.tabillas_en_lineal_B.pdf).

## Decision and limits

**Gate failed; no Linear A scoring and no directed family-edge recovery.** The repeated-form
diagnostic supplies a limited positive result for one known marker family, while the full
construction control lacks repetition. This is a curated, assisted development study with
source-visible selection, not an unseen benchmark or independent specialist validation.

A useful next source target is the Theban comparison between attached `-u-jo` and the disputed
`-*65`/FAR readings discussed by Duhoux. It requires checking actual sign function, edition
readings and quantities; neither an ending match nor a supplied family label can settle it.
Any larger control should count per-construction held-out support before fitting and preserve
family, service and occupational rivals. No earlier frozen corpus, failed gate or Etruscan file changed.

## Reproduction

[Protocol](PROTOCOL.md), [freeze](freeze.json), [cases](cases.json), [sources](sources.json).
DĀMOS-derived transcriptions retain CC BY-NC-SA 4.0 attribution. Source PDFs remain local
reference material and are not committed. Duhoux was read through the author-uploaded web text;
no local PDF is claimed. The driver refuses existing outputs and added/unpinned corpus items.

```sh
.venv/bin/python -m experiments.linear_b_relations verify
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py'
```

To compare an in-memory rerun with the archive without overwriting either output:

```sh
.venv/bin/python - <<'PY'
import json
from experiments.linear_b_relations import OUT, validate, main
from linear_a import relation_control
main('verify')
rows, labels, cases, records = validate()
assert relation_control.run(rows, labels, cases) == json.loads((OUT / 'results.json').read_text())
assert relation_control.inventory(records) == json.loads((OUT / 'marker-inventory.json').read_text())
print('Exact result and inventory matches; archives unchanged.')
PY
```
