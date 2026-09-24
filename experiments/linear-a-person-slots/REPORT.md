# HT85/117: role before relation

This audit checks which written entries might represent people before searching for parentage.
It improves the source inventory but finds no convincing “son of” or “daughter of” formula.

**Slot:** one counted entry, potentially a person, group or designation.
**Logogram:** a sign used as a category symbol; its conventional syllabic reading is not a translation.
**Working reading:** the preferred source interpretation for this audit, with alternatives retained.

## What was done

- Read the full Davis–Valério chapter through its author-uploaded text, superseding the
  previous abstract-only access. Inspected GORILA photographs, drawings and transcriptions
  for both faces of HT85/117, and transcriptions of HT87, HT88 and HT122a.
- Built a [43-row inventory](INVENTORY.md) with physical lines, original words and signs,
  quantities, damage, erasures, alternative readings and unknown gender. Froze sources,
  annotations and descriptive code at **74610d3** before generating counts and concordances.
- Reviewed occupation, ownership/responsibility and place affiliation against kinship for
  [four ranked leads](candidate-review.json). All 49 focused tests pass; all seven repair/
  follow-up freezes verify. No meaning-recovery model was fitted.

## Why

The same quantity can accompany a personal name, a designation or a category symbol.
HT85b's eight word entries and three logogram entries would become eleven false “names”
if we ignored per-occurrence roles.

## Results

| Face | Counted word entries | Counted logogram entries | Other rows |
|---|---:|---:|---|
| HT85a | 7 | 0 | header; total 66 |
| HT85b | 8 | 3 | header |
| HT117a | 15 | 0 | two headers; total 10; divider |
| HT117b | 2 | 0 | header |
| **Total: two objects** | **32** | **3** | **8** |

These are source-entry counts, not counts of established personal names. In particular,
HT117's 17 single-unit entries are plausible person/designation slots, not 17 labelled people.

```text
Check source signs and layout
Keep possible people, group labels and logograms separate
Compare repeated headers and entries across different tablets
Require evidence that distinguishes kinship from other affiliations
Abstain when that evidence is absent
```

1. **The HT117 reading now has a defensible preference.** The normalized and diplomatic
   edition draws **AB57 (ja)** in `te-ja-re`, distinct from **AB56** in `ku-*56-nu` on the
   same tablet. The chapter agrees. Live SigLA still encodes `te-*56-re`, so the inventory
   retains that alternative and leaves the frozen corpus unchanged. This is a provisional
   edition-supported decision, not independent expert adjudication; the low-resolution
   photograph and recycled facsimile are not multiple independent votes.
   [GORILA I, p.197](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=233),
   [live SigLA word record](https://sigla.phis.me/document/HT%20117a/index-word-11.html).
2. **Variable counts weaken the automatic-name assumption.** `qa-A310-i` has 1 on HT85b
   and 3 on HT122a; `da-si-*118` has 24 on HT85a and 2 on HT122a. These exact strings need
   no speculative spelling equivalence. A group label fits, but an individual responsible
   for several people or goods also fits. Neither example establishes occupation, ownership,
   place or family membership by itself.
   [GORILA I, p.133](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=169),
   [p.207](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=243).
3. **The strongest structural lead is shared classification.** HT87 combines `qi-tu-ne`
   and `ma-ka-ri-te` in its header and lists `di-ki-se 1`. HT117 distributes those header
   words across its two faces and also lists `di-ki-se 1`. This supplies two physical
   objects with a repeated association. It does not identify a parent or even certify that
   the header is a person's name. An occupation, responsible person, household or locality
   remains possible.
   [GORILA I, p.137](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=173),
   [p.199](https://cefael.efa.gr/detail.php?site_id=1&actionID=page&serie_id=EtCret&volume_number=21&issue_number=1&sp=235).
4. **Unnamed relatives remain possible but unsupported here.** HT85b's `pa`, `ka` and
   `di` logograms are conceivable subordinate designations, each with a separate count.
   There is no observed controlled substitution between a second personal name and a
   known relation designation. Attached parentage also cannot be ruled out from these
   spellings, but no stem/ending contrast identifies it. Assigning “daughter” to one sign
   would add a meaning and gender absent from the evidence.

The [Davis–Valério study](https://www.researchgate.net/publication/352343395_Names_and_designations_of_people_in_Linear_A_A_contextual_study_of_tablets_HT_85_and_117)
argues from wider context for likely people in HT117 and possible mixed names/designations
in HT85. We retain those as hypotheses. Its proposed sound correspondences, variant
spellings and administrative reconstruction are not imported as independently verified labels.
Its ordered-list proposal is not a kinship formula or a new significance result.

## Limits and next discriminating check

The manual inventory has not had independent epigraphic review. Glyph clarity, a published
reading and semantic certainty are different things. The [source review](source-review.json)
also preserves the HT85b gap, overwritten signs, physical-versus-logical line distinctions,
incorrect page metadata and conflicting HT85 scribe metadata.

The [literal concordance](literal-occurrences.json) is a navigation aid. Its object counts
use corpus keys; `PH (?)31a` and `PH 31a` share museum inventory HM1609 and must not become
independent training examples. The [post-run review](CONCORDANCE_REVIEW.md) records that
alias and an additional lead: pinned HT7b has `qi-tu-ne 1`, potentially an entry use of a
word that heads HT87/117. HT7b still needs image collation.

The next bounded check should therefore contrast **header versus counted-entry roles**
for `qi-tu-ne`, beginning with HT7b, and extend the reviewed context set for `di-ki-se`.
That could distinguish a recurring entity label from an administrative category. Parentage
needs a further relation-specific contrast; the present audit supplies none. Earlier failed
correspondence and structural gates remain failed. All inspected material stays development.

## Records and reproduction

[Protocol](PROTOCOL.md), [machine-readable inventory](inventory.json), [sources](sources.json),
[freeze](freeze.json), [summary](results.json). SigLA-derived annotations retain attribution
to Ester Salgarella and Simon Castellan and CC BY-NC-SA 4.0. GORILA scans and the copyrighted
chapter are not redistributed in git; the chapter was read through web extraction, not
downloaded as a PDF. Source snapshots are local and hashed.

```sh
.venv/bin/python -m experiments.linear_a_person_slots verify
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py'
# In a checkout at 74610d3 with pinned local sources attached:
.venv/bin/python -m experiments.linear_a_person_slots run
```

The run refuses to overwrite released outputs. The tests check source preservation, complete
row/sign coverage, erased-sign accounting, logogram/word separation, literal multiplicity
and physical-object grouping for the audited faces; they do not validate a translation.
