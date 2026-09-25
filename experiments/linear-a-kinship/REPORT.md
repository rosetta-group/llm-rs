# Parentage feasibility: the unnamed-relative pattern

Parentage formulas are a plausible structural research target, but their participants need
not both be named. The useful transfer from the Etruscan work is relation structure with
explicit input assumptions, not an already established Linear A word for son or daughter.

**Named edge:** a stated relation between two identified people, such as Alektruon–Eteokles.
**Unnamed relative:** a counted participant designated by kinship, such as “X and daughter”.
**Graphic slot:** a divided sign run; neither its status as a name nor its meaning is known.

## What was done

- Inspected the concurrent Etruscan scaffolding prototype read-only. It receives manual
  person/deity spans and known name case/gender, then transfers relation templates.
- Annotated five known-language development fixtures on three objects, with positive
  parentage, ordinary pairing and collective-children cases. Two daughter examples share
  one tablet and are not independent positive objects. [Controls](controls.json).
- Froze literal Linear B searches and an ordered Linear A shape inventory at `1c3dd37`
  before running either inventory. Preserved order and repeated slots; no semantic scoring.

## Why

A two-name detector would miss an unnamed daughter and could mistake two ordinary names
for relatives. Known-language examples expose those errors before any unknown sign is labelled.

## Mechanics and concrete controls

```text
Supply independently justified person spans in the known-language controls
Represent either named CHILD_OF(child, parent) or unnamed-relative(parent, role)
Keep ordinary pairs and collective children as different outcomes
Test whole objects with names hidden and matched relation negatives
Only then apply a validated method to source-checked Linear A slots
```

1. **Son/patronymic, including line wrapping.** PY An654 .8–.9 has
   `a-re-ku-tu-ru-wo e-te-wo-ke-re-we / i-jo`. Chadwick interprets the complete expression
   as Alektruon, son of Eteokles; his patronymic transcription joins what DĀMOS prints
   across lines. We must test the whole expression, not assume standalone `i-jo` means
   “son” in every context. [Chadwick 1960, p.60](https://biblioteca-digitala.ro/reviste/StudiiClasice/02-revista-studii-clasice-II-1960-67-72.pdf).
2. **Daughter as a substituted name.** MY V659 .5 and .6 replace the second personal name
   with `tu-ka-te-qe`; each entry counts two people. The parent is named, the daughter is
   not. The same tablet's ordinary two-name pairs supply a close negative. KN Ak624's
   child categories supply a separate group-level negative. Abbreviated Knossos `tu`
   should remain a distinct, less certain interpretation. [Killen, pp.208–212](https://antiquitasviva.com/wp-content/uploads/2021/05/16.1-2.22.-killen-j.-t.-the-abbrevitation-tu-on-knossos-woman-tablets.pdf).
3. **Etruscan inputs do not transfer automatically.** `public_view` in the inspected
   `etruscan/scaffolding.py` uses supplied entity spans and name morphology. It is an
   untracked concurrent prototype, not an established result. The preceding repair report
   records KIN precision 8/84 (9.5%) for the forced classifier, and no accepted KIN calls
   for the abstaining models. None validates Linear A names, case or gender.
4. **A promising restriction has very little current coverage.** The frozen scan finds
   48 graphic shapes on 22 objects: 30 single-word + 1, 16 single-word + 2, and two
   two-word + 1 rows. No two-word + 2 or three-word shapes survive this strict scan.
   Both two-slot hits contain a non-name sign in the record's own role annotations:
   HT63 begins A305 + AB04, both transactions; KH6's last slot A306 is a logogram.
   These are parser ambiguities, not parentage candidates. [Shape output](linear-a-shapes.json),
   [role review](role-review.json).

## Coverage and limits

| Literal Linear B form | Occurrences | Objects |
|---|---:|---:|
| i-jo | 4 | 2 |
| i-jo-qe | 1 | 1 |
| i-je-we | 1 | 1 |
| tu-ka-te-qe | 2 | 1 |
| tu-ka-te-re | 1 | 1 |
| ko-wo | 187 | 151 |
| ko-wa | 113 | 100 |

Exact bare `u-jo` and `tu-ka-te` have zero hits in this literal search. These are not lemma
frequencies: damaged forms and attached forms are not normalised, and i-jo may participate
in a patronymic. The full [inventory](linear-b-markers.json) retains raw lines and multiplicity.
The plentiful ko-wo/ko-wa group categories cannot substitute for independently named parentage.

The Linear A inventory rejects 1,395 non-tablets, 204 conflicted/uncertain tablet records and
73 tablets without a sign layer; 212 tablet records remain eligible for line scanning.
Aggregate sign roles are only a parsing aid. The two audited false hits demonstrate why
per-occurrence role review is required before treating graphic slots as words or people.
This scan cannot detect line-wrapped formulas, attached patronymics, formulas without a
numeral or records excluded for damage. Zero plausible hits here is not evidence of no kinship.

HT117a remains a useful development lead: it contains many word + 1 entries, but the pinned
record explicitly disagrees between `te-*56-re` and glyph `te-ja-re`. It was excluded rather
than silently reconciled. The [Davis–Valério chapter's institutional abstract](https://portalinvestigacion.um.es/documentos/69ca489802adbc22ca0c8b13?lang=en)
identifies 1-lists as the basis of an anthroponym hypothesis; the full chapter was not obtained.
A numeral 1 does not itself certify a personal name. HT85a also deserves contextual review,
but neither tablet presently supplies labelled parent-child edges for this test.

## Decision

Pursue **name-slot substitution** before assigning “son” or “daughter”: look for a recurrent
short designation taking the place of a second name in otherwise comparable lists. Preserve
named and unnamed relations separately, and test ownership/occupation/location as rival
interpretations. Parent/child gender is a later hypothesis, not an input guessed from endings.

The concrete next dependency is a small independently reviewed person/designation inventory
for Linear A, starting with HT85/117 and resolving the HT117 source discrepancy, plus more
independent Linear B parentage controls. The current five fixtures and graphic scan are
insufficient for a held-out recovery claim. No Linear A kinship meaning was assigned and no
Etruscan files were changed. Earlier failed controls remain failed.

## Records and reproduction

[Protocol](PROTOCOL.md), [controls](controls.json), [sources and Etruscan review hashes](sources.json),
[freeze](freeze.json). Derived SigLA/DĀMOS records retain CC BY-NC-SA 4.0 attribution.
The source scan and PDFs are local reference material; no paper is redistributed in git.

```sh
.venv/bin/python -m experiments.linear_a_kinship verify
.venv/bin/python -m unittest tests.test_linear_a_kinship
# In a fresh checkout without generated JSON outputs:
.venv/bin/python -m experiments.linear_a_kinship run
```
