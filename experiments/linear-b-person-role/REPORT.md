# Linear B names versus designations: structural control failed

This is a known-language control for the name/category ambiguity exposed by `qi-tu-ne`.
The supplied layout features do not support reliable classification, even on this curated development sample.

**Designation:** an occupational, title, work-group or status expression, without requiring its precise gloss.
**Balanced recall:** correct predictions weighted equally by class, then physical object within class, then its cases; abstentions count as failures.
**Feature ceiling:** the best possible in-sample balanced recall of a deterministic rule using exactly the supplied features.

## What was done

- Annotated **38 cases on 19 physical objects**, with **19 PERSON and 19 DESIGNATION** labels,
  from published interpretations in Ventris & Chadwick (1956), checked against pinned DĀMOS
  text. Personal names occur on 13 objects; designations on 15. Both classes have headings
  and counted entries. See the [source inventory](INVENTORY.md) and [qualified annotations](cases.json).
- Separated the [anonymous structural inputs](public.json) from the [label key](labels.json).
  Supplied entry spans and roles are hints; no word spelling, morphology, site, series,
  translation or gender enters the classifier. Exact word equality is used only to derive
  within-object repetition. The classes and source interpretations are known to the investigator.
- Froze and committed sources, protocol, features, labels, code and tests at **`2574eef`**
  before scoring. Predicted each tablet using only other tablets; all faces share an object key.
- Ran three fixed feature sets, a forced-majority diagnostic and **199 whole-object label-swap
  negatives**. Passed **57 focused tests** and verified all **nine follow-up freezes**.

## Why it was done

HT7b's `qi-tu-ne 1` and HT87/117b's headings cannot by themselves distinguish a person from
a designation. Linear B supplies readable examples of both classes in these positions, so it
can test whether position, quantity, word order and repetition resolve that ambiguity.

```text
Hold out one physical tablet and all its faces
Match its feature signatures only against other tablets
Give each matching training tablet one vote
Predict with at least two tablets and 90% agreement; otherwise abstain
Score all held-out cases, including abstentions
Repeat with whole-tablet label swaps and retain the failed gate
```

## Results

All percentages below use the declared class/object weighting. Raw counts are separate;
for example **15/16 = 93.75% raw accuracy**, while primary weighted conditional accuracy is **95.1%**.

| Fixed feature set | Balanced recall | Coverage | Accuracy among called cases | Raw correct / called | Optimistic feature ceiling |
|---|---:|---:|---:|---:|---:|
| Role | 0.0% | 0.0% | undefined | 0 / 0 | 55.1% |
| Role + quantity | 13.3% | 13.3% | 100.0% | 7 / 7 | 68.5% |
| **Layout: role + quantity + position + repetition** | **37.4%** | **39.4%** | **95.1%** | **15 / 16** | **84.9%** |
| Primary gate | >=90% | >=90% | >=95% | — | — |

The primary method made **one error and 22 abstentions**. Its recall and coverage fail the gate.
The forced-majority layout diagnostic reaches only **57.4% balanced recall**, **74.5% coverage**
and **77.1% conditional accuracy** (23 correct, seven wrong, eight abstentions). It cannot replace
the predeclared primary method. All predictions and training support are in
[results.json](results.json), with a [readable prediction table](PREDICTIONS.md).

1. **The shared-position ambiguity survives richer layout hints.** Personal names `pe-se-ro-jo`
   (KN Ai 63) and `ma-re-wo` (PY An 657) share the exact heading/none/initial/not-repeated
   signature with occupations `to-ko-do-mo` (PY An 35) and `ka-ke-we` (PY Jn 658).
   Four name cases on three objects collide with four designation cases on four objects.
2. **Later position does not certify an occupation.** The name `o-wo-to` in PY An 261 shares
   entry/one/noninitial/not-repeated with `po-me` in PY Ae 134 and four other designation
   cases. Holding out An 261 leaves five unanimously designation-labelled supporting objects,
   so the method misclassifies this name. Training agreement is not calibrated certainty.
3. **Repetition does not resolve the remaining ambiguity.** The name `a-pi-jo-to` in PY An 261
   and designation/status phrase `te-ko-to-na-pe` in PY An 18 both have
   entry/one/initial/repeated. Across the three mixed signatures, unavoidable weighted error is
   **15.1%**, giving the **84.9% ceiling**. This ceiling is computed in sample with labels;
   it is an identifiability diagnostic, not a held-out recovery score.
4. **The negative control cannot rescue a failed positive control.** **0/177 evaluable**
   swapped-label runs pass the performance thresholds; 22/199 fail the fixed representation
   requirements and are excluded from that denominator. The negative gate passes, but the
   actual-label performance gate fails. No statistical significance or p-value is claimed.

| Evaluable negative-run metric | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Balanced recall | 0.0% | 0.0% | 34.4% |
| Coverage | 1.5% | 3.6% | 36.0% |
| Conditional accuracy | 0.0% | 0.0% | 100.0% |

The zero median recall reflects wrong calls or abstentions under conflicting swapped labels;
a negative run with a few correct calls can still have 100% conditional accuracy and fail coverage.
All 199 swap assignments and the 177 evaluable runs' metrics are retained. The spelling-renaming
software test confirms that changing syllables while preserving repetition cannot change these predictions.

## What this establishes and its limits

**Decision: retire this exact structural classifier for semantic transfer. No Linear A was scored.**
The concrete collisions rule out reaching the 90% recall gate on these fixed annotations using
only these features. They do not rule out distinctions using morphology, lexical identity,
longer relational frames, or a richer account of the tablet.

This is a deliberately curated development challenge, not an unseen or representative sample.
Labels come from a 1956 publication, with modern DĀMOS transcription and explicit exclusions;
they still need independent modern specialist review. Labels and selection were visible before
freeze. Supplied graphic entry boundaries, heading roles and quantity bins are assistance, not
recovered structure. Counts in multiword entries do not count each word separately. For example,
An 261's `a-pi-jo-to / ke-ro-si-ja / o-wo-to VIR 1` includes a leader, designation and individual;
calling all three “counted people” would be an annotation error.

No parentage, gender or language-family inference follows. A further experiment would need
**new evidence**, such as recurring relational frames and inflectional substitutions, plus a
known-language control distinguishing parentage from responsibility/ownership and occupation.
That would require expanded, independently reviewed name/relation labels; it is not a licence
to label Linear A slots from this result or rerun the failed method at a looser threshold.

## Records and reproduction

- [Protocol](PROTOCOL.md), [freeze](freeze.json), [source hashes and licences](sources.json),
  [annotations](cases.json), [inventory](INVENTORY.md), [results](results.json).
- Source labels: [Ventris & Chadwick, Documents in Mycenaean Greek](https://ignca.gov.in/Asi_data/17617.pdf),
  printed pp.162–174, 179–189 and 355 as specified per case; one-based PDF page = printed page + 39.
  Local copyright reference only; book scans are not redistributed.
- Transcriptions: [DĀMOS](https://damos.hf.uio.no/), 19 pinned item JSONs and exact item URLs
  in sources.json. Derived transcription data retain CC BY-NC-SA 4.0 attribution.
- Driver: `experiments/linear_b_person_role.py`; library: `linear_a/person_role_control.py`;
  tests: `tests/test_linear_a_person_role.py`. No previous frozen input was changed.

```sh
.venv/bin/python -m experiments.linear_b_person_role verify
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py'
```

The archived `results.json` is immutable. To reproduce the evaluation without replacing it,
validate and evaluate in memory, then compare against the archive:

```sh
.venv/bin/python - <<'PY'
import json
from experiments.linear_b_person_role import OUT, validate
from linear_a import audit, person_role_control
audit.verify(OUT)
rows, labels = validate()
assert person_role_control.run(rows, labels) == json.loads((OUT / 'results.json').read_text())
print('Exact result match; archive unchanged.')
PY
```
