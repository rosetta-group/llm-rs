# Theban *65/FAR: the commodity-or-kinship audit

This audit examines whether a disputed Theban sign writes “son” or names an allocated commodity.
The son interpretation has a published linguistic case, but the inspected quantities cannot independently decide it.

**FAR:** the editorial commodity reading of the glyph, not a neutral record of its function.
**Conditional value:** the value of supplied quantity components if their readings are accepted.
**Closed account:** complete allocations with a secure total and known scope.

## What was done

- Preserved 13 DĀMOS objects: 11 Theban cases and two commodity controls, including the
  joined Fq254[+]255 as one object. [Source cards](cases.json).
- Read the relevant Palaima, Duhoux and James discussions; visually checked the published
  James quantity table and Palaima pp36–38. Original tablet surfaces were not obtained.
- Froze code, annotations and local source hashes at **aeb16d9**, then generated six conversion
  comparisons and two account-feasibility checks. [Protocol](PROTOCOL.md), [results](results.json).
- Passed **74 focused tests**; verified **11 follow-up freezes**, exact result reproduction
  and refusal to overwrite the archived output.

## Why

A quantity after a proposed kin term can still measure food allocated to that person.
The useful question is whether changing the glyph's function changes which commodity account balances.

```text
Retain both son and commodity readings
Check original sign placement and the proposed linguistic parallel
Calculate each reading's commodity assignments using secure quantities
If allocations, total or scope are uncertain, leave balance comparison unevaluable
```

## Results

Dry conversion: BASE = 240 Z, T = 24 Z, V = 4 Z. These are relative units, not litres.
The following checks compare the two columns of James's Table 1, not reconstructed tablet totals.

| Tablet | Printed components | Printed Z value | Conditional component Z | Difference: printed minus calculated |
|---|---|---:|---:|---:|
| Fq214 | T6 V5[ | 164 | 164 | 0 |
| Fq252 | ]T7 V2 Z2 | 186 | 178 | +8 |
| Fq254[+]255 | [T]3 V3 Z2 | 169 | 86 | +83 |
| Fq269 | 1 V2 [ ] Z3 | 251 | 251 | 0 |
| Fq276 | T8 V2[ | 200 | 200 | 0 |
| Fq277 | ]2 T6 V4 Z1 | 525 | 641 | −116 |

Three conversions disagree with their printed components. The discrepancy is present in the
page image, not just OCR. Neither column is silently corrected. Fq252 combines a superscribed
V2 with the total; Fq277's broken leading 2 is treated as BASE2 only conditionally. James also
warns that lost entries limit quantitative inference and explicitly adopts the *65 reading
in her recipient table; that table is therefore not an independent semantic control.
[James, pp406–407](https://gredos.usal.es/bitstream/handle/10366/133917/The_Thebes_Tablets_and_the_Fq_Series_A_C.pdf?sequence=1).

| Account | Disputed allocations | Conditional visible shift between HORD and FAR | Exact balance eligible? |
|---|---|---:|---|
| Fq214 | .7 V2; .13 doubtful V, numeral lost | 8 Z plus unresolved content | No |
| Fq254[+]255 | .6 V1 doubtful; .7 V1 before break; .13 V2 | 16 Z on supplied components | No |

These shifts are neither exact totals nor unconditional lower bounds. On the son reading,
these amounts belong with the inherited commodity; on the FAR reading, they move to a
separate product. Both bodies are incomplete, the totals are qualified, and the scope of
recapitulation is not independently settled. A difference of 16 Z does not select a reading
when the missing amount is unknown. [Fq214](https://damos.hf.uio.no/5235),
[Fq254[+]255](https://damos.hf.uio.no/5252), [annotations](quantities.json).

1. **The linguistic parallel is substantive but qualified.** Duhoux reads Gp227's
   `ra-]ke-da-mo-ni-jo-u-jo` as an unnamed son of [La]kedaimonios and compares it with
   Fq229's `ra-ke-da-mi-ni-jo` plus the disputed glyph. Different name spellings and scribal
   assignments prevent treating this as a verified same-person substitution. Duhoux agrees
   with Palaima on the son interpretation but disputes its precise grammatical explanation.
   [Duhoux2007, pp97–100](https://www.researchgate.net/publication/327727893_Le_nom_du_fils_en_lineaire_B_dans_F_LANG_-_C_REINHOLDT_-_J_WEILHARTNER_ed_STEPHANOS_ARISTEIOS_Archaologische_Forschungen_zwischen_Nil_und_Istros_Festschrift_fur_Stefan_Hiller_zum_65_Geburtstag_Vienne_),
   [Gp227](https://damos.hf.uio.no/5410), [Fq229](https://damos.hf.uio.no/5239).
2. **Sign placement needs the surface.** Palaima reports that Fq236 .5's glyph touches the
   preceding `-no`, despite misleading separation in the edition's transcription. His p36
   reference says Gp236; p38 explicitly identifies Fq236 and TOP pp94–95. Our DĀMOS text
   has a space and doubtful FAR, so it cannot independently confirm his observation.
   Gp124's `FAR , VIN` is another useful placement check, not proof from transcription alone.
   [Palaima2003, pp36–38](https://sites.utexas.edu/scripts/wp-content/uploads/sites/3428/2020/06/2003-TGP-ReviewingTheNewLinearBTabletsFromThebesKADMOS-1.pdf),
   [Fq236](https://damos.hf.uio.no/5240), [Gp124](https://damos.hf.uio.no/5375).
3. **Commodity use remains a real rival.** KN Fs2 lists FAR separately from VIN and HORD;
   PY Un718 .10 has `me-re-u-ro , FAR T6`. Replacing every FAR with “son” would destroy
   these controls. Nor does recognising a phonographic use alone fix its meaning.
   [KN Fs2](https://damos.hf.uio.no/2), [PY Un718](https://damos.hf.uio.no/5012).
4. **No new family edge is established.** Gp215 .1 is a possible named-child/parent
   construction under the son interpretation, but its first name is damaged and the sign's
   function remains disputed here. Gp227 leaves the child unnamed. The output therefore
   contains zero semantic predictions and zero directed edges. [Gp215](https://damos.hf.uio.no/5409).

## Decision and limits

Keep the Theban son proposal as a source-supported hypothesis. Do not add these occurrences
as undisputed training labels or transfer their meaning to Linear A. There is no new daughter
formula in this deliberately bounded *65 review. The previous daughter cases and all failed
controls remain unchanged.

The original TOP photographs/facsimiles and the full AGS2003 response were not obtained.
The opposing reading is represented through Palaima/Duhoux, not a claim to have directly read
its full defence. This limits adjudication. James's two downloaded copies are one paper;
Palaima, James and Duhoux are not three independent validations of the same hypothesis.
This was exploratory source work with discrepancies noticed before freezing, not blind discovery.
[Source/access manifest](sources.json).

The next useful step is narrowly specified: obtain TOP pp94–95 for Fq236 and the corresponding
Gp124/Gp227/Fq254 images, then annotate glyph boundaries without the FAR/*65 interpretation
shown. Read AGS2003 pp24–25 directly alongside the published reply. If that resolves sign
function, test the son interpretation against other phonographic readings separately. More
classification on the present incomplete quantities would not answer either question.

## Records and reproduction

New arithmetic: `linear_a/theban_65.py`; driver: `experiments/linear_b_theban_65.py`.
Source PDFs stay in ignored local storage; their hashes, URLs, licences and reviewed pages are
in the manifest. Duhoux was read through author-uploaded web text, with no local PDF/hash claimed.
DĀMOS-derived transcriptions retain CC BY-NC-SA 4.0 attribution.

```sh
.venv/bin/python -m experiments.linear_b_theban_65 verify
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py'
```

In a fresh checkout at `aeb16d9` with the pinned sources attached, run
`.venv/bin/python -m experiments.linear_b_theban_65 run`. Existing output is protected.
An in-memory comparison works without modifying the archive:

```python
import json
from experiments import linear_b_theban_65 as driver
from linear_a import theban_65
quantities, cases, _ = driver.validate()
assert theban_65.run(quantities, cases) == json.loads((driver.OUT / 'results.json').read_text())
```
