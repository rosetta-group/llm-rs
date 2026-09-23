# Scope: Proto-Elamite

Status: scoping only, 2026-09-24, branch `proto-elamite`. Nothing is downloaded yet. Sources and
prior work are in [BACKGROUND.md](BACKGROUND.md) (AI-compiled; items tagged [U] unverified).

## Why Proto-Elamite

It is the largest undeciphered corpus of the candidates in
[UNDECIPHERED.md](../../docs/UNDECIPHERED.md): about 1,600 administrative tablets from Iran,
c. 3100–2900 BCE. Its accounts carry numbers, so arithmetic can test hypotheses without knowing
the language. That is the kind of evidence that worked for Linear A (`ku-ro` "total"). It also
has a known-answer control in the same database and notation: **proto-cuneiform**, whose number
systems and many signs are understood.

The aim is sign *function* (numbers, commodities, persons, totals), not translation.

## Data

| Need | Source | Size | Licence |
|---|---|---:|---|
| Proto-Elamite texts | CDLI API, period Proto-Elamite (`cdli.earth/search?period=Proto-Elamite`, JSON with ATF) | 1,755 records; 1,399 clean transliterated tablets; 33,778 tokens, 11,364 of them numerals | CDLI terms: free copying and re-use under academic practice, credit to CDLI (no Creative Commons licence named) |
| Control texts | same API, periods Uruk III and Uruk IV, administrative genre only | about 7,000 records | same |
| Comparison snapshot | `sfu-natlang/pe-headers` (the corpus behind Born et al.) | — | none stated; used only to compare counts |

## What is already done, and must not be repeated

| Work | Result |
|---|---|
| Born et al. 2023/2025, numerals | Sum checks disambiguated 24 texts (IDs unpublished); textbook ratios assumed; no null model |
| Born et al. 2022, headers | 92–95% agreement with expert header labels |
| Born et al. 2019, 2021 | sign clusters, topic models, partial compositionality of complex signs |
| Kelley et al. 2022 | Linear Elamite sound values do not carry over |
| nlarch/proto-elamite (2026, not reviewed) | full-sum matches 63 real vs about 6 under a permutation null the author calls too easy |

Every published analysis assumes the textbook conversion ratios, and none has a proto-cuneiform
positive control.

## Proposed first round: blind recovery of number systems and totals

```text
download Proto-Elamite and proto-cuneiform (administrative) via the CDLI API; pin hashes
parse entries in both orders (PE: signs then number; proto-cuneiform: number then signs)
unknowns: the ratio between each pair of adjacent numeral units, per number system
score: tablets whose reverse line equals the sum of obverse entries under the candidate ratios
search: all ratio sets from a fixed candidate range (2 to 60), per system
control: proto-cuneiform must recover its known ratios (grain N14 = 6 N01, counting N14 = 10 N01)
         from arithmetic alone, and beat a null where numbers are shuffled across tablets
only if the control passes: run on Proto-Elamite; report recovered ratios and balancing tablets
```

Why this is new: it treats the ratios as unknown instead of assuming them, and it is validated
where the answer is known. A second round could classify frequent signs by position (object,
person, header), validated on proto-cuneiform glosses with a sign-shuffle null.

## Risks

1. **Damage.** Many tablets are broken; a missing line breaks a sum. The search must tolerate
   unbalanced tablets and report how many balance.
2. **Ambiguous numerals.** The same numeral signs belong to several systems; Born et al. found
   only 1,899 of 8,011 unambiguous. The model must assign systems as latent variables.
3. **Easy nulls.** A permutation null that ignores magnitude is too easy; the null must keep each
   tablet's number sizes and shuffle only which lines they sit on.
4. **Licence.** CDLI's terms are academic re-use with credit, not an open licence; derived files
   stay in the repository with attribution and are not redistributed in bulk.
