# Source-checked accounts: development benchmark

## What was done

- Built nine annotated accounts from eight physical Linear B objects, including one
  hard negative. Froze code, quantities, boundaries, functional labels and sources at
  `f048a56` before producing [results.json](results.json).
- Checked the named arithmetic examples in Ventris & Chadwick, *Documents in Mycenaean
  Greek* (1956; scanned 1959 reprint), against pinned DĀMOS transcriptions.
- Implemented exact relative units and separate strict/conditional arithmetic. Seven
  software tests cover carries, missing values, target hiding and incompatible dimensions.

## Why

The previous integer parser rejected useful accounts because of measures or damaged names.
A benchmark must also retain broken accounts and published errors, rather than selecting
only equations that happen to balance.

## Results

**Strict** means the annotated numeric evidence is complete and certain. **Transcribed**
means the displayed readings are used despite damage/restoration; these are conditional
calculations, not confirmations of the missing text. A residual is written total minus sum.

| Account | Strict result | Conditional sum / written total | Source basis |
|---|---|---|---|
| PY Jn845 | balanced | M12 / M12 | p.118: eight allocations M1 N2; M=4N |
| KN As1517 | balanced | 17 / 17 people | pp.118,172: first roster, damaged names but certain counts |
| KN As1516 first section | not evaluable | 30 / 31 people | p.171: boundary and excluded first MAN; gaps remain |
| KN As1516 second section | not evaluable | 23 / 23 people | p.171; restored numeral in .14 |
| PY Jn658 | not evaluable | L2 M20 / L3 M20 | pp.118,355: published error; doubtful numerals retained |
| MY Fo101 | not evaluable | BASE2 S1 V1 / same | pp.118,218; restored/damaged cells |
| KN Fp1 | not evaluable | BASE3 S1 V2 / BASE3 S2 V2 | p.118; right-edge breaks in pinned .6/.7 |
| KN F51 | not evaluable | T7 V5 Z3 / same | pp.55,118; doubtful Z3 and break |
| PY Fr1184 | incompatible | 38 jars / 18 oil units | p.217, older Gn1184 designation: specified repayment |

1. **Representation works on two certain cases.** Jn845 carries weight units exactly:
   8 × (M1 + N2) = M12. As1517 keeps seventeen individual entries despite damaged names.
   This is a check with supplied boundaries and commodity labels, not learned word meaning.
2. **Conditional agreement is weaker.** Five accounts balance on the transcribed readings,
   but only two meet strict requirements. Three conditional mismatches remain. No missing
   amount was inferred to force a match. Jn658's excess is one L, matching the published
   discussion; its modern doubtful readings still prohibit a strict certain-text claim.
3. **A marker alone is insufficient.** Fr1184 contains `to-so` but does not itemize the
   eighteen oil units. Adding its thirty-eight jars to that volume would be a dimensional
   error. As1517 also has an edition/DĀMOS `to-sa`/`to-so` discrepancy; quantities do not
   decide which lexical reading is right.
4. **Boundaries matter.** Fo101's published total follows four blank lines. As1516 has
   separately headed sections, with the initial MAN excluded by the edition's discussion.
   These annotations are exposed development hints and cannot be credited to a learner.

## What this establishes

A working, source-auditable development fixture for counts, weights, dry and liquid measures.
The [labels](labels.json) cite printed pages and explain each boundary, uncertainty and
commodity inheritance; [source-review.json](source-review.json) retains the pinned raw text.
The PDF page offset changes: printed 55 is PDF92, 118 is PDF157, 171 is PDF210.

It does **not** establish power for the 33-object target. There are eight objects overall,
seven with source-labelled totals, and only two strictly evaluable sums. The existing gate
is unchanged, no Linear A arithmetic is scored, and no word is translated. Historical
interpretations are stated as such, not claimed to be a current critical edition. The next
accounting expansion needs more independent source-verified accounts, including balances;
this fixture currently tests addition and a dimension negative, not balance recovery.

## Records and reproduction

[Protocol](PROTOCOL.md), [inputs](inputs.json), [labels](labels.json), [sources](sources.json),
[freeze](freeze.json), [results](results.json).
The [book scan](https://ignca.gov.in/Asi_data/17617.pdf) is stored locally and hash-pinned,
not redistributed. DĀMOS-derived records carry the corpus's CC BY-NC-SA 4.0 attribution.
Only relative units are used; no litres, kilograms or assumed Linear A fraction values.

```sh
.venv/bin/python -m experiments.linear_a_account_benchmark verify
.venv/bin/python -m unittest tests.test_linear_a_account_benchmark
# In a fresh checkout without results.json (exclusive output creation):
.venv/bin/python -m experiments.linear_a_account_benchmark run
```
