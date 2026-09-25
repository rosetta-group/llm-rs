# Source-annotated accounting development benchmark

## What is frozen

- Eight whole Linear B objects, nine annotated accounts, selected from the named
  arithmetic examples in Ventris & Chadwick p.118 plus the contrasting transaction p.217.
- Units, row boundaries, commodity inheritance, doubtful quantities and separate
  functional labels. All are manually reviewed development material, already inspected.
- Relative units from p.55; no absolute litre/kg estimates and no Linear A fraction values.

## Why

The integer pilot could not cover the target corpus. A source-backed development fixture
must first distinguish parser loss, damaged evidence and genuine arithmetic discrepancies.

## Mechanics

```text
Read the fixed account annotation
Sum the fixed item rows using exact rational units
Do not inspect the target quantity while predicting
Compare prediction with separately read target
Abstain strictly if quantity uncertain, account incomplete or dimensions incompatible
Also report visible transcribed readings as conditional arithmetic
```

`labels.json` records published functional interpretations; the arithmetic code does not
read it. This is an oracle-boundary, oracle-commodity test, not a learned function detector.
Success on it is no control gate. No fitted subsets, inferred missing numbers or repairs to
the written totals. An uncertain aggregate row is excluded as a whole; it is not resolved
using the target. Damage in a name alone need not exclude its certain numeric allocation.

The published account on KN As1516 supplies two cases but one independent object. PY Jn658
retains the published discrepancy. PY Fr1184 is a hard negative for `to-so implies summation`:
18 units of oil and 38 jars have different dimensions. Erased entries in Jn658 are excluded.
MY Fo101 deliberately bridges blank lines .11-.14 to its published total .15.

Both strict and conditional results will be reported, with exact residual target-minus-sum.
Conditional arithmetic never upgrades restored text to evidence. Eight objects cannot satisfy
the planned >=33-object matched control, including >=10 function-positive objects. Linear A
will not be scored from this fixture. Previously inspected objects are development exclusions
for any future held-out control. Freeze and commit before running the arithmetic report.
