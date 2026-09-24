# Accounting-program pilot: control not yet evaluable

The tablet-as-a-program pilot predicts a quantity from other rows using an anonymous word's
learned arithmetic rule. Its first source-coverage check stops the natural-data experiment:
the conservative parser yields **7 Knossos objects versus 33 Linear A objects**.

**Quantity vector:** separate commodity counts; `{grain: 2, oil: 3}` is not the scalar 5.
**Preflight:** a feasibility check before fitting, testing or computing a p-value.

```text
Build opaque integer-account records
Require >=10 eligible control objects and enough for 33-object matching
Find 7 control objects
Record not_evaluable; do not fit either natural corpus
```

## What was done

- Implemented three fixed arithmetic programs: sum preceding rows, sum following rows, and
  first preceding quantity minus the remaining preceding quantities. No arbitrary subsets,
  fitted constants or tolerance; different commodities remain separate coordinates.
- Added opaque word/commodity IDs, whole-object folds, minimum independent-object support,
  abstention on conflicting rules, numeric-vector shuffling with full refitting, and a
  preflight that catches insufficient control coverage before scoring.
- Archived parsed inventories, all exclusion reasons, evaluator vocabulary and the seven
  eligible source records for review. The raw sources and extracted data are hashed.
- Froze the implementation, tests, coverage and protocol at **`8793ad6`**, then ran the
  preflight. [results.json](results.json) records `not_evaluable`; control, negative-control
  and Linear A scores are null. No real-data model was fitted and no permutation was run.
- Passed **34 focused Linear A tests**, including ten new ledger tests. The new freeze,
  correspondence audit, repair audit and structural v2 freezes verify.

**Why:** arithmetic could ground a word function in a numerical prediction, but the control
must be possible on the available source representation. A failed coverage check is not a
failed test of arithmetic structure or a reason to retire the accounting idea.

## Coverage result

| Corpus | Source records | Parsed numeric lines | Eligible physical objects | Eligible prediction rows |
|---|---:|---:|---:|---:|
| Knossos Linear B control | 4,228 | 321 | **7** | 34 |
| Other Linear B, development only | 1,704 | 418 | 30 | 150 |
| Linear A retained source records | 212 | 286 | **33** | 134 |

Eligibility requires a word-bearing row in a contiguous block with at least two other numeric
rows on one side. It does **not** require that any arithmetic relation hold. The seven B
objects are candidates, not seven verified complete balance sheets. Rows are neither independent
tablets nor independent trials; the physical object is the sampling and fold unit.

The B parser rejects 8,193 Knossos lines as **damage or unsupported notation**, and 120 for
measurement units. That broad first category is not a count of physically illegible lines:
formatting, incomplete words, quotes and other unsupported syntax also cause rejection. Blank
and text-only lines bound numeric blocks, even where a specialist could establish continuity.
This is a limitation of the current automatic extraction, not a census of all usable accounts.

For A, 204 tablet records are excluded for an uncertain sign or a source conflict and 73 for
having no sign layer. Among the 212 retained records, 110 lines contain fractions and become
barriers. Unknown *sound values* are retained as sign IDs when the source sign itself is certain.
See [coverage.json](coverage.json) for every category and the fold distribution.

1. **The control cannot be matched without inventing independence.** Seven objects cannot
   supply the declared minimum of ten, or a 33-object sample without replacement. Duplicating
   objects or treating their rows as separate tablets would not repair this.
2. **The old arithmetic anchor was weaker than my earlier summary implied.** The old code
   explicitly ignored fractions. HT 13's displayed 130=130 compares integer parts only; its
   source contains fractions in both entries and the total. It does not verify the complete
   quantities. The new parser rejects fractional lines rather than silently treating them as
   integers. The original implementation and result are preserved.
3. **The software can recover a planted function.** On 30 synthetic objects, the model learns
   one opaque total word and predicts all 30 withheld-object totals exactly. A fixed shuffled
   case yields no predictions. Changing a withheld target number cannot change its prediction.
   These unit tests establish implementation behaviour, not power on ancient records.

## Source review candidates

[control-source-review.json](control-source-review.json) includes the complete pinned DĀMOS
item and parsed line numbers for each candidate. These links allow an edition/image review;
no new palaeographic readings have been made in this pilot.

| DĀMOS ID | Document | Primary entry |
|---|---|---|
| 1355 | KN As(2) 1519 | [DĀMOS](https://damos.hf.uio.no/1355) |
| 1358 | KN Uf(-) 1522 | [DĀMOS](https://damos.hf.uio.no/1358) |
| 37 | KN As(-) 40 | [DĀMOS](https://damos.hf.uio.no/37) |
| 476 | KN L-(-) 520 | [DĀMOS](https://damos.hf.uio.no/476) |
| 731 | KN B-(-) 798 | [DĀMOS](https://damos.hf.uio.no/731) |
| 833 | KN C-(-) 902 | [DĀMOS](https://damos.hf.uio.no/833) |
| 842 | KN C-(2) 911 | [DĀMOS](https://damos.hf.uio.no/842) |

## Interpretation and next experiment

**The accounting route remains open.** This first release establishes an extraction and
coverage problem, not a negative linguistic result. No natural-data accuracy or p-value is
reported. The learner was not asked to reinterpret Linear A after an unevaluable control.

The next bounded deliverable is a source-checked account benchmark, as specified in
[BENCHMARK_PLAN.md](BENCHMARK_PLAN.md). It needs complete quantities, explicit unit conversion,
row/face grouping and independently justified function labels. A word such as `to-so` alone
is not a gold label for a sum equation: it can introduce a specified amount. The conditional
reference-marker gate in this pilot therefore cannot substitute for source adjudication.

This release does not model fractions, mixed units, inter-tablet joins, nested subtotals or
allocation ratios. Those require new source-supported definitions, not changing this freeze.
The strict rejection of whole lines when a personal name is damaged also loses usable numeric
evidence; a future parser should separate uncertainty in names from uncertainty in quantities
when the source establishes their boundaries.

## Reproduction and attribution

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py' -v
.venv/bin/python -m experiments.linear_a_ledger verify
```

For the exact released run, use a clean worktree at `8793ad6`, attach the pinned `artifacts/`
sources and Python environment, then run `python -m experiments.linear_a_ledger run`.
Outputs refuse overwrite. `prepare` extracts sources without fitting an arithmetic model;
rerun it only in a scratch copy and compare its four extraction artifacts' hashes against
`freeze.json`. Coverage was examined during development before the freeze;
there were no post-freeze implementation changes or deviations.

Transcription-derived records retain CC BY-NC-SA 4.0 attribution to DĀMOS (Federico Aurora),
SigLA (Ester Salgarella and Simon Castellan), and the Navarre-AI/lineara.xyz collation sources.
See the original source manifests and [PROTOCOL.md](PROTOCOL.md).
Published fractional research provides a later starting point, not values used here:
[Corazza et al., 2021](https://cris.unibo.it/handle/11585/789546).
