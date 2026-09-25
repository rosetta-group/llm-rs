# Three rejection follow-ups: development results

## What was done

- Compared the original rule with a transfer-centered candidate on released cases.
- Fitted independent keys under five frozen language priors for each frequency-preserving copying control, then transferred sealed keys.
- Implemented and benchmarked chunked and incremental refinement in a new module; earlier frozen code remains unchanged.

## Why

The original fit ceiling rejected genuine English even though its key transferred. Mutation controls also introduced unfamiliar pieces, while exhaustive full-text swap scoring made repeated controls expensive.

**Transfer:** decoding a separate passage with the learned key fixed.
**Development:** released examples used to design a rule; they cannot independently confirm that rule.

## 1. Decision calibration

The candidate accepts **3/3** released genuine ciphers, versus **2/3** for the original rule. It removes only the fit-excess ceiling: agreement of language winners, both 0.25 margins, 0.50 transfer ceiling, 95% coverage and cap checks remain.

| Input | Cases | Original accepted | Candidate accepted | Inconclusive |
|---|---:|---:|---:|---:|
| positive | 3 | 2 | 3 | 0 |
| absent | 3 | 0 | 0 | 0 |
| shuffle | 3 | 0 | 0 | 0 |
| copy | 3 | 0 | 0 | 1 |
| frequency_copy | 3 | 0 | 0 | 0 |

Acceptance is desirable for positives and an error for the negative classes. The original Latin mutation case remains inconclusive under both rules. All examples here share only three released source/key blocks. The apparent improvement is development evidence, not a fresh sensitivity or false-positive estimate.

The [sensitivity table](calibration-results.json) records 21 transfer ceilings from 0.00 to 1.00. No threshold was selected on new sealed text; the 0.50 candidate was declared before the stronger-control fits.

## 2. Frequency-preserving copying control

Each copied passage consumes exactly its source token bag. Local copying changes order while preserving token count, frequencies, types and glyph inventory. Each new control gets a full search under all five priors; the positive cipher’s fitted key is not reused.

| Source block | Fit tokens / types | Recent-repeat rate, original → copied | Transfer coverage | Transfer winner / excess | Candidate |
|---|---:|---:|---:|---|---|
| english | 3427 / 1552 | 12.4% → 52.8% | 98.64% | old_french / 3.000 | rejected |
| italian | 3442 / 1461 | 15.8% → 55.9% | 98.47% | old_french / 2.999 | rejected |
| latin | 3426 / 1372 | 16.2% → 57.6% | 98.53% | old_french / 3.137 | rejected |

Recent-repeat rate is the share of tokens already seen within the preceding 50 positions. This generator is a controlled stress test, not an implementation of Timm’s generator and not a model of all meaningless text.

## 3. Refiner cost and equivalence

| Fixture | Legacy swaps (s) | Chunked (s) | Incremental (s) | Speedup, legacy / incremental | Peak RSS, legacy / incremental (MiB) |
|---|---:|---:|---:|---:|---:|
| synthetic-64 | 0.0133 | 0.0137 | 0.0052 | 2.6× | 314.0 / 313.8 |
| synthetic-192 | 0.2527 | 0.2593 | 0.0370 | 6.8× | 328.5 / 315.3 |
| english-positive | 0.8550 | 0.8941 | 0.0498 | 17.2× | 364.5 / 317.2 |
| english-copy | 4.8167 | 4.9421 | 0.1547 | 31.1× | 1694.0 / 331.0 |

Times are medians of two warmed exhaustive pair-swap batches, with two threads. Every backend and repetition chose the same earliest winning pair and exact full score. The largest sampled incremental-score difference was 1.16e-10 bits; close minima are rescored with the original objective. Small complete-refinement tests also match the legacy mapping, recovered text, score, evaluation count and kicks; two-letter keys use the chunked full scorer.

Chunking alone reduced memory but did not improve runtime on these fixtures. Incremental scoring changes only the n-grams touched by a proposal and its homophone cost. The 31× copying-fixture result is a search-step speedup, not an end-to-end decoder speedup. The initial EM and repair stages are unchanged.

With U units, N decoded one-letter positions and fixed n-gram order, the old pair sweep scans O(U² N) positions and builds O(U³) candidate-key storage. Incremental proposal work visits O(U N) affected-window occurrences across the sweep; near-tie full rescoring can increase runtime and has the old worst-case scoring order. Candidate batches bound key storage to O(batch_size × U), with linear input/position-index storage.

## Resource and provenance record

- Stronger controls: 3/3 pairs, 15 fits, 1.028 aggregate fit-worker hours; status: `completed`.
- Refiner time within those fits: 0.085 aggregate hours; 0 fitted candidates reached a limit.
- The first follow-up freeze was preserved but not run. Preflight increased the sweep guard from 50 to 200 to accommodate the inherited 30 kicks; time and evaluation budgets stayed fixed. See [the recorded correction](CONTROL_RESOURCE_REPAIR.md).
- An unsupported synthetic alphabet symbol was caught before any benchmark timing; the fixture now reads the pinned prior alphabet. See [setup correction](benchmark-setup-correction.json).
- CPU only; five workers with two threads each. No manuscript text or unused sealed answers were opened.

## Decision and limits

Use the incremental backend for further development with the declared numerical checks and work limits. Keep the transfer-centered rule as a candidate for fresh confirmation; its development results do not replace the original screen. Keep frequency-preserving copying in the negative-control set alongside shuffle and mutation controls. The existing 1% character-error / 10% word-error manuscript gate remains unchanged.

A [costed confirmation plan](CONFIRMATION_PLAN.md) is a proposal only. No fresh confirmation was run as part of these three development follow-ups.

## Records and reproduction

[Protocol](PROTOCOL.md) · [executed freeze](freeze-v2.json) · [benchmark](benchmark.json) · [controls](control-results.json) · [calibration](calibration-results.json) · [verification](verification.json) · [released development records](evaluated-records.tar.gz)

```sh
.venv/bin/python -m experiments.rejection_followups_v2 verify
.venv/bin/python -m experiments.audit_rejection_followups
NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_rejection_followups -v
```

Working sources/priors must be restored at the paths in the freeze. The development archive preserves its input plan, public texts, fit records, key seals, transfer records, graded decisions and benchmark fixtures. Creation stages refuse overwrites.
