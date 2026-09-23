# Linear A repair audit

Declared 2026-09-24 before corrected runs. CPU only; no downloads. This is a retrospective
repair on previously inspected corpora, not independent confirmation or a language claim.
Historical code, protocols and results remain unchanged. New code lives in `probes_v2.py`
and `linear_a_audit.py`. Freeze hashes code and every consumed source, including all DĀMOS
items and the TLHdig parsed cache. Existing source/licence records: rounds one, two and five.
Derived SigLA/DĀMOS results remain CC BY-NC-SA 4.0.

## What will be done

- Deduplicate profile inputs. Sample every target/reference without replacement under the same
  exact-length quotas, capped by the capacity available in every pool. At most 300 types per
  draw, 20 draws. Report actual size and every quota; this conditions on common length support
  rather than claiming to preserve each corpus's original length distribution. Scale features
  using reference samples only. Samples overlap; 20 wins are stability, not 20 independent tests.
- Recreate round five's formerly interactive diagnostics using fixed seeds: Linear B/Linear A,
  globally shuffled syllables, bigram pseudo-words, Hittite reference capped at 700 types, and
  o/u merged in all corpora. New seeds are explicit; original interactive numbers need not agree.
- Correct probe 6's unreachable threshold using 9,999 confirmation and pre-named null draws.
  Preserve its original 100-run discovery stage, split, random streams, statistic and bigram
  null (including repeated pseudo-words), so this audits resolution rather than changing the
  hypothesis. Use an equivalent indexed pair counter, tested against the original implementation.
- Also correct probe 4's 50-run control to 999 null draws per 20 samples. Each draw gets its own
  deterministic stream. The two-sign trade stems and sample size remain fixed; this is a new
  retrospective control, not an exact extension of the old draws.
- Record raw null counts, exceedances, corrected p, and Wilson intervals for Monte Carlo
  exceedance probability. Reject any configured budget whose p floor cannot reach the threshold.

## Why

Duplicates were counted as morphological variation, and two Monte Carlo tests could not
attain their own pass marks. Repairs measure how much these defects affected the conclusions.

## Fixed decisions

1. Profile positive controls: Linear B → Greek at least 18/20; each candidate's held-out
   hash half → itself at least 18/20. Negative control: globally shuffled Linear B → Greek
   at most 1/20. Run corrected Linear A only if both positive and negative controls pass and
   at least one non-Greek candidate passes its own control. A failed gate retires this profile
   as a test of word structure; historical diagnostics remain explicitly retrospective.
2. Correspondence family threshold stays 0.05/7; no threshold relaxation. The pooled set from
   the original discovery half is primary. Individual discovered rules use alpha divided by
   the number of discovered rules. The pre-named re/ru → ro result is exploratory even if small.
   At 9,999 draws the minimum p is 0.0001. A confidence interval crossing alpha is unresolved
   Monte Carlo precision, not a reason to extend runs after seeing the result.
3. Trade control requires at least 18/20 samples below 0.05/7. The minimum p at 999 draws is
   0.001. No new Linear A trade test is included in this repair audit.
4. Passing corrected controls does not identify Linear A's language. No retuning after results.

## Reproduction

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py' -v
.venv/bin/python -m experiments.linear_a_audit freeze
# Commit the code, protocol and freeze before the following runs.
.venv/bin/python -m experiments.linear_a_audit verify
.venv/bin/python -m experiments.linear_a_audit profiles
.venv/bin/python -m experiments.linear_a_audit rules
.venv/bin/python -m experiments.linear_a_audit trade
```

Each run refuses to overwrite its output. Reproduce in a new worktree at the protocol commit.
