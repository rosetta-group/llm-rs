# Linear A repair audit: reachable tests, stronger exploratory correspondence, no language claim

Completed 2026-09-24 on `codex/linear-a-audit`. [Protocol](PROTOCOL.md), code and
[freeze](freeze.json) committed at `0fd5f74` before corrected runs. This is a retrospective
repair on previously inspected data. Historical results and code remain unchanged.

## What was done

- Added type-level profiles and unique, exactly length-matched samples in `linear_a/probes_v2.py`.
  Reference feature scales are fitted without the targets. Recorded all sample quotas, distances,
  target profiles and reference centres in [profiles.json](profiles.json).
- Recreated the old profile method and its formerly interactive diagnostics with declared seeds.
- Extended probe 6's confirmation and pre-named simulations to 9,999 draws, preserving the
  original discovery stage, split, random streams, pair counts and threshold. Indexed matching
  is checked against the original implementation. Raw counts are in [rules.json](rules.json).
- Found and repaired a second unreachable test: probe 4's trade-word control had only 50 null
  draws. Ran 999 per sample with independent deterministic streams; [trade.json](trade.json).
- Added resolution guards, Monte Carlo exceedance counts and Wilson intervals. No thresholds
  were relaxed and no Linear A language result was promoted.

## Why

The old profile mistook duplicate occurrences for changing word forms; two significance tests
could not reach their own threshold. The audit measures the consequences before drawing a
conclusion about the information in the corpus.

## Results

| Check | Original | Corrected |
|---|---:|---:|
| Two identical `ka-ta` words: prefix/suffix alternation | 100% / 100% | 0% / 0% |
| Hittite held-out half → Hittite | 20/20 | 19/20 |
| Luwian held-out half → Luwian | 7/20 | 19/20 |
| Palaic held-out half → Palaic | 0/20 at 189 types | 20/20 at 164 types |
| Hurrian / Hattic / Akkadian self-control | 20 / 18 / 20 | 20 / 20 / 20 |
| Linear B → Greek | 20/20 | 20/20 |
| Globally shuffled Linear B → Greek | not a gate | 20/20; required at most 1 |
| Profile gate for a word-structure test | shuffled control absent | **failed**; corrected Linear A skipped |
| Discovery-selected re → ro, held-out half | 2 pairs; p = 0.1584 | 2 pairs vs 0.7265; **p = 0.1653** |
| Pre-named re/ru → ro, already inspected full set | 12 pairs; p = 0.0099 (floor) | 12 pairs vs 2.9244; **p = 0.0001** (floor) |
| Trade-word known-answer samples passing p < 0.05/7 | 0/20; impossible with 50 draws | **3/20**, required 18 |

1. **Duplicate repair changes known-answer performance.** Luwian and Palaic now recognise
   their held-out halves. Palaic's common exact-length support allows only 164 distinct types.
   The repair also changes length sampling and reference-only scaling, so these improvements
   cannot be attributed solely to deduplication. The original failures were not proof of an
   information limit for these languages.
2. **Profile recognition is not specific to word structure.** Greek wins every real and every
   shuffled Linear B sample. The gate fails on specificity, not positive-control accuracy.
   This retires this profile as a test of word structure; phonotactic similarity is a different
   question. No corrected target-language ranking is reported after that gate failure.
3. **More simulation resolves the p-value floor, not the data's history.** For the held-out
   discovered rule, 1,652/9,999 null counts are at least 2, giving p = 0.1653. For the full-set
   pre-named pattern, 0/9,999 reach 12, giving p = 0.0001, not zero. The respective Wilson 95%
   intervals for the null exceedance probabilities are [0.1581, 0.1726] and [0, 0.0003840].
   The latter supports an exploratory correspondence under this particular bigram null.
   The combined re/ru pattern was already seen before the original protocol and is not the
   same hypothesis as the discovery-selected re-only held-out test. It is not independently
   confirmed and does not identify the language or meanings of the words.
4. **The second impossible test mattered, but does not rescue trade matching.** With 50 null
   draws, the old trade control's floor was 1/51 = 0.0196, above 0.05/7 = 0.00714. The new
   floor is 0.001, and 3/20 draws now pass. This is still far below 18/20. New sample streams
   were declared in advance; this is not a paired extension of each historical sample.

### Scripted historical diagnostics

| Historical algorithm, declared seeds | Nearest-language counts |
|---|---|
| Linear B | Greek 20 |
| Linear A | Hittite 20 |
| Globally shuffled Linear A syllables | Hittite 20 |
| Linear A bigram pseudo-words | Hittite 14, Hurrian 4, Hattic 2 |
| Hittite reference capped at 700 | Greek 6, Hattic 11, Hurrian 3 |
| o merged into u everywhere | Hittite 11, Greek 9 |
| Same o/u merge, Linear B | Greek 20 |

Only the real-data calls reuse the old driver seeds. The earlier interactive negative-control
seeds were not recorded, so these are reproducible re-creations rather than exact reproductions
of every old number. The diagnostic conclusions persist: shuffled words retain the Hittite
match, and reference size and vowel conventions affect it.

## What this does and does not establish

The implementation and simulation defects are real, and fixing them changes several results.
The repaired methods still do not identify a language. The pre-named spelling correspondence
deserves its exploratory label rather than being grouped with an undifferentiated “no lead.”
An independent confirmation would need evidence not used to select this pattern, such as newly
held-out inscriptions or independent linguistic/epigraphic checks; more simulations of the same
corpus are not that evidence. This audit does not change the null's phonotactic assumptions,
the lexicons, the spelling approximations or the noisy context labels.

Samples overlap; 20 wins are a stability check, not 20 independent replications. Historical
rounds one to three still pass their `verify` commands. The separate [structural test](../linear-a-structure-v2/REPORT.md)
also failed its known-answer gate; this does not establish that every structural approach must fail.

## Records and reproduction

Use a worktree at `0fd5f74` with the pinned sources attached under `artifacts/`:

```sh
.venv/bin/python -m experiments.linear_a_audit verify
.venv/bin/python -m experiments.linear_a_audit profiles
.venv/bin/python -m experiments.linear_a_audit rules
.venv/bin/python -m experiments.linear_a_audit trade
```

Each output refuses overwrite. The freeze includes the consumed TLHdig cache and every DĀMOS
item. Source origins/licences remain in the original rounds' manifests. Derived SigLA/DĀMOS
results: CC BY-NC-SA 4.0. Current verification: 16 focused tests pass, including duplicate
invariance, strict-threshold boundary, exact unique sampling, matching equivalence, damage
handling, holdout isolation, synthetic order recovery and the structural driver's reporting.
