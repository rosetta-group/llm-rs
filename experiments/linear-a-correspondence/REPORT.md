# Linear A correspondence audit: archive the specific adaptation lead

The conditional ending test audits the proposed Linear A `-re`/`-ru` → Linear B `-ro` pattern.
Six of the 12 historical pairs survive source filtering, but the specific final-vowel claim
fails its declared robustness gate: p = **0.1605** after conditioning on final onset.

**Stem:** all signs before the last, e.g. `qa-qa` in `qa-qa-ru`.
**Strict pair:** an exact shared stem supported by intact, unmarked source readings in both scripts.
**Conditional null:** redistribute observed endings among real stems within fixed groups.

```text
Audit all 12 historical pairs → retain 6 under fixed source rules
Shuffle endings within word length → p = .0011, passes
Also preserve final onset → p = .1605, fails
Require both → archive the statistical adaptation lead
```

1. **Reading quality changes the evidence.** The old code merged `q` with `k` and numbered
   signs with plain signs, and removed editorial uncertainty. `a-ka-ru` matched `]a-qa-ro`;
   `ta2-ta-re` matched `ta-ta-ro`. HT 117a's glyph layer says `te-ja-re`, but its word/sign
   layers say `te-*56-re`. [All 12 decisions and source links](PAIRS.md) are preserved.
2. **General overlap survives; the vowel-specific inference does not.** Keeping real stems
   and length-specific ending counts still makes six matches unusual (null mean 1.5342).
   Preserving the final onset raises the mean to 4.2191. There are already 15 `re`/`ru` endings
   among the strict corpus's 32 three-sign words ending in an r-onset sign; the latter null
   tests their placement beyond that generic r pattern. It deliberately conditions away any
   broader r correspondence rather than refuting it.
3. **The earlier strong score depends on source treatment.** The historical normalised
   inventory passes both new comparisons, including p = .0049 for the onset-conditioned null.
   The strict inventory fails. Source exclusions reduce coverage and power as well as removing
   questionable matches; this result does not establish that any individual borrowing is false.

## What was done

- Added `linear_a/correspondence.py` and a reproducible `prepare|freeze|verify|run` driver.
- Archived exact source lines, uncertainty, sign IDs, source conflicts, accepted attestations,
  complete inventories and the 12-pair audit table. No old frozen code or result was edited.
- Committed protocol/code/inputs/source hashes at **`75128e8` before real-data permutations**.
- Ran 9,999 permutations for each of two nulls on the strict and historical inventories.
  Every score, fixed seed and Monte Carlo interval is in [results.json](results.json).
- Passed all **24** focused Linear A tests, including eight new source/permutation tests.
  Original rounds one–three, the repair audit and structural v2 freezes still verify.

**Why it was done:** the repaired p = .0001 from the earlier syllable null did not establish
that the readings were comparable or that the association exceeded existing stem/end patterns.
The new source rules and two gates were fixed before these simulations; the pattern itself was
already known, so this remains an exploratory audit rather than independent confirmation.

## Frozen results

Threshold: p < .05/7 = .007142857, with the Wilson 95% upper bound also below that threshold.
Both strict comparisons were required. Ties count as exceedances; p uses the plus-one rule.

| Inventory | Null grouping | Stems observed | Null mean | Exceedances / 9,999 | p | Wilson 95% exceedance interval | Pass |
|---|---|---:|---:|---:|---:|---|---|
| Strict | length | 6 | 1.5342 | 10 | .0011 | [.000543, .001840] | yes |
| Strict | length + final onset | 6 | 4.2191 | 1,604 | .1605 | [.153353, .167740] | **no** |
| Historical, descriptive | length | 12 | 2.7487 | 0 | .0001 | [0, .000384] | yes |
| Historical, descriptive | length + final onset | 12 | 7.2026 | 48 | .0049 | [.003623, .006358] | yes |

The strict inventory has **208 Linear A types** (248 accepted tokens in 121 inscriptions)
and **2,984 Linear B types** (6,721 tokens in 2,050 documents), all at least three signs long.
The historical inventory has 434 and 3,738 types respectively. There are 12 strict A stems
with any possible B `ro` counterpart; six currently have A `re` or `ru`. All 208 A slots can
change in the length null; 192 can change under the onset-conditioned null. No small-bucket
fallback or replacement sampling was used.

The retained pairs are `a-ta-re → a-ta-ro`, `a-ti-ru → a-ti-ro`, `di-de-ru → di-de-ro`,
`ka-sa-ru → ka-sa-ro`, `pa-ja-re → pa-ja-ro`, and **`qa-qa-ru → qa-qa-ro`**. Multiple faces or
attestations and alternative endings of one stem never add votes. KH 41's `ka-ta-re` is
excluded conservatively because gaps bound the run, not because its visible signs are wrong.

## Limits and next action

**Archive the specific adaptation lead; it fails the declared statistical gate.** Do not rerun
ending variants, relax the source filter, or repartition these same exposed words to claim
confirmation. Preserve the six pairs as unresolved spelling observations. Reopening requires
new source adjudication or independently dated, context-compatible attestations, with a new
question and protocol. No new language identification or exclusion follows from this audit.

This was a transcription-level source check, not an independent reading of every photograph.
The strict A filter requires three collated layers to agree, so coverage is limited and
selective. Unknown sign readings are excluded for this exact-spelling question. Null intervals
measure simulation error only; they do not quantify epigraphic uncertainty. The seven-probe
threshold is inherited for continuity and does not account for all earlier exploration.

No deviations from the committed protocol occurred. Before the freeze, a parser regression
was caught and fixed: a trailing broken-edge `[` on KN Dc 1272 .A must not invalidate intact
`a-ti-ro` on .B. The test fixes this scope at the transcription-line level. Synthetic checks
recover a planted ending association, give p = 1 for an all-ties case, and match optimised
permutation scoring against full corpus reconstruction; these are software checks, not evidence
of linguistic power at this corpus size.

## Reproduction and attribution

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py' -v
.venv/bin/python -m experiments.linear_a_correspondence verify
```

For an exact rerun, use a clean worktree at `75128e8`, attach the pinned `artifacts/` sources
and Python environment, then run `python -m experiments.linear_a_correspondence run`.
Existing outputs refuse overwrite. To regenerate extraction, use a scratch copy without
`inputs.json`, `accepted-attestations.json` or `pair-evidence.json`, run `prepare`, and compare
their SHA-256 values to the freeze; do not replace the released files.

Source scope, hashes and CC BY-NC-SA 4.0 attribution are in [PAIRS.md](PAIRS.md),
[PROTOCOL.md](PROTOCOL.md) and [freeze.json](freeze.json). SigLA: Ester Salgarella and
Simon Castellan, [database](https://sigla.phis.me/). DĀMOS: Federico Aurora (2015),
*DAMOS (Database of Mycenaean at Oslo). Annotating a fragmentarily attested language*,
[doi:10.1016/j.sbspro.2015.07.415](https://doi.org/10.1016/j.sbspro.2015.07.415).
The [DĀMOS search guide](https://damos.hf.uio.no/howto) explicitly distinguishes actual forms
from normalised searches that remove brackets and subscript dots.
