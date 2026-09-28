# Decoder C (joint whole and half admission): small gain; the real error is the split point

Development diagnostic on the 24 released positives of the second confirmation, true-language prior
only; answers used to score cost. Code: `experiments/lexicon_admission_development.py`,
`voynich/half_admission.py`.

**Decoding cost:** transfer excess of the decoded text minus that of the true text, same prior.

C lowers the median decoding cost from 0.344 to 0.311 and is better in 19 of 24 cases, but the
number of cases over the 0.50 ceiling stays at 5. It admits a median of 2 new pieces per case.
Not adopted; no full development run.

## Why C cannot close the gap

Missing true units after A, over the 24 fit passages:

| Occurrences in the passage | Units | Token occurrences |
|---|---:|---:|
| 1 | 758 | 758 |
| 2–4 | 526 | 1,345 |
| 5 or more | 172 | 1,398 |

For the frequent missing units, what A does with the tokens that contain them:

| A's reading | Tokens |
|---|---:|
| split at the wrong point | 1,224 |
| true split read as one whole | 54 |
| true whole split | 30 |

The wrong split point is systematic: `ch`/`sh` moves from the suffix to the prefix, for example
`l`+`chdy` → `lch`+`dy`, `qo`+`chdy` → `qoch`+`dy`, `t`+`shody` → `tsh`+`ody`. The missing units are
mostly suffixes (1,136 of 1,398 occurrences). A's key compensates on the fit passage, so a context
test sees little saving; the error shows on the transfer passage.

## Next

Target the split point: compare the two lexicon factorizations that explain the same tokens, for
example by piece-inventory description length or by prefix-final versus suffix-initial glyph
statistics, then check on these released cases before any fresh test.
