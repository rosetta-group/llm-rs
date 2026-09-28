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

## Addendum: rerunning joint EM on the unpruned lexicon (variant D)

The true suffixes such as `chdy` pass the frozen candidate minimum and are pruned later, so the
first stage was repeated from A's state on the unpruned candidate lexicon plus A's pieces, then
refined and reparsed as A. Scratch diagnostic, same 24 positives.

| Variant | Median cost | Over 0.50 | Better than A |
|---|---:|---:|---:|
| A | 0.344 | 5 | — |
| D, EM warm-started from A's key | 0.347 | 6 | 6 of 24 |
| D, EM from four random starts | 0.412 | 7 | 4 of 24 |

A larger lexicon lets EM choose more wrong splits, not the right one. Not adopted. Fixing the split
point needs a constraint EM does not have, not more candidate pieces.

## Addendum: split-point moves by description length (`voynich/split_shift.py`)

Greedy moves that re-split every token using a new piece, or shift one glyph run across all tokens,
accepted when the key-search description length falls. Scratch diagnostic, same 24 positives: median
cost 0.344 → 0.335, better in 14 of 24, cases over 0.50 unchanged at 5; some accepted moves go the
wrong way (`p:qoch`). The description length does not prefer the true split. Successor variety
(Harris) points the wrong way too: A's split has the higher variety in 2,390 of 2,871 wrongly split
tokens, the true split in 343. At 5,200 letters the split point may not be identifiable from the
ciphertext alone. Not adopted.
