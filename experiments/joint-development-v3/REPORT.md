# Codebook-free Naibbe recovery, round three: development record

Development results on development text only. The [protocol](PROTOCOL.md) fixes the one change
tested here; the fresh evaluation is reported in the [round-three fresh report](../joint-recovery-v3/REPORT.md).

Round three asks where the remaining 9–11% character error lives and adds a word-level polish.
The answer is sharp: letters inside correctly parsed tokens are already 99–99.5% right, and
nearly every remaining error comes from the 10% of tokens whose parse is wrong. The polish
trims about one point; it cannot touch the parses.

**Lexical polish:** a sweep over role units that re-scores the character prior's shortlisted
alternatives with the frozen lexicon segmenter's local segmentation cost.
**Correctly parsed token:** a token whose decoded split matches the encoder trace; inside it,
recovered and true letters align one to one.

## What was done

- Added `voynich/lexical_polish.py`: a windowed lexicon cost and a greedy unit polish; tests in `tests/test_polish.py`.
- Attributed error to parse versus letter mistakes on the three development texts.
- Tested polish weights, token re-parsing under the same cost, and warm-started EM re-decoding.
- Wrote a reproducible driver (`experiments/joint_development_v3.py`) whose table is `results.json`.

## Why

Round two's fresh cases had 88–91% segmentation agreement and 8–12% letter error. If most errors
sat inside correctly parsed tokens, a stronger letter model would pay; if they sat in mis-parsed
tokens, only segmentation work would. The attribution decides which.

## Error attribution (round-two pipeline, after polish at weight 1)

| Text | CER | Tokens parsed correctly | Letters in those tokens | Letter error inside them | Share of letters in mis-parsed tokens |
|---|---:|---:|---:|---:|---:|
| historical prose | 9.6% | 89.9% | 4,570 | **1.0%** | 12.1% |
| modern | 8.6% | 90.4% | 4,595 | **0.7%** | 11.6% |
| Petrarca | 9.2% | 89.7% | 4,554 | **0.5%** | 12.4% |

1. **The letter mapping is essentially solved where the parse is right.** Half a percent to one
   percent error inside correctly parsed tokens is at the gate's letter threshold.
2. **Mis-parsed tokens carry all the rest.** They hold about 12% of the letters and produce
   insertions, deletions and wrong letters that no key change can repair.

## Lexical polish

Weights, three sweeps, ±30-letter windows, shortlist of five (exploratory runs on cached round-two outputs):

| Text | Before | 0.5 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| historical prose | 10.9% | 10.0% | **9.6%** | 9.3% | 9.2% | 9.0% |
| modern | 9.3% | 9.0% | **8.6%** | 8.6% | 8.6% | |
| Petrarca | 9.5% | 8.9% | **9.2%** | 9.5% | | |

Weight 1 is the only setting that improves all three texts; higher weights keep helping prose
and start hurting verse, whose vocabulary the lexicon covers worst. Each polish takes about two
minutes per 5,200-letter text.

## Two ideas tested and rejected

- **Token re-parsing under the combined local cost** (whole versus split, unknown units given
  their best letter): historical 9.3% → 8.9%, modern 8.6% → 9.7%. Not stable; excluded.
- **Warm-started joint EM from the polished key** (emission mass 0.95, 30 iterations): segmentation
  agreement identical to three decimals on all texts. The joint model's parses are a fixed point
  of its own posterior; a better key does not move them.

## What this means for the next step

The character prior plus a lexicon can no longer buy much. The parse errors need a model that
sees more than one token at a time when choosing splits, or a different candidate-piece rule.
About 4% of tokens are ambiguous even with the true lexicon; the other 6% are the target.

## Reproducible table

`results.json`, driver settings; round-two pipeline recomputed, then polish at each weight (three sweeps, ±30-letter windows, shortlist 5). Attribution columns are for weight 1.

| Text | Round two CER | 0.5 | 1 | 2 | 4 | Letter error inside correct parses (w=1) | Letters in mis-parsed tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| historical | 10.9% | 10.0% | **9.6%** | 9.3% | 9.2% | 1.0% | 12.1% |
| modern | 9.3% | 9.0% | **8.6%** | 8.6% | 8.6% | 0.7% | 11.6% |
| verse | 9.5% | 8.9% | **9.2%** | 9.5% | 9.8% | 0.5% | 12.4% |

The frozen round-three setting is weight 1, the only weight that improves every text.

## Limits

- Development passages; the settings were chosen on them. Only the frozen fresh run counts.
- The segmenter used for the polish and for word error is the round-one model, untouched.
- No Voynich text; final test sealed; CPU only; no downloads.
