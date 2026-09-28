# Where decoder A's decoding cost comes from: missing lexicon pieces

Development diagnostic on the 24 released positives of the second confirmation, true-language prior
only. It uses the answers, so it is not evidence for any method. Code: `experiments/recovery_oracles.py`.

**Decoding cost:** transfer excess of the decoded text minus that of the true text, same prior.

Missing lexicon pieces are the main cause. Giving the fitter every missing true piece removes about
60% of the reducible cost; fixing letters alone removes about 20%.

## Results (medians over 24 positives)

| Variant | What is supplied | Transfer cost | Transfer CER |
|---|---|---:|---:|
| A | nothing (the decoder as confirmed) | 0.344 | 5.5% |
| whole | missing true whole-token pieces | 0.259 | 4.1% |
| halves | missing true first/second pieces | 0.459 | 6.8% |
| lexicon | every missing true piece | 0.190 | 2.7% |
| segment | the true segmentation | 0.189 | 2.6% |
| letters | true letters on A's own segmentation | 0.295 | 4.5% |
| truth | true segmentation and letters | 0.089 | 1.6% |

Median missing pieces per fit passage after A: 33 whole-token and 27 half pieces.

## What it shows

1. **The floor is 0.09.** Even the true key costs that much on unseen tokens of the second passage.
2. **The lexicon carries most of the rest.** A complete lexicon (0.190) is as good as the true
   segmentation (0.189).
3. **Half pieces help only together with whole pieces.** Added alone they raise the cost to 0.459:
   more tokens split wrongly.
4. **Letter errors are secondary.** Correct letters on A's segmentation save 0.05.
5. **The fitted text looks as typical as the true text** (fit cost about 0.00), so the fit passage
   cannot show its own errors; transfer to a second passage is what exposes them.

## Next

Admit missing whole and half pieces jointly and feed them back through joint EM, rather than
re-segmenting directly; then check on released blocks, then a fresh confirmation.
