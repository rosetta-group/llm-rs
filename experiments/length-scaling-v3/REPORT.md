# Context reparse with key refit: mean CER 1.79% at 20,800 letters; the self-inclusive glue test collapses

Development only, 2026-09-24. Protocol declared before any run. The original commit is `3171cbb`
(2026-09-23 23:11), made on a local branch while another session had this checkout; it was
cherry-picked unchanged to master. Six parallel processes, about 75 minutes wall-clock for the
20,800-letter stage. No cap was hit except one refine cap in the glue arm.

## Mean polished CER over the three development texts ([summary.json](summary.json))

| Letters | Square-root baseline | **Context reparse** | Self-inclusive glue test |
|---:|---:|---:|---:|
| 10,400 | 3.19% | **2.45%** | 48.9% |
| 20,800 | 3.25% | **1.79%** | 59.6% |

Per text, CER / parse agreement / WER with v3 segmentation:

| Text | Baseline 20,800 | Reparse 10,400 | **Reparse 20,800** |
|---|---|---|---|
| Historical prose | 3.38% / 95.2% / 27.0% | 3.01% / 95.7% / 26.9% | **1.77%** / 96.3% / 20.9% |
| Modern ISDT | 2.57% / 95.8% / 19.6% | 2.18% / 96.5% / 19.0% | **1.31%** / 96.8% / **12.8%** |
| Petrarca verse | 3.80% / 94.9% / 34.4% | 2.15% / 95.9% / 29.2% | **2.28%** / 95.9% / 29.2% |

**Selection:** the reparse is eligible. At 20,800 letters it is 1.46 points below the baseline (the
rule needs 0.5), and at 10,400 it is lower, not higher. The glue test is not eligible. The reparse's
20,800 mean is at or below 2.81%, so **a sealed long-passage round is licensed**.

## Why the reparse works, and the earlier version did not

- **It needs few changes and converges.** The first round changes 55–77 parses at 10,400 letters and
  189–209 at 20,800. The second round changes only 0–18. The beam takes 1–13 s per round.
- **The key refit is the difference.** The rejected earlier reparse (−0.29 points) kept round four's
  key fixed. Here, refitting the key on the new parse gives a further 0.05–0.12 points before polish,
  and it lets polish start from a better key.
- **It gains more with length.** It cuts 0.74 points at 10,400 letters and 1.46 at 20,800, because
  context separates competing splits better when every letter pair has been seen more often.
  Spurious pieces remain in the lexicon (178–200), but the reparse picks the split reading when
  context favours it.
- **Verse gains least.** It goes from 2.15% to 2.28% between 10,400 and 20,800 letters. The prose+verse
  prior fits Petrarca's forms less well.

## Why the glue test collapses

Adding whole-parsed occurrences to the expectation also inflates it for genuine one-letter pieces
the EM parses as whole. Those then fail θ = 5 and are dropped. True whole tokens get split
(`whole->split` rises to 1,600–4,400), 69–91 true pieces go missing, and CER reaches 35–87%. The
correction needs a model of which hypothesis produced the whole parses, not a flat add-back. This is
a clean negative.

## Next

A sealed long-passage round: the round-five decoder with square-root-scaled thresholds, two rounds
of context reparse with key refit, and v3 segmentation. It runs on fresh ciphertext of 20,800 letters
per case, with the 1% CER / 10% WER gate reported per case. That needs fresh long text: Dante and
unused Compagni remain, and a modern half needs a newly pinned corpus.
