# Voynich transcription suspects: near-hapax forms are 2.6–12× commoner than in natural text

Analysis 2026-09-23, training pages only (148 pages, 2,878 lines). No decoding and no meaning
claim. Validation and final-test pages are untouched. CPU, 10 seconds. Code:
`experiments/voynich_suspects.py`.

## Signals

- **Near-hapax:** the form occurs once and is one edit from a form seen at least 5 times.
- **Flagged:** the transcriber marked the reading uncertain.
- **Uncertain space:** an uncertain space (`,`) touches the token.
- **Space disagreement:** v101 and EVA word counts on certain spaces differ by 2 or more on the line.

The two transcriptions use different glyph alphabets, so readings are not compared letter by
letter. Only word counts per line are compared.

## Results ([summary.json](summary.json))

| | v101 | EVA | Latin | Italian | Old French | German |
|---|---:|---:|---:|---:|---:|---:|
| Tokens | 25,768 | 24,113 | 25,000 | 25,000 | 25,000 | 25,000 |
| Hapax share of types | 71.2% | 70.6% | 50.7% | 64.4% | 58.5% | 69.8% |
| **Near-hapax share of tokens** | **10.4%** | **6.8%** | 0.9% | 1.5% | 2.6% | 1.5% |

1. **Near-hapax forms are 2.6–12× commoner than in these natural-language samples of equal size.**
   Most are therefore not transcription errors. They are the manuscript's own small variations of
   frequent words, the same local regularity the copy baselines found. Some are errors, but this
   filter cannot say which.
2. **Transcriber-flagged forms are few:** 45 in v101 and 520 in EVA. Uncertain spaces touch 2,820
   and 3,185 tokens.
3. **Word boundaries disagree often.** 1,365 of 2,878 lines differ in certain-space word count,
   and 447 differ by 2 or more. Word boundaries are the least reliable part of either transcription.
4. **Which glyphs get swapped.** Among 623 EVA single-glyph substitutions, the commonest pairs are
   e/o 37, o/y 29, a/o 28, s/y 27, f/k 17, n/r 16, k/p 14 and c/s 14. Some pairs are visually close
   in the script (a/o, f/k, k/p, n/r, c/s), which fits misreading. Others (o/y, s/y, d/y) sit at word
   endings and look more like morphology. This split is a reading of glyph shapes, not a test.

## Equal-size controls ([equal-size-controls.json](equal-size-controls.json))

These are near-hapax shares of tokens in 10,000-token windows, averaged over up to two disjoint
windows. The same filter and threshold apply throughout.

| Source | Near-hapax share |
|---|---:|
| Voynich v101 | 12.20% |
| Voynich EVA | 8.33% |
| Naibbe cipher (published sample) | 5.86% |
| Timm–Schinner generator (published sample) | 5.63% |
| Old French | 2.23% |
| Finnish (agglutinative) | 1.78% |
| German | 1.64% |
| Italian | 1.42% |
| Latin | 1.22% |

- **Cleaning does not remove the excess.** Merging uncertain spaces and dropping the 447 disputed
  lines gives 12.0% (v101) and 7.8% (EVA) at 20,000 tokens.
- **Natural languages stay at 1.2–2.2%, Finnish included.** Rich morphology alone does not produce it.
- **Both artificial controls sit at about 5.7%.** That is 3–4× natural text, but still below the
  Voynich text. So a verbose cipher or copying-with-modification produces this signature, and the
  published samples produce less of it than the manuscript.
- **The figure depends on the transcription alphabet.** 12.2% against 8.3% shows it. One EVA edit is
  often a pen stroke, not a letter. Each control is a single published sample.

## Visual-review list ([review-top50.csv](review-top50.csv))

This list has 164 candidates, and the file holds the top 50. Each is an EVA near-hapax of plain
letters, at least 4 glyphs long, one substitution from a word seen at least 50 times. Examples:
`dairn`/`daiin` (f102v1.8), `darin`/`daiin` (f50r.10), `daiig`/`daiin` (f52v.13) and `shddy`/`shedy`
(f66r.50). Each row gives the folio image path when the scan is labelled (foldout sub-pages are not).
Checking a row means looking at that line in the scan and asking whether the glyph could be the
frequent word's. [suspects.csv](suspects.csv) has every flagged token, and
[review-loci.json](review-loci.json) has the 1,182 lines flagged in both transcriptions.

## Implication

- **Don't correct the transcription with this filter.** Since near-hapax forms are several times
  commoner than in natural text, "correcting" them toward frequent words would erase real structure.
- **Sensitivity check for any later Voynich scoring.** Rerun it with flagged forms, uncertain spaces
  and lines with a space gap of 2 or more excluded. A result that moves is not robust.
- **The scans can settle a sample.** Reviewing 50 candidates against the images would estimate the
  misreading rate among them. That needs a human eye, or a glyph model trained on reviewed crops.
