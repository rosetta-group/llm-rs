# Exploratory: Linear A endings that change in Linear B

Status: exploratory, 2026-09-23, no protocol, no gate. Run after round three to choose a next step.

## What was done

- Paired every Linear A word type of three or more signs (`linear_a/contexts.py`) with every
  DĀMOS Linear B type of the same length that differs in exactly one syllable.
- Counted each substitution by position (initial, medial, final).
- Compared with 30 runs of the same count on pseudo-words from a syllable bigram model of Linear A.

## Why

Published comparisons of names shared by the two archives report that Linear A `-re`/`-ru`
becomes Linear B `-ro` (for example `ki-da-ro`, see [BACKGROUND.md](BACKGROUND.md) §8). A regular
change is evidence that the shared forms are the same names, adapted by Greek scribes, not
chance look-alikes.

## Result

| Substitution | Pairs | Chance (mean ± sd) |
|---|---:|---:|
| final `re` → `ro` (`a-du-re` / `a-du-ro`, `a-ta-re` / `a-ta-ro`) | 7 | 1.7 ± 1.6 |
| final `ru` → `ro` (`di-de-ru` / `di-de-ro`, `ka-ka-ru` / `ka-ka-ro`) | 5 | 0.7 ± 0.9 |
| any change keeping the consonant, final syllable | 44 | 33.3 |
| any change keeping the consonant, medial syllable | 35 | 21.6 |
| all one-syllable pairs | 781 | 776.2 |

1. **The predicted change is there.** `-re`/`-ru` → `-ro` gives 12 pairs against 2.4 expected.
   The hypothesis came from the literature before this count, but the count was made after
   looking at other substitutions too, so it is not a confirmatory test.
2. **Pairs that keep the consonant are over-represented.** This is the signature of the same word
   written twice with a vowel adjusted, and it holds in every position.
3. **Total pair counts equal chance.** Most one-syllable look-alikes are chance; the signal is in
   a few specific, regular changes.

Reading: consistent with Greek scribes giving Minoan names ending in -e/-u a Greek o-stem
ending (-os). It says nothing about the meaning of any Linear A word.
