# Linear A, round three: are Linear A entry words names from a known onomasticon?

Status: written before any development result, 2026-09-23, branch `linear-a`. CPU only.
Rounds one and two ([one](../linear-a/REPORT.md), [two](../linear-a-context/REPORT.md)) matched
all words against whole-language lexicons; chance matches swamped true ones. Most Linear A words
that head an entry (a number follows) are likely personal or place names, so this round compares
only those words with only proper names.

## Data

- **Samples:** word types whose majority context is `entry` (`linear_a/contexts.py`). Linear A:
  330 types of two or more signs, 214 of three or more. Linear B control: DĀMOS, mainland for
  development, Knossos for the test (the round-two download; its Knossos split was scored once,
  in round two, by a different statistic).
- **Name lexicons** (`linear_a/names.py`, pinned in [sources.json](sources.json)):
  Greek (Wiktionary names, 4,976 spelled forms), Anatolian (LAMAN names in Hittite texts, 4,287),
  Levant (Oracc: Ugarit, Amarna, Alalakh, 2,975), Babylonia (Oracc: Old Babylonian Rīm-Anum, 607).
  Egyptian is left out: its names are unvocalised, and round one showed consonantal lexicons give
  false winners.

## Statistic

Round one's statistic, unchanged (`linear_a/matching.language_scores`):

```text
for each name lexicon L:
  m    = sample words whose nearest L name is within theta
  null = 10 pseudo-samples from a syllable bigram model of the sample itself, same lengths
  z_L  = (m - mean(null m)) / max(sd(null m), 1)
identified = argmax z_L if z_L >= 3 else none
```

## Stages and gates

1. **Development**, mainland entry words: 20 samples of the Linear A entry count at each minimum
   length. Grid: `theta` in {0, 0.2} x minimum length in {2, 3} signs. Choose the setting with the
   highest share of samples where Greek is identified, among settings where another lexicon is
   identified in at most 5%. Ties go to `theta` = 0, then to minimum length 2.
2. **Freeze**, commit.
3. **Knossos test**, entry words, 20 samples. Gate: Greek identified in at least 90%, another
   lexicon in at most 5%.
4. **Linear A** only if the gate passes: all entry words at the frozen minimum length, 50 null
   samples, `z` per lexicon, and the matched names of the top lexicon.

## What would count

A lexicon identified on Linear A with the gate passed would say Linear A entry words resemble
that region's names more than chance: a lead about the name stock, not about the language of
the texts. Names cross language borders, so any lead would need a test on non-name words.
