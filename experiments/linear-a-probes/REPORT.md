# Linear A, round four: seven targeted probes, no lead survives

Status: complete, 2026-09-23, branch `linear-a`. Protocol and lists committed before any run
([PROTOCOL.md](PROTOCOL.md), [lists.json](lists.json)); results in `results-1.json` to
`results-7.json`. Family threshold p < 0.007. No reading of Linear A is proposed.

## What was done

- Built seven probes from the research brief ([BACKGROUND.md](../linear-a-next/BACKGROUND.md)).
  They use short, specific lists (Egyptian place names, Keftiu names, gods, trade words) or
  aggregate profiles instead of whole lexicons.
- Gave each probe a chance null (pseudo-words from Linear A's own syllable model) and, where one
  exists, a Linear B control with a known answer.
- One bug fix during the run (probe 2 could not sort its output); no result had been produced.

## Why

Rounds one to three showed that whole-lexicon matching has no power at Linear A's size. Small
lists lower the chance rate, and profiles pool evidence across hundreds of words.

## Results

| # | Probe | Linear A result | Chance | p | Control |
|---|---|---:|---:|---:|---|
| 1 | Egyptian (Kom el-Hetan) place names | 8 of 8 names match some word | 7.9 | 0.89 | passed: 6 of 6 Linear B forms found |
| 2 | Keftiu names (BM EA 5647) | 6 of 7 names have a look-alike | 5.9 | 0.70 | none possible |
| 3 | Libation words vs god lists | not closer than tablet words | — | 0.21–1.0 | failed: 3 vs 2.4 |
| 4 | Trade-word stems | 19 of 27 stems found | 18.3 | 0.44 | failed: 0 of 20 samples |
| 5 | Name-stock profile, entry words | nearest Levant, 20 of 20 | — | — | names part passed (Linear B → Greek 20 of 20); running-word part failed (15 of 20) |
| 6 | New Linear A → B spelling rules | none confirmed | 0.8 | 0.16 | — |
| 6 | Pre-named `-re`/`-ru` → `-ro` | 12 pairs | 3.1 | 0.0099 | seen before the protocol |
| 7 | Carried-over words keep their role | 65 of 114 agree | 58.2 | 0.13 | — |

1. **The loose probes find everything.** A consonant skeleton or a two-syllable stem matches
   something in any 700-word list, so probes 1, 2 and 4 score at chance. Probe 1's control shows
   the Egyptian renderings do fit the Linear B names. The test is too permissive to be specific
   in Linear A.
2. **The god and trade controls fail.** A method that cannot find Greek gods in Mycenaean
   theonyms, or Linear B trade words in Linear B, says nothing about Linear A.
3. **The Levant lead dissolves.** Linear A entry words are far from every name stock (distance
   5.1 Levant, 6.0 Greek, 6.2 Anatolian, 7.0 Babylonia). They are shorter than the names in any
   list (2.9 syllables against 3.5–4.2). In a post-hoc check with length-matched references,
   they split 11 Levant and 9 Greek; Linear B entry words still go to Greek 20 of 20.
4. **The name adaptation is real but not proven here.** `-re`/`-ru` → `-ro` gives 12 pairs against
   3.1. The p-value, 0.0099, is the smallest 100 null runs can give. That is a design flaw of this
   protocol (the threshold 0.007 was unreachable), and the rule was seen before the protocol.
5. **Carried-over words do not keep their tablet role detectably** (p = 0.13), so the Linear B
   archive does not yet give a reliable role for any Linear A word.

## What this does and does not establish

Short foreign lists and profile statistics do not identify a source for Linear A names or words.
The one regular pattern, Greek scribes turning Minoan `-re`/`-ru` names into `-ro`, is supported
but not confirmed. The Keftiu names are unverified transcriptions; the Egyptian list has only two
Cretan names with known Linear A counterparts. Luwian and Hurrian running text were not available.

## Records and reproduction

```bash
.venv/bin/python -m experiments.linear_a_probes      # refuses to overwrite results-*.json
```
