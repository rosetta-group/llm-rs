# Linear A, round three: name lexicons do not find Greek names in Linear B

Status: complete, 2026-09-23, branch `linear-a`. Protocol: [PROTOCOL.md](PROTOCOL.md), frozen by
[freeze.json](freeze.json) before the Knossos test. **The gate failed, so the protocol did not
run Linear A.** No reading of Linear A is proposed.

## What was done

- Pinned four open name lexicons ([sources.json](sources.json)): Greek (Wiktionary names),
  Anatolian (LAMAN, CC BY-SA 4.0), Levant (Oracc Ugarit, Amarna, Alalakh, CC0) and Babylonia
  (Oracc Rīm-Anum, CC0).
- Restricted every sample to entry words (a number follows), the position of names.
- Scored them with round one's statistic, unchanged: exact or near matches over a bigram null.
- Development on mainland DĀMOS, then the frozen test on Knossos.

## Why

Rounds one and two lost the signal in whole-language lexicons. Names are the bulk of
administrative words, so a names-only comparison was the strongest remaining lexical test.

## Results

| Stage | Greek identified | Other identified | Greek median z |
|---|---:|---:|---:|
| Development, 4 settings | 0% in all | 0% | −0.91 to 0.13 |
| Knossos test (θ = 0, two or more signs) | 0% of 20 | 0% | −0.64 |
| Gate needed | ≥ 90% | ≤ 5% | — |

1. **Linear B names match Greek names less often than chance words do.** Mainland entry words match a
   Greek name exactly 6.5% of the time, and pseudo-words from their own syllable model 8.1%. The
   pseudo-words recombine frequent syllables into typical shapes that short classical names share.
   Real names are more idiosyncratic.
2. **The Greek name list misses most Mycenaean names.** Of 32 Mycenaean names in Wiktionary, 9
   match a classical Greek name (`a-ki-re-u` Achilles, `a-re-ka-sa-da-ra` Alexandra,
   `ku-do-ni-ja` Kydonia). Most Linear B personal names left no trace in the classical lists.
3. **So the matches that are true are too few.** Examples found by chance and by truth mix
   (`e-re-u-te-ro` Eleutheros is real; `a-ko` Argo is not).

## What three rounds establish

| Round | Lexicon | Added information | Greek identified on Knossos-size control |
|---|---|---|---:|
| [One](../linear-a/REPORT.md) | whole languages | spelling only | 10% |
| [Two](../linear-a-context/REPORT.md) | whole languages | name vs position | 0% |
| Three | names only | entry words only | 0% |

With open lexicons, Linear B spelling and about 700 readable Linear A words, no lexical test
identifies Greek from Linear B, where the answer is known. So none can identify, or rule out, a
language for Linear A. The limits are the spelling (no final consonants, merged sounds), the
small corpus, and lexicons that lack the Bronze Age names the tablets use. None of these is
a compute problem.

## Records and reproduction

```bash
.venv/bin/python -m experiments.linear_a_names develop
.venv/bin/python -m experiments.linear_a_names verify
.venv/bin/python -m experiments.linear_a_names test
```
