# Linear A, round four: seven targeted probes

Status: written before any probe was run, 2026-09-23, branch `linear-a`. CPU only.
Rounds one to three showed that whole-lexicon matching has no power at Linear A's size.
These probes use small, specific lists and aggregate profiles instead. Hand-entered lists are
in [lists.json](lists.json), each with its source and a confidence tag taken from the research
brief ([BACKGROUND.md](../linear-a-next/BACKGROUND.md)). Items tagged [U] there are used as given
and flagged in the results.

Common pieces: Linear A word types with context labels (`linear_a/contexts.py`, 696 types);
DĀMOS Linear B types (all sites); spelling rules and matcher from round one; a chance null of
pseudo-words from a syllable bigram model of the word set under test, same lengths. **Every
probe reports a one-sided p-value against its null; the family threshold is 0.05 / 7 = 0.007.**
A probe that passes is a lead to be checked by a scholar, not a reading.

| # | Probe | Question | Measure and null | Control with a known answer |
|---|---|---|---|---|
| 1 | Egyptian toponyms | Do Kom el-Hetan Cretan names (Egyptian script, c. 1370 BCE) occur in Linear A? | Consonant-skeleton matches (Egyptian s, n, m, r may drop in clusters; t ~ d; glides ignored) per name; count of names with a Linear A match vs 200 null corpora | The same names must find their Linear B forms (ko-no-so, a-mi-ni-so, pa-i-to, ku-do-ni-ja, ru-ki-to, ku-te-ra3) in DĀMOS: at least 5 of 6 |
| 2 | Keftiu names | Do the 7 "Keftiu names" of writing board BM EA 5647 occur among Linear A words? | Near matches (distance ≤ 0.25) and shared first two syllables; count vs 200 null corpora | None available; exploratory |
| 3 | Theonyms | Are words on libation vessels closer to divine names than administrative words are? | Excess match rate of libation words minus that of tablet entry words, against LAMAN deities, Oracc divine names (Levant, Babylonia) and Greek gods; permutation of the libation / tablet label, 2,000 times | Wiktionary Mycenaean theonyms (di-wo, e-ma-a2, a-te-mi-to, ...) must match the Greek god list above chance |
| 4 | Trade words | Do Bronze Age trade-word stems occur in Linear A, and next to the right commodity sign? | (a) Stems from lists.json (LB and Semitic forms, first two syllables, vowel of the second free): Linear A words starting with a stem vs null. (b) Gordon's proposals (ku-ni-su with grain, HT 31 vessel words, ku-ro as total): context check, descriptive | (a) The LB stems must be found in DĀMOS words at a Linear-A-sized sample above chance |
| 5 | Grammar profile | Which profile do Linear A words resemble? | Profile per word list: share of types with a same-stem partner differing only in the last syllable, only in the first, final-vowel shares, final-syllable entropy, mean length. Euclidean distance on standardised features, 20 samples of equal size | LB words must be nearest to Greek running words among {Greek, Akkadian}, and LB entry words nearest to Greek names among the four name lists, in at least 18 of 20 samples |
| 6 | Adaptation rules | Which Linear A → Linear B spelling changes are regular? | One-syllable substitutions between Linear A and Linear B types of three or more signs, by position; discovery half and confirmation half of Linear A types (split by hash); a rule found in the first half (≥ 3 pairs, z ≥ 3) is confirmed if its pooled excess in the second half has p < 0.007 against 100 null runs. Pre-named rule: final re/ru → ro | — |
| 7 | Function transfer | Do carried-over words keep their role? | Carried-over set: Linear A types with a Linear B match (exact, or through a confirmed rule of probe 6). Agreement between Linear A label (entry or not) and the Linear B label; permutation, 10,000 times. Plus a table of each word's Linear B series (A people, D sheep, F offerings, ...) | — |

## What is not tested

Luwian and Hurrian running text (no open source found; eDiAna has no export), Hattic, the
Keftiu spells (no word division), and Mari and Ugarit sources (no Cretan names in them).
