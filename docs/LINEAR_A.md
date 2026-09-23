# Linear A: what three frozen rounds found

This page summarises the Linear A track (branch `linear-a`, 2026-09-23). It does not contain a
translation. It records whether word-matching methods can tell which language Linear A is, and
why they cannot at present. Round records: [one](../experiments/linear-a/REPORT.md),
[two](../experiments/linear-a-context/REPORT.md), [three](../experiments/linear-a-names/REPORT.md).

## The question, and how it differs from Voynich

Linear A is the reverse of the Voynich problem. Most sign sounds are roughly known, because
about 60 signs share a shape with Linear B signs of known value; the language is unknown. So
the Naibbe task (recover a key, language given) becomes a language test: do Linear A words
look like the words of some known language more than chance allows?

The corpus is small and administrative: 1,884 records, 696 readable word types of two or more
signs, and 330 of those in *entry* position (a number follows, as for a person or place on a
list). Every method had to pass first on Linear B, which is Greek, cut to Linear A's size.

## What the data does support

| Check | Result | Chance |
|---|---:|---:|
| Linear B Cretan place names found in Linear A (`pa-i-to` Phaistos, `se-to-i-ja`) | 2 of 14 | 0.025 |
| Linear A word types also found in Linear B | 94 | 72.7 (max 88) |
| `ku-ro` "total" equals the sum of its entries (HT 13: 130 = 130) | 8 of 37 | — |

The assumed sign values reproduce known place names, and about 20 Linear A words survive into
the Linear B archives beyond chance. These support the values; they say nothing about grammar.

## Why every language test failed

| Round | Method | Greek found in Linear B control | Needed |
|---|---|---:|---:|
| One | whole-language lexicons, exact spelling match | 10% | 90% |
| Two | as one, plus: is the match a name where the tablet puts a name? | 0% | 90% |
| Three | entry words only, against proper-name lists | 0% | 90% |

1. **Spelling erases the words.** Linear B spelling drops final consonants, merges r and l, and
   does not write voicing. 37% of Linear-A-shaped pseudo-words match some Greek lemma exactly;
   real Mycenaean words match at 45%. True matches exist (`a-ko-ra` ἀγορά) but are only
   8 points above chance.
2. **Position adds nothing detectable.** Greek-matched names sit in entry position 35% of the
   time and Greek-matched common words 34%. Most Linear B names are absent from dictionaries.
3. **Name lists miss Bronze Age names.** Only 9 of 32 Mycenaean names appear in the classical
   Greek name list; Linear B entry words match Greek names *less* often than chance words do
   (6.5% against 8.1%).

A fourth round tried seven targeted probes (Egyptian and Keftiu names, gods, trade words, name
profiles, spelling rules, role transfer); none passed its threshold
([report](../experiments/linear-a-probes/REPORT.md)). Greek scribes turning Minoan `-re`/`-ru` names
into `-ro` gives 12 pairs against 3.1 by chance, supported but not confirmed.

A fifth round compared grammar profiles with Anatolian and Hurrian running text (TLHdig). Linear A
came out nearest Hittite in 20 of 20 samples, but so did shuffled Linear A syllables; the match
comes from syllable and vowel frequencies (cuneiform, like Linear A, rarely shows *o*), not from
words ([report](../experiments/linear-a-tlhdig/REPORT.md)).

So with open lexicons and this corpus, no lexical test finds Greek where the answer is known.
None can identify or rule out a language for Linear A. Compute is not the limit: information is.

## Data and licences

Linear A: Navarre-AI collation of SigLA and lineara.xyz (CC BY; SigLA fields CC BY-NC-SA 4.0).
Linear B: DĀMOS, 5,932 documents (CC BY-NC-SA 4.0). Lexicons: Wiktionary via kaikki.org (CC BY-SA),
LAMAN (CC BY-SA 4.0), Oracc (CC0). Results derived from SigLA and DĀMOS carry CC BY-NC-SA 4.0.
