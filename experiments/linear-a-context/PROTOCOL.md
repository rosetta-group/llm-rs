# Linear A, round two: does meaning predict tablet position?

Status: written before any development result, 2026-09-23, branch `linear-a`. CPU only.
Round one ([report](../linear-a/REPORT.md)) showed that spelling matches alone cannot separate
languages: 37% of Linear-A-shaped words match some Greek lemma by chance.

## Idea

On an administrative tablet, the word right before a number is usually a person or a place
(Linear B: `a-mi-ni-so ... S 1`, "to Amnisos"). A chance match picks a random lemma, so its
class (name or common word) is unrelated to the word's position. A true match should agree:
names in entry position, common words elsewhere. This adds information that spelling erases.

## Data

- **Linear B control:** DĀMOS (CC BY-NC-SA 4.0), downloaded 2026-09-23 with the owner's approval,
  5,932 documents. **Development split:** every site except Knossos. **Test split:** Knossos, the
  Cretan archive closest to Linear A, not scored before the freeze.
- **Linear A:** the round-one collation, now parsed from its Unicode lines so that numbers and
  logograms are visible (`linear_a/contexts.py`); 696 readable word types.
- **Lexicons:** round one's eight, with a name flag from the Wiktionary part of speech
  (`linear_a/lexicons_v2.py`).

## Statistic

```text
label(w)   = majority context of type w: entry | logogram | header | other
for each language L:
  matched  = sample words whose nearest L form is within theta
  A_L      = matched words where (form is a name) == (label is entry)
  null     = 1000 permutations of labels among words of the same syllable length
  z_L      = (A_L - mean(null)) / max(sd(null), 1)
identified = argmax z_L if z_L >= 3 else none
```

## Stages and gates

1. **Development** (mainland only): 20 samples of 696 word types. Choose `theta` in {0, 0.2}
   and the name rule in {any entry is a name, most entries are names} to maximise the share of
   samples where Greek is identified, among settings where another language wins in at most 5%.
   Ties go to `theta` = 0, then to rule "any".
2. **Freeze**, commit.
3. **Knossos test:** 20 samples of 696 types. Gate: Greek identified in at least 90%, another
   language identified in at most 5%.
4. **Linear A** only if the gate passes: all 696 types, 10,000 permutations, `z` per language.

## What would count

A language identified on Linear A with the gate passed would be a lead, not a reading. It would
need an independent check (for example commodity words next to their logograms) before any
word is glossed.
