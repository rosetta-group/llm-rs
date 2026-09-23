# Linear A: adapting the Voynich methods

Status: round one complete on branch `linear-a`, 2026-09-23: gate failed, see [REPORT.md](REPORT.md).
Sources were downloaded with the owner's approval and pinned. Nothing here is a reading of Linear A.

## Why the methods need adapting

Linear A is the reverse of the Voynich problem.

| | Voynich | Linear A |
|---|---|---|
| Script values | unknown | roughly known: most signs share a shape with a Linear B sign of known sound |
| Language | unknown; Naibbe tests assume Italian | unknown; the question itself |
| Text size | ~38,000 words | 1,884 records, 979 unique multi-sign words (Navarre corpus, 2026-09-18) |
| Content | continuous pages | administrative lists: names, commodities, numbers, totals |
| Known anchors | none | numerals, fractions, `ku-ro` = total (arithmetic), toponyms such as `pa-i-to` (Phaistos) |

So the Naibbe task (recover a key, language given) becomes: **given approximate sign values,
which language, if any, do the words come from?** And the Naibbe length finding applies with
more force: recovery needed 2,600 to 5,200 letters of running text; Linear A has under a
thousand distinct words, most of them probably personal names.

## Terms

- **Spelling rules:** the Linear B conventions that turn a word into CV signs: final and
  most cluster consonants dropped, r/l merged, voicing and aspiration not written.
- **Lexical match:** a candidate word of language L whose spelled form is within a small,
  scored distance of a Linear A word.
- **Identifiability:** at Linear A's size, the matcher ranks the true language first among
  decoys in controls where the answer is known.

## Track mapping

| Voynich track | Linear A version | Reason |
|---|---|---|
| Prediction (closed) | dropped | too little text; the Voynich result already showed prediction does not detect meaning |
| Corpus statistics | ported: word length, sign frequency, inflection-like final alternation (Kober triplets), each against Linear B cut to the same size | cheap, needs only the corpus |
| Images | dropped | tablets carry no depictions linked to text |
| Cipher recovery | replaced by language matching under known sign values, validated first by controls | values are known; language is not |
| Language-ID control | reused as the decoy design | same question: does a wrong language also "fit"? |

## Phases

```text
phase 0  data: pin corpus, lexicons, Linear B; write sources.json; check ku-ro sums
phase 1  anchors: do assumed values reproduce known toponyms and totals?  (validates data + values)
phase 2  statistics: Linear A vs Linear B at matched size
phase 3  power analysis on controls, before touching Linear A:
           for each control language L and decoy set D:
             build a Linear-A-sized word sample (same type count, same name share)
             spell it with the spelling rules
             run the matcher against L and D
           record: rank of L, share of correct matches, by sample size
           negative controls: shuffled Linear A signs; random CV strings with Linear A sign frequencies
phase 4  gate: apply to Linear A only if phase 3 ranks L first in >= 90% of draws at
           Linear A size and the negative controls never produce a winner
           else: report "not identifiable at this size" and stop
phase 5  Linear A run, frozen before execution; results reported with controls
```

Positive controls, strongest first:

1. **Linear B to Greek.** Same script family, same document types, known answer. Cut to Linear A's
   size. This is the closest real analogue.
2. **Synthetic Linear-A-style corpora.** Real lexicons of Hittite, Luwian, Akkadian, Ugaritic,
   Hebrew, Etruscan and Greek, spelled with the rules, mixed with a name share taken from Linear B.

Candidate languages for phase 5 are the same set, so each serves as a decoy for the others.

## Data (pinned in [sources.json](sources.json))

| Data | Source | Licence | Note |
|---|---|---|---|
| Linear A corpus | [Navarre-AI/linear-a](https://github.com/Navarre-AI/linear-a) `linear_a/data/corpus.json`, pinned commit | CC BY 4.0, SigLA-derived fields CC BY-NC-SA 4.0 | derived files here must then carry CC BY-NC-SA |
| Linear B words with Greek glosses | Wiktionary Mycenaean Greek lemmas (kaikki.org extract) | CC BY-SA | DAMOS has no stated bulk licence; NeuroDecipher's list has no licence |
| Candidate lexicons | Wiktionary extracts per language (kaikki.org) | CC BY-SA | uneven size: Hittite and Akkadian small, Hebrew and Greek large; size is controlled in phase 3 |

## Gate and expected outcome

The gate is phase 4. The likely outcome, given the Naibbe length result and an earlier
pre-registered study that found only 9.2% Mycenaean lexicon coverage of Linear A words
([Zenodo 22854212](https://zenodo.org/records/22854212)), is a negative: no language
identifiable at this size. That would still be a result: it says how much new text a
language test would need.
