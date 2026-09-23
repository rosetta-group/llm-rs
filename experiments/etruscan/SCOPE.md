# Etruscan: word meaning from formula context

Status: round one done, branch `etruscan`, 2026-09-23 ([REPORT.md](REPORT.md)): the gate passed
narrowly for M2 (44% balanced accuracy on Latin and Etruscan, nulls at 20%), but its KIN and
LIFE calls are right only 6% of the time. Sources pinned in [sources.json](sources.json).
Nothing here is a reading of Etruscan.

## Why Etruscan is a different task

Linear A and the Voynich manuscript are decipherment problems: the sound or key is unknown.
Etruscan is not. The alphabet is read, and names, numerals and the common funerary words are
known. What is missing is the meaning of many rarer words.

| | Linear A | Etruscan |
|---|---|---|
| Sign values | roughly known | known (Greek-derived alphabet) |
| Language | unknown | known to be Etruscan; no close relative is attested except Raetic and Lemnian |
| Corpus | about 700 readable words | about 12,000 inscriptions known; 7,139 in the Larth dataset |
| Content | administrative lists | mostly short funerary and ownership texts, repeated formulae |
| What a method could add | which language | which meaning class an unglossed word belongs to |

So the question becomes: **from where a word appears in formulae, can a method tell what kind
of word it is, and does it do so reliably on words whose meaning is already known?**

## Terms

- **Formula:** a recurring template, such as name + father's name + `clan` ("son") + `avils`
  ("years") + numeral. Etruscan epitaphs are dominated by a few such templates.
- **Meaning class:** a coarse label for a word: praenomen, family name, kinship term, numeral,
  age or time word, verb (dying, giving, making), object or tomb word, god name, other.
- **Glossed word:** a word with a meaning given in the Larth word list (from Wallace's *Zikh
  Rasna*). Unglossed words are the target.
- **Circularity:** scholarly glosses were themselves found largely by the combinatory method,
  which is also distributional. Agreement with them shows the method reproduces that method,
  not that either is true.

## Method, simplest first

```text
build a context vector per word type:
    left and right neighbours, position in the text, suffix, next to a numeral or not
seed: a share of glossed words with their meaning class
predict the class of every other word:
    1. nearest labelled neighbour (PPMI + SVD vectors)
    2. label propagation over the word graph
    3. only if 1 and 2 pass the gate: a small language model trained on the corpus
```

The first two are cheap and fully inspectable. A language model is added only if it beats them
on the controls.

## Phases

```text
phase 0  data: pin Larth and a Latin control corpus; write sources.json; count tokens and types
phase 1  anchors: do known formula words come out as expected?
           clan, sec, puia (kinship); avils, lupu (age, death); ci, zal, huθ (numerals)
phase 2  Latin control at Etruscan size (known answer):
           take Latin epitaphs, cut to the Etruscan text count and length distribution
           hide all glosses; seed with the same number of words as Etruscan has glossed
           predict classes of held-out Latin words; score against a Latin lexicon
phase 3  Etruscan held-out control:
           split glossed words 80/20 by type; seed with 80%, predict 20%
           repeat over 20 splits
negative controls (both phases):
           shuffle word order inside each text (breaks formula position)
           permute seed labels
           both must fall to the majority-class baseline
phase 4  gate: apply to unglossed words only if
           Latin and Etruscan held-out accuracy both beat the majority baseline by >= 20 points
           and both negative controls stay within 5 points of it
           else: report "formula context does not carry meaning class at this size" and stop
phase 5  fresh test, frozen before execution:
           seed with the Larth/Zikh Rasna glosses only
           evaluate on words glossed in a source not used anywhere above
           (candidates: Wiktionary Etruscan entries; the CIEP translation column)
phase 6  predictions for unglossed words, ranked by confidence, reported as hypotheses
```

The shuffled-order control comes from the Linear A lesson: round five's Hittite match also
appeared for shuffled syllables. Any class signal must vanish when formula order is destroyed.

## Data (downloaded 2026-09-23 with approval; hashes in [sources.json](sources.json))

| Data | Source | Licence | Size | Note |
|---|---|---|---|---|
| Etruscan texts | [Larth](https://github.com/GianlucaVico/Larth-Etruscan-NLP) `Data/Etruscan.csv` | CC BY 4.0 | 302 KB; 7,139 texts (561 ETP, 6,578 CIEP) | CIEP part was extracted from PDF and is noisy; ETP part is clean |
| Etruscan word list with glosses | Larth `Data/ETPWords.txt`, `ETPNames.txt`, `ETPSuff.txt`, `ETP_POS.csv` | CC BY 4.0 | 15 KB, 24 KB, 1 KB, 165 KB | 1,122 words, 956 glossed, 54 grammatical features |
| Latin control | [LIRE v3.0](https://zenodo.org/records/8431452) (EDH + EDCS aggregate), epitaphs only | CC BY 4.0 | 62 MB parquet | same records as the 474 MB geojson; Larth pinned at `daf4972` |
| Fresh-test glosses | kaikki.org Wiktionary Etruscan extract | CC BY-SA | small | not downloaded; needs approval before phase 5 |

The Larth paper ([arXiv 2310.05688](https://arxiv.org/abs/2310.05688)) reports that CIEP and
ETP use different transliterations and that Larth's normalisation is not reversible. Phase 0
records both and runs phases 1 to 3 on ETP alone and on ETP + CIEP.

## Phase 0 result (description only)

`python -m experiments.etruscan_phase0`. The open corpus is far smaller than "7,139 texts" suggests.

| | ETP part | CIEP part | Latin epitaph pool (LIRE v3.0) |
|---|---|---|---|
| Texts (after joining CIEP lines by CIE number) | 551 | 4,240 | 65,740 |
| Tokens | 2,433 | 7,356 | 858,317 |
| Tokens per text | 4.4 | 1.7 | median 11 |
| Tokens of 12+ letters | 0.9% | 6.2% | |
| Texts using o, b, d or g (not Etruscan letters) | 2.2% | 7.7% | |
| Tokens with a gloss in the ETP word list | 78% | 15% | |

1. **Duplicates.** 444 of 7,139 Larth rows are exact repeats; they are dropped.
2. **CIEP rows are lines, not texts.** One CIE number spans many rows. Joined, the part is 4,240
   texts, 60% of them one token long.
3. **CIEP word division is broken.** PDF extraction runs words together (`milarthasarsinaia` for
   *mi larthas arsinaia*); long tokens are seven times more common than in ETP.
4. **CIEP holds non-Etruscan text.** CIE 15896 is the Umbrian Iguvine Tables (`ikuvina`,
   `pesnimu`); 7.7% of CIEP texts use letters Etruscan lacks.
5. **The clean part is small and mostly glossed.** ETP has 2,433 tokens and 1,173 types; 78% of
   tokens already have a gloss, because the word list was built from these texts. Unglossed ETP
   types number about 400, most of them seen once.
6. **Latin is not a constraint.** The epitaph pool is 350 times the ETP token count, so any
   Etruscan size can be matched with many independent draws.

Consequence for phase 1: formula context has to be measured at about 2,400 clean tokens (ETP),
or at about 9,800 noisy tokens with CIEP after re-segmentation and removal of non-Etruscan
texts. The Latin control is cut to whichever size is chosen. That choice goes into the phase 1
protocol, frozen before any run.

## Expected outcome

Two outcomes are likely and both are informative:

1. **Classes are recoverable.** Funerary formulae are rigid, so kinship, age and numeral words
   should separate from names on Latin and Etruscan controls. Then phase 6 gives ranked class
   hypotheses for unglossed words, with a measured error rate.
2. **Only names separate.** If held-out accuracy comes only from the praenomen/family-name split,
   the method adds nothing a reader of the formulae does not already know. That is reported
   as a negative.

Neither outcome translates Etruscan. The best case is a calibrated guess of what kind of word
an unglossed word is.
