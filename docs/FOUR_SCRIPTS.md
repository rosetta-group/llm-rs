# Four undeciphered scripts, one rule

Written 2026-09-24. Covers work from 21 to 24 September 2026. A formatted version is published as
a private page, *Four-Script Decipherment Trials*.

We tested whether computational methods can decipher the Voynich manuscript, Linear A,
Rongorongo and Proto-Elamite. Each method first had to recover a known answer from a corpus of
the target's size. Most could not. Only arithmetic on Proto-Elamite accounts passed that test.

No script was translated. Every result below was frozen in git before it was scored, and every
sealed text was used once.

> **The rule.** A decipherment claim is only as good as the method's result on a known answer,
> at the same size, with the same losses.

The known answers:

| Script | Known answer |
|---|---|
| Voynich | Naibbe, a published cipher that turns Italian into Voynich-like text |
| Linear A | Linear B, a related script that is Greek |
| Rongorongo | Māori text disguised as signs |
| Proto-Elamite | proto-cuneiform, whose number systems are understood |

## Verdict by script

| Script | Known-answer control | Best result | Status |
|---|---|---|---|
| Voynich | Naibbe cipher of Italian | sealed historical text recovered at 1.83% character error, 26.8% word error | control improving |
| Linear A | Linear B → Greek | Greek found in at most 10% of samples; Linear A never scored | failed, 5 rounds |
| Rongorongo | Māori disguised as signs | 65% of syllables recovered at the tablets' size; 90% needed | failed |
| Proto-Elamite | proto-cuneiform accounts | both N14 ratios recovered blind; grain signs follow the grain ratio | passed, 2 of 5 rounds |

## Voynich manuscript (main track, active)

Prediction models find real structure, but they find the same structure in shuffled text, so
they cannot detect meaning. The live question is cipher recovery: can a solver recover the
plaintext of a Voynich-like cipher from ciphertext alone, given the language?

| Round | Setup | Character error | Word error |
|---|---|---:|---:|
| One | joint splitting and mapping, 5,200 letters | 12.5% modern / 33.5% historical | 58% / 87% |
| Four | lexicon repair, Dante | 5.7% | 45% |
| Six (sealed) | about 20,900 letters, 4 historical passages | 1.83% | 26.8% |

A separate control ciphered five languages; the solver ranked the true one first in 5 of 5. The
pass mark (1% character error, 10% word error) is not met yet, and Voynich's own reserved test
pages have never been scored. Details: [OVERVIEW.md](OVERVIEW.md), [RESULTS.md](RESULTS.md).

## Linear A (closed, 5 rounds)

Most Linear A sign sounds are roughly known from Linear B; the language is not. So the task
became a language test: do Linear A words look like some known language's words more than chance
allows? There are only 696 readable word types.

| Round | Method | Linear B control |
|---|---|---|
| One | whole-language lexicons, Linear B spelling | Greek found in 10% of samples |
| Two | plus: is a matched name where the tablet puts names? | 0% |
| Three | entry words against proper-name lists | 0% |
| Four | seven targeted probes (Egyptian and Keftiu names, gods, trade words, profiles, spelling rules, roles) | no probe below p = 0.007 |
| Five | grammar profiles against Hittite, Hurrian, Hattic, Luwian | passed, but shuffled syllables match just as well |

- **Spelling erases words.** Linear B drops final consonants and merges *r* and *l*, so 37% of
  random Linear-A-shaped words match some Greek word exactly.
- **The sign values hold up.** 2 of 14 Linear B Cretan place names occur in Linear A (`pa-i-to`
  Phaistos, `se-to-i-ja`) against 0.025 by chance.
- **Names carried over.** Greek scribes turned Minoan names in `-re`/`-ru` into `-ro`: 12 pairs
  against 3.1 by chance, supported but not confirmed.
- **The Hittite match was the script.** Cuneiform has no *o*, and Linear A rarely writes it;
  merging *o* into *u* removes the match.

Details: [LINEAR_A.md](LINEAR_A.md).

## Rongorongo (closed, 1 round)

The language is almost certainly Old Rapa Nui, so this is the cipher setup: language known, key
unknown. The control disguised held-out Māori narratives and songs as signs. A hidden Markov
model then recovered the key using a Māori syllable model of 2.4 million syllables.

| Syllables | Recovered, mean | Worst passage |
|---:|---:|---:|
| 1,300 | 61% | 38% |
| **2,600** (tablets' usable size) | **65%** | **44%** |
| 5,200 | 77% | 67% |
| 8,000 | 84% | 71% |

The gate was 90% in all four passages at 2,600 syllables. An earlier independent study (Rochala
2026) hit the same limit with a different solver. The tablets are too short, and this was the easy
case: it assumed one sign per syllable and a clean transcription.

None of the tablets is on Rapa Nui, and the community seeks their return; this is a method test,
not a reading. Details: [report](../experiments/rongorongo/REPORT.md).

## Proto-Elamite (closed, 5 rounds, two validated results)

About 1,600 account tablets from Iran, c. 3100–2900 BCE. Their totals let arithmetic test
hypotheses without knowing the language. Every published analysis assumes the textbook ratios
between numeral signs. Here they were treated as unknown and recovered from the tablets' own
sums, on proto-cuneiform first.

| Round | Question | Proto-cuneiform control | Proto-Elamite |
|---|---|---|---|
| One | ratio from two-sign tablets | N14 = 10 found (p = 0.001); 6 missed | not run |
| Two | three systems searched jointly | sexagesimal and capacity values exact | `1(N14) = 10(N01)` on 11 tablets, `= 6(N01)` on 9 |
| Three | object signs, strict label | only 15 tablets labelled; failed | not run |
| Four | object signs, held-out labels | barley on capacity tablets, p = 0.018 | grain signs on 10 of 11 capacity vs 1 of 8 counting tablets, p = 0.0012 |
| Five | fixed exchange ratios (exploratory) | no control available | none found |

- **Number systems, recovered blind.** The counting ratio (N14 = 10 N01) and the grain-capacity
  ratio (N14 = 6 N01) come out of the arithmetic without being assumed. This confirms the
  standard reading; it adds no new value.
- **Grain signs follow the grain ratio.** Tablets that balance only with the capacity ratio almost
  all carry `M288`, `M036` or `M297`. That supports their reading as grain containers or products.
  The support is only partly independent: M288 was already classed from its numerals.
- **People and animals do not separate.** Workers and animals receive grain rations, so their signs
  appear on both kinds of account.

Details: round [one](../experiments/proto-elamite/REPORT.md), [two](../experiments/proto-elamite-joint/REPORT.md),
[three](../experiments/proto-elamite-signs/REPORT.md), [four](../experiments/proto-elamite-signs-loo/REPORT.md),
[five](../experiments/proto-elamite-ratios/REPORT.md).

## What we would do the same way again

1. **Control at the target's size.** Naibbe needed 2,600–5,200 letters. Linear A has 700 words
   and Rongorongo 2,600 usable sign pairs. Size alone decided two tracks.
2. **A shuffled-sign negative control in every profile test.** It was missing in Linear A round
   five, and it is what exposed the Hittite match.
3. **Enough null runs to reach the threshold.** 100 runs cannot give p below 0.0099; one Linear A
   test missed a 0.007 threshold for that reason alone.
4. **Check spellings in the data before freezing.** The ATF writes barley as `SZE`, not `ŠE`; one
   protocol looked for a spelling that does not occur.
5. **Nulls must keep the structure they are not testing.** Re-pairing amounts across tablets
   cannot model a scribe repeating an amount, and it made a trivial ratio of 1 look significant.

## Next

Send the Proto-Elamite results to a specialist (J. Dahl, K. Kelley or L. Born) for an independent
check. So far all blinding has been procedural, on one machine. The remaining candidates (Indus,
Cypro-Minoan, Cretan hieroglyphs) are all smaller than Linear A, so no new script is planned
([UNDECIPHERED.md](UNDECIPHERED.md)).

## Data and licences

CDLI (academic re-use with credit); DĀMOS and SigLA (CC BY-NC-SA 4.0); TLHdig (CC BY 4.0); LAMAN
(CC BY-SA 4.0); Oracc (CC0); Wiktionary via kaikki.org (CC BY-SA); CEIPP transliteration
(non-profit use with credit); public-domain Māori and Rapa Nui texts. Full records and
reproduction steps are in the linked reports and [REPRODUCE.md](REPRODUCE.md).
