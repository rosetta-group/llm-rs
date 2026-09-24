# Other undeciphered scripts: which ones these methods could test

Written 2026-09-23, after the [Linear A track](LINEAR_A.md) closed. The Voynich and Linear A work
gives two rules for choosing a target:

1. **A known-answer control must be possible.** Every method is first run on text where the
   answer is known, at the target's size. Linear A failed here: nothing like it is known.
2. **The corpus must be large enough.** Naibbe recovery needed 2,600 to 5,200 letters of running
   text with the language given. Linear A's 700 readable words were too few for any language test.

| Script | What is known | Corpus | Fit | Why |
|---|---|---|---|---|
| **Rongorongo** (Rapa Nui, Easter Island) | Language almost certainly Old Rapa Nui (Polynesian); sign values unknown | about 26 objects, about 15,000 glyphs | **Best** | Same setup as Naibbe: language known, key unknown. Size is above the recovery threshold. A control can be built by writing real Rapa Nui text in an invented script of the same size. Risk: it may not record full language. |
| **Proto-Elamite** (Iran, c. 3100–2900 BCE) | Number systems and accounting structure partly read; language probably Elamite, but not certain | about 1,600 tablets, openly available through CDLI | Good for structure | Sign functions can be inferred from accounting logic, as `ku-ro` "total" was for Linear A. A translation is unlikely: Elamite itself is poorly known. |
| **Indus script** (c. 2600–1900 BCE) | Neither language nor sign values | about 4,000 inscriptions, about 5 signs each | Poor | Texts are too short for any recovery method; it is disputed whether the signs encode language. |
| **Cypro-Minoan** and **Cretan hieroglyphs** | Related to Linear A; some values guessed from it | a few hundred short texts each | Poor | Smaller than Linear A, which was already too small. |
| **Etruscan** and **Meroitic** | Sounds known (scripts read); language only partly understood | thousands of mostly short texts | Possible, different task | The problem is meaning, not decipherment. Distributional methods on formulae could help; there is no key to recover. |

Rongorongo was tried next ([scope](../experiments/rongorongo/SCOPE.md), [report](../experiments/rongorongo/REPORT.md)).
A Māori text disguised as signs at the tablets' size (2,600 syllables) was recovered at 65% on
average (44% worst), against a 90% gate; an earlier study (Rochala 2026) found the same limit with
a different solver. The tablets are too short for statistical sign-to-syllable recovery.

Proto-Elamite ([scope](../experiments/proto-elamite/SCOPE.md)): round one recovered the counting
ratio N14 = 10 N01 blind on proto-cuneiform but missed the grain ratio
([report](../experiments/proto-elamite/REPORT.md)). Round two searched three systems jointly and
passed its proto-cuneiform gate. On Proto-Elamite it recovered N14 = 10 N01 (11 tablets) and
N14 = 6 N01 (9 tablets) from the tablets' own sums
([report](../experiments/proto-elamite-joint/REPORT.md)). These are the textbook ratios, confirmed
without assuming them; no sign other than a numeral is read. Round three tried to link object
signs to the two systems but could label only 15 control tablets and failed its gate
([report](../experiments/proto-elamite-signs/REPORT.md)). Round four labelled tablets with held-out
fits and passed its barley control (p = 0.018). On Proto-Elamite, 10 of 11 tablets that need the grain
ratio carry the proposed grain signs M288, M036 or M297, against 1 of 8 counting tablets (p = 0.0012).
Animal and people signs did not separate ([report](../experiments/proto-elamite-signs-loo/REPORT.md)).
An exploratory round five found no fixed exchange ratio between entries ([report](../experiments/proto-elamite-ratios/REPORT.md)).
