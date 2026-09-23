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

Started next: Rongorongo, scoped in [experiments/rongorongo/SCOPE.md](../experiments/rongorongo/SCOPE.md).
Etruscan (branch `etruscan`) is scoped as a meaning-class task in
[experiments/etruscan/SCOPE.md](../experiments/etruscan/SCOPE.md). Round one
([REPORT.md](../experiments/etruscan/REPORT.md)): formula context predicts word class at 44%
balanced accuracy (chance 20%) on 2,433 Etruscan tokens and on size-matched Latin, but
rare-class calls are mostly wrong. Round two ([REPORT.md](../experiments/etruscan-fresh/REPORT.md))
fails the fresh test on words ETP does not gloss (28%, chance 25%): their neighbours are
unglossed too. Stopped.
