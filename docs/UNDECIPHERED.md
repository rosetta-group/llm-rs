# Other undeciphered scripts: which ones these methods could test

Historical scoping written 2026-09-23; methodological claims corrected 2026-09-24 after the
[Linear A closeout](LINEAR_A_CLOSEOUT.md). This is not a current status report for concurrent
branches or a queue of new work. Corpus descriptions below retain the original scoping estimates.

1. **Known-answer control:** validate a particular method at the target's size and with relevant
   ambiguity. Linear B supplies one for Linear A; the tested methods failed it. That does not
   make a control impossible.
2. **Method-specific data requirement:** Naibbe recovery depended on passage length in those
   experiments. Its 2,600–5,200-letter range cannot be transferred as a universal threshold to
   another script. Linear A's 696 readable word types are not 696 running words.

| Script | What is known | Corpus | Fit | Why |
|---|---|---|---|---|
| **Rongorongo** (Rapa Nui, Easter Island) | Language almost certainly Old Rapa Nui (Polynesian); sign values unknown | about 26 objects, about 15,000 glyphs | **Initial candidate** | An assumed Rapa Nui prior allows a simulated control, but neither the language assumption nor transfer from Naibbe is established by corpus size. Risk: it may not record full language. |
| **Proto-Elamite** (Iran, c. 3100–2900 BCE) | Number systems and accounting structure partly read; language probably Elamite, but not certain | about 1,600 tablets, openly available through CDLI | Good for structure | Sign functions can be inferred from accounting logic, as `ku-ro` "total" was for Linear A. A translation is unlikely: Elamite itself is poorly known. |
| **Indus script** (c. 2600–1900 BCE) | Neither language nor sign values | about 4,000 inscriptions, about 5 signs each | Poor | Short inscriptions limit the tested running-text methods; it is disputed whether the signs encode language. |
| **Cypro-Minoan** and **Cretan hieroglyphs** | Related to Linear A; some values guessed from it | a few hundred short texts each | Poor | Sparse short texts make matched controls difficult; the Linear A failures establish no universal size limit. |
| **Etruscan** and **Meroitic** | Sounds known (scripts read); language only partly understood | thousands of mostly short texts | Possible, different task | The problem is meaning, not decipherment. Distributional methods on formulae could help; there is no key to recover. |

Historical follow-ups (2026-09-23 snapshot): the earlier note named a Rongorongo scope at
`experiments/rongorongo/SCOPE.md`; that file is absent from this checkout.
Etruscan (branch `etruscan`) is scoped as a meaning-class task in
[experiments/etruscan/SCOPE.md](../experiments/etruscan/SCOPE.md). Round one
([REPORT.md](../experiments/etruscan/REPORT.md)): formula context predicts word class at 44%
balanced accuracy (chance 20%) on 2,433 Etruscan tokens and on size-matched Latin, but
rare-class calls are mostly wrong. Round two ([REPORT.md](../experiments/etruscan-fresh/REPORT.md))
fails the fresh test on words ETP does not gloss (28%, chance 25%): their neighbours are
unglossed too. That round was stopped; this historical note does not describe later Etruscan work.
