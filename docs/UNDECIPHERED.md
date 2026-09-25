# Other undeciphered scripts: which ones these methods could test

**Etruscan update, 2026-09-24:** the track is closed. The
[closure record](ETRUSCAN.md) consolidates all experiments, source qualifications
and conditions for reopening. The candidate ranking below is the historical scope,
not an instruction to start another track.

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

Etruscan (branch `etruscan`) is scoped as a meaning-class task in
[experiments/etruscan/SCOPE.md](../experiments/etruscan/SCOPE.md). Round one
([REPORT.md](../experiments/etruscan/REPORT.md)): formula context predicts word class at 44%
balanced accuracy (chance 20%) on 2,433 Etruscan tokens and on size-matched Latin, but
rare-class calls are mostly wrong. Round two ([REPORT.md](../experiments/etruscan-fresh/REPORT.md))
fails the fresh test on words its ETP loader did not label (28%, nominal baseline 25%).
Round three ([repair-and-abstain report](../experiments/etruscan-repair/REPORT.md)) found that
the loader had discarded usable dictionary rows. After repair and a stricter grouped test,
endings plus abstention reaches 85.4% accepted precision at 45.0% coverage, but almost all
accepted calls are names; both methods fail the frozen continuation gate. Stopped without
unknown-word predictions. The result limits these tested pipelines, not all Etruscan research.

A subsequent [names-as-scaffolding pilot](../experiments/etruscan-scaffolding/REPORT.md)
uses 30 audited translated inscriptions and hides eight translations across two word
families. Joint relationship inference abstains on all eight; local alignment recovers
one target relationship. The frozen gate fails. The report separates missing candidate
participant pairings from ambiguous action meanings; it does not close all relational approaches.

The [participant-complete graph follow-up](../experiments/etruscan-graphs/REPORT.md)
retains alternative name mappings and requires every supplied person to appear.
Using all 30 prior cases as training, it recovers 10/16 complete interpretations
on different inscriptions: 11/11 accepted edges correct, but only 11/23 reference
edges recovered. It beats the baselines and 99 translation shuffles (p=0.01), yet
fails the frozen 60% recall requirement. Candidate loss is repaired; daughter and
mixed cases remain misranked. This is same-source recovery of known relationships,
not discovery of new meanings.

The final bounded [clause-coverage pilot](../experiments/etruscan-clauses/REPORT.md)
uses all 46 prior cases for training and tests 20 different inscriptions. It
recovers 4/20 exact graphs, with 4/8 accepted edges correct and only 4/36 reference
edges recovered; mixed interpretations score 0/8. The name-count baseline does
better. Learned span associations confuse age/death context with parentage and
do not reliably bind participants. The frozen gate fails. Local model tuning is
stopped; independently checked translations and grammatical annotations are the
next data requirement. Earlier positive results remain preserved and qualified.

A [source evidence audit](../experiments/etruscan-evidence/REPORT.md) now records
nine exposed inscriptions with pinpoint citations and grammatical spans. Three
are corroborated, one partly corroborated, three disputed and two unverified.
The disputed cases concern gift roles and name interpretation. Six relationships
have source support, but all nine records still need expert review. This is
development evidence, not a new test or a revision of the frozen scores.
