# Overview: what this project is and what it has found

This repository tests whether computational methods can recover meaning from the Voynich
manuscript, a 15th-century book written in an unknown script. It does not contain a
translation. It contains a sequence of controlled experiments, each frozen before it was
graded, that narrow down what kind of method could work and what would count as evidence.
A fifth track applied the same methods to Linear A, a second undeciphered script.

Read this page first. The [results table](RESULTS.md) has every number; the
[research log](../RESEARCH_LOG.md) is the full chronological record.

## The question, and why it is hard

Nobody knows a single Voynich word for certain. There is therefore no answer key. Any method
has to be validated on text where the answer is known and hidden, then applied to Voynich
with independent checks. The project uses two kinds of evidence:

- **Prediction.** How well can a model predict the next character of Voynich text? This
  measures structure, not meaning: a cipher, a real language and a well-made hoax can all be
  predictable.
- **Recovery.** Can a method recover the original text of a cipher that produces
  Voynich-like output, when given only the ciphertext? Here the answer is known and sealed
  until grading, so success and failure are unambiguous.

## Track one: prediction (closed)

Large language models (Qwen 1.7B and 8B with small adapters) and small models trained from
scratch (a 1.3-million-parameter GRU) were trained to predict Voynich characters on fixed
page splits, with shuffled and synthetic control texts.

What came out:

- Voynich text is predictable, and a small GRU (2.33 bits per character) beats the adapted
  large model (2.48). Pretraining on real languages did not help.
- Every gain also appeared on the controls: word-shuffled Voynich, a copy-based generator and
  the Naibbe cipher. A lower prediction score is therefore not evidence of meaning.
- Longer context helped shuffled text as much as intact text across 18 matched runs.

The track was closed on 2026-09-21. It can reopen only for an experiment that names a
mechanism and a contrast that could reject it.

## Track two: corpus statistics (complete)

Voynich word lengths cluster: long words sit next to long words and short next to short,
far more than in eleven Italian, Latin, Greek, French, German, English, Arabic and Finnish
samples. Neither of the two synthetic controls reproduces this. It rules out a plain
substitution of a European language that keeps its word spaces, but it does not decide
between an invented text, a cipher that changes word boundaries, and an unfamiliar language.

## Track three: images (parked)

Three studies asked whether the text on a page predicts what the page's picture shows,
using existing catalogue descriptions. None found an association beyond what scribe hand
and layout already explain, and some comparisons cannot be tested at all because picture
type, hand and quire move together in this manuscript. Association modelling remains parked until independently reviewed, text-masked
annotations exist. Image description is active: the
[24-panel Object–Relation pilot](../data/folios/object-pilot/REPORT.md) records plants,
people, animals, vessels and diagrams, with evidence boxes and relations such as
figures inside connected basins. These are one AI observer's development annotations,
not independently validated labels or recovered meanings.

## Track four: cipher recovery (active)

This is where the project now lives. The test cipher is **Naibbe**, a published cipher that
turns Italian letters into Voynich-like glyph strings and reproduces many manuscript
statistics. Each Italian letter, or pair of letters, becomes one of several glyph strings;
pairs are glued into one "word", and the string used for a letter depends on its position
in that word.

The solver receives the ciphertext, the knowledge that the language is Italian, and a
character-level model of Italian. It does not receive the codebook, the key, or any paired
example. Passages are generated, encrypted with a fresh hidden key, and the originals stay
sealed until predictions are saved.

What came out, in order:

1. **Early solvers failed completely** (300%+ character error: they produced far too many
   letters). Diagnosis showed two causes. The scoring rule was wrong in one version and the
   search was wrong in another: the correct key always scored best, but no search reached it.
2. **Passage length was a hidden limit.** With 300 to 400 glyph strings to assign and only
   1,300 letters of text, many wrong keys look as Italian as the right one. Recovery becomes
   possible around 2,600 letters and reliable around 5,200. Earlier benchmarks had used
   1,200 to 1,800 letters, so their negatives were partly guaranteed by the setup.
3. **Segmentation and mapping have to be solved together.** Guessing how each cipher word
   splits into glyph strings, then mapping strings to letters, collapses if more than about
   5% of the splits are wrong. A model that treats the split as a hidden variable, learned
   jointly with the letter mapping (an extension of the Berg-Kirkpatrick and Klein HMM),
   recovered about 88% of letters.
4. **Two fixes, each validated on sealed passages.** Pruning glyph strings the model barely
   uses raised split accuracy to about 90%. Adding Petrarch's verse to the Italian model
   fixed Dante, which the prose-only model could not read at all.

A later [development audit](../experiments/segmentation-audit/REPORT.md) separates
cipher-piece errors from word-space errors. Even perfect letters give 15.3% word error
on historical prose and 28.5% on Petrarca verse. A fixed-key context reparse produced
only a 0.29-point mean character-error improvement, below its declared requirement;
it was rejected without using fresh test passages. These are development diagnostics,
not additional sealed evaluations.

The next [word-only experiment](../experiments/word-segmentation-v2/REPORT.md) added
Petrarca training words to the segmenter. Held-out Petrarca word error fell from
28.5% to 14.0%, but fresh Villani error only fell from 28.4% to 27.3%. That misses
the predeclared transfer threshold; the candidate was not promoted. Modern VIT
error fell from 8.5% to 8.4%. The Villani source audit found retained chapter
rubrics (2.24% of words), so the historical result also carries an extraction caveat.
The test exposes a word-boundary problem even when every letter is already correct.

A diagnosis then showed that most wrong spaces sit inside words missing from the
segmenter's lexicon. The flat unknown-word penalty always loses to a split into known
pieces. The [v3 segmenter](../experiments/word-segmentation-v3-fresh/REPORT.md) scores
unknown words with a letter-level spelling model instead. On a new author, Dino Compagni,
word error fell from 24.0% to 17.5%, and modern ParTUT fell from 7.2% to 6.0%. This passes
the declared transfer threshold, but no historical passage reaches 10% word error.

Round four, on four sealed 5,200-letter Dante passages:

| Text | Character error | Word error | Round three |
|---|---:|---:|---:|
| Dante | 5.7% | 45% | 10.5% / 56% |

The declared pass mark is 1% character error and 10% word error. It has not been met. Round
three added a dictionary-based polish of the letters and confirmed that the remaining error sat
in wrongly split words. Round four asked why the splits were wrong and found the answer with an
oracle: given the true list of word pieces, the same solver splits 97% of words correctly and
reaches 0.5% character error. The solver was starved, not broken. Two defects of the piece
list were repaired from the ciphertext alone: frequent two-letter strings that had been admitted
as one-letter pieces, and rare-letter pieces that never passed the frequency threshold. Dante
error halved. Modern Italian was not re-run: the test corpus is used up after three rounds.

[Round five](../experiments/joint-recovery-v5/REPORT.md) is the current standing. It put the
v3 segmenter into full recovery on eight new sealed passages (four Dante, four Compagni).
Character error stayed at 5.63%; word error fell from 45.8% to 41.5%, and all 8 cases improved.
Remaining word error comes mostly from letter errors. A
[language-ID control](../experiments/language-id/REPORT.md) encrypted five languages with
Naibbe; the true language's prior fit best 5 of 5 times.

## Track five: Linear A (audited; tested methods retired)

Linear A is a Bronze Age Cretan script. Most sign sounds are roughly known from Linear B; the
language is not. So the question became: do Linear A words match some known language more than
chance allows? The corpus has 696 readable word types. Every method had to find Greek first in
Linear B (DĀMOS, 5,932 documents), cut to Linear A's size. Linear B is Greek, so the answer there
is known.

| Round | Method | Linear B control | Record |
|---|---|---|---|
| One | whole-language lexicons, Linear B spelling | Greek found in 10% of draws; gate 90% | [report](../experiments/linear-a/REPORT.md) |
| Two | as one, plus name versus tablet position | 0% | [report](../experiments/linear-a-context/REPORT.md) |
| Three | entry words against proper-name lists | 0% | [report](../experiments/linear-a-names/REPORT.md) |
| Four | seven targeted probes (Egyptian, Keftiu, gods, trade, profiles, spelling rules, roles) | no probe below p = 0.007; god and trade-word controls failed | [report](../experiments/linear-a-probes/REPORT.md) |
| Five | grammar profiles against Hittite, Hurrian and other TLHdig languages | passed, but Linear A's Hittite match also holds for shuffled syllables | [report](../experiments/linear-a-tlhdig/REPORT.md) |

The sign values pass a sanity check: 2 of 14 Linear B Cretan place names occur in Linear A,
against 0.025 by chance. The lexical tests fail because Linear B spelling drops final
consonants, so 37% of random Linear-A-shaped words match some Greek lemma. Open name lists also
lack Bronze Age names. These tested methods cannot identify or rule out a language for Linear A.
The [repair audit](../experiments/linear-a-audit/REPORT.md) fixed duplicate-sensitive profiles
and unreachable thresholds. Luwian and Palaic now pass their own profile controls, but shuffled
Linear B still classifies as Greek 20/20. The full-set re/ru → ro pattern has exploratory
p = 0.0001, while the discovery-selected held-out test remains negative at p = 0.1653. A
[sign-only ending test](../experiments/linear-a-structure-v2/REPORT.md) passes only 4/20
known-answer controls; Linear A is not scored. These failures do not prove an information limit.
The [summary](LINEAR_A.md) has the details; [UNDECIPHERED.md](UNDECIPHERED.md) ranks other
scripts, with Rongorongo next.

## What is known, in six sentences

Voynich text has real structure, and prediction alone cannot say what kind. Word-length
clustering rules out the simplest cipher of a European language. Nothing yet links text to
pictures beyond scribe and layout. A codebook-free solver can now recover about 94 letters
in 100 of a Voynich-style cipher when given 5,000 letters of it and the right language. No
Voynich word, passage or image link has been established, and the manuscript's own reserved
test pages have never been scored. For Linear A, lexical tests cannot find Greek even in Linear
B at Linear A's size, and the one profile match also appears for shuffled syllables, so no
language has been identified or ruled out.

## Rules the project follows

- Methods are frozen and committed before any sealed passage is generated.
- Every sealed passage is used once; after grading its source is disclosed and excluded.
- Failures are recorded with the same care as successes.
- Development text, evaluation text and Voynich text never mix.
- No claim about Voynich meaning is made until an interpretation predicts something it was not built on.

See [CONVENTIONS.md](CONVENTIONS.md) for how this is enforced and how to add a round.
