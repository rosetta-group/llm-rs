# Separating prediction, decipherment and meaning

Methods note, 2026-09-21. This is a research draft, not a claim of Voynich translation.
The [standard-method report](../standard-decipherment/REPORT.md) contains fresh results, graphs and an audit.

## Research question

Can a computational method recover independently checkable content from Voynich-like
text, rather than merely predict its characters? We separate prediction, letter recovery,
word recovery and semantic grounding. Success at one stage does not establish the next.

## Benchmark design

```text
Preserve manuscript transcription and page groups
Compare prediction against local-copy and layout controls
Validate decipherment on generated ciphers with known hidden originals
Freeze decoder and selection rules before fresh evaluation
Grade letters and words separately
Require independent anchors before interpreting Voynich text
```

The manuscript's final test pages remain sealed. Historical/modern language statistics
use pinned corpora. Image analyses use grouped pages and nuisance controls. Cipher
challenges expose ciphertext only; an evaluator retains original passages, encryption
seeds and provenance. This is procedural separation on one machine, not an independent
third-party blind trial.

The cipher controls distinguish four assumptions: one-to-one substitution, genuine
one-letter homophonic substitution, one/two-letter homophonic substitution, and the
published Naibbe encoder. The last is also tested in an older codebook-assisted setting.
Those are different tasks; their scores must not be pooled as equivalent recovery tests.

## Evidence already established

| Track | Observation | What it does not establish |
|---|---|---|
| Prediction | Models learn predictable structure; matched context tests do not establish useful extra semantic context | A language identification, word meanings, or a translation |
| Assisted decipherment | Substitution can be recovered; Naibbe letter recovery reaches 99.44% with structural codebook assistance | Recovery without the codebook |
| Earlier word segmentation | Fresh modern WER 6.1%; historical WER 39.9% after modern-prior tuning | Reliable historical passage recovery |
| Earlier family-free search | Correct historical letters were candidates but lost to a wrong expansion | An informative rejection of the entire cipher hypothesis |
| Root-color pilot | +1.45 accuracy points over controls; p=.348 | A repeatable visual meaning |
| Complex images | 12 corrected tests, no established gain | Proof that text and images are unrelated |
| Broad domains | Text alone distinguishes domains, but adds no gain over hand/layout; conditional tests often lack variation | An independently grounded domain vocabulary |

The corresponding records are the [prediction synthesis](../report-matched/REPORT.md),
[assisted control](../decipherment/REPORT.md), [segmentation](../segmentation/REPORT.md),
[family-free control](../codebook-free/REPORT.md), [root-color study](../association/REPORT.md),
[complex associations](../association-complex/REPORT.md), and [domain study](../image-domains/REPORT.md).

## Standard-method extension

The new [protocol](../standard-decipherment/PROTOCOL.md) fixes candidate selection using
an explicit description length. It charges every plaintext character, the mapping and
inventory, and the residual choices needed to reconstruct the cipher. The residual term
prevents many cipher symbols from collapsing to one letter for free. The code length is
an explicit modelling choice, not a certificate that its shortest plaintext is true.

Search comparators implement the key-beam framework of
[Nuhn, Schamper and Ney](https://aclanthology.org/P13-1154/), with their
[improved partial-context/order heuristics](https://aclanthology.org/D14-1184/), and the
trigram HMM of [Berg-Kirkpatrick and Klein](https://aclanthology.org/D13-1087/).
The implementations, Italian priors and CPU limits are documented; published Zodiac
performance and million-restart budgets are not claimed to be reproduced.

Historical training and development use anonymous Novellino prose and Boccaccio's
Decameron. Whole tales are separated. Dante is reserved for fresh evaluation, although
other Dante passages have been studied previously. This distinguishes new passages from
a wholly untouched evaluation author. Editorial orthography and poetry/prose differences
remain possible sources of error. Source pages and transcluded content are archived.


## Fresh standard-method results

The method was committed as `146fa75` before creating 16 encodings from four new passages.
The combined selector recovers exact letters in seven of eight substitution/homophonic
cases, with 0.63% character error in the remaining case. All eight meet the 1% letter
threshold. This is positive evidence that the new search can solve these model classes.

On the historical substitution cases, reranking the same candidates changes character
error from 145.82% to zero. On exact-letter segmentation controls, adding historical prose
changes fresh Dante WER from 37.67% to 24.20%; modern WER changes from 7.38% to 6.02%.
The historical word gate still fails. Variable-length controls and Naibbe remain far from
recovery, and the reserved Voynich experiment remains closed.

The failure decomposition also finds a residual selection problem: an exact historical
homophonic candidate costs 7,185 bits, but a same-length candidate with eight wrong letters
costs 7,173. Length-aware selection removes a specific defect; it does not make a limited
language prior identify truth. On Naibbe, no generated candidate approaches the letter gate.

These are four independent source passages, not 16 independent language samples.
The 13.3-minute CPU decode completed 48 beam runs and eight restarts on each of eight
eligible HMM cases. High-inventory HMM cases were skipped by the frozen resource rule.
All results and exact records were committed after grading; these passages are now disclosed.

## What the negatives can support

A failed control can locate a defect in candidate generation, candidate selection or
word segmentation. Oracle candidate accuracy is a diagnostic that may only be computed
after prediction freeze. It must never select the reported answer.

A one-letter homophonic solver cannot, in general, represent a cipher unit encoding two
letters. Failure on such units is not evidence against every variable-length cipher.
Likewise, a language prior trained on edited Italian prose is not a universal prior for
medieval writing. A false or fluent output is a negative result for the stated method.

Image studies are now parked. More models on the same catalogue will not resolve the
hand/quire confounding or establish that annotations were independent of text. Reopening
requires text-masked annotations, two annotators, agreement statistics and an identifiable
sampling design. The existing nulls and unidentifiable tests are reported separately.

## Reserved mechanism and publication limits

The first permitted Voynich-facing cipher contrast remains intact training text versus
within-line shuffled text under the same frozen multilingual decoder and lossless coding
criterion. It is gated on codebook-free Naibbe recovery. Any language selection and search
budget must be matched for intact and shuffled inputs. A positive contrast would reject
that particular shuffle null; it would not establish a reading. The final test stays sealed.

The contribution is a reproducible benchmark with explicit assumptions, gates and failure
analysis. Broader independent passages, external evaluation, closer reference-solver
replication and independent annotations would strengthen a submission. Publication is not
established by the present pilot. Translation into English or Italian remains the long-term
goal; accepted Voynich meanings are not available as supervised labels here.
