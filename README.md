# llm-rs: Voynich meaning-recovery research

Controlled experiments on whether computational methods can recover meaning from the
Voynich manuscript. The repository contains no translation. It contains a prediction track
(closed), corpus statistics, three image studies (parked), an active cipher-recovery
track in which a solver is tested on Voynich-like ciphertext whose answers stay sealed
until grading, and a Linear A track (closed) that applied the same methods to a second
undeciphered script.

**Start with [docs/OVERVIEW.md](docs/OVERVIEW.md)** for the story in plain English, then
[docs/RESULTS.md](docs/RESULTS.md) for every number with its record.
The full Voynich account (methods, sealed results, corrections, mechanism checks, open items) is
[docs/VOYNICH.md](docs/VOYNICH.md).

## Where things stand

| Question | Answer so far | Record |
|---|---|---|
| Is Voynich text predictable? | Yes, but shuffled and synthetic controls are predictable to the same degree; prediction cannot detect meaning | [prediction report](experiments/report-completed/REPORT.md) |
| Is it a simple cipher of a European language? | Not with its word spaces kept: adjacent word lengths cluster in Voynich and anti-cluster in Romance languages | [corpus statistics](experiments/language-comparison/REPORT.md) |
| Do pictures explain the text? | No association found beyond scribe hand and layout; some tests are unidentifiable | [image studies](experiments/image-domains/REPORT.md) |
| Can a solver break a Voynich-style cipher without its codebook? | Partly: 1.83% character error and 26.8% word error on four sealed historical passages of about 20,800 letters; pass mark not met | [round six](experiments/joint-recovery-v6/REPORT.md) |
| Has any Voynich word been read? | No. The manuscript's reserved test pages have never been scored | [research log](RESEARCH_LOG.md) |
| Can these methods tell which language Linear A is? | No. Lexical tests cannot find Greek even in Linear B at Linear A's size (about 700 words); the one profile match (Hittite) also appears for shuffled syllables | [Linear A summary](docs/LINEAR_A.md) |

Earlier short-passage recovery results (Naibbe cipher, codebook-free; round six below):

| Text | Round four | Round three | Round two | Round one |
|---|---:|---:|---:|---:|
| Dante, character error | **5.7%** | 10.5% | 10.3% | 33.5% |
| Dante, word error | **45%** | 56% | 60% | 87% |
| Modern Italian, character error | not run (test text exhausted) | 8.8% | 9.5% | 12.5% |

Pass mark: 1% character error and 10% word error. Round four repaired the solver's list of
candidate cipher pieces from the ciphertext alone. The new
[segmentation audit](experiments/segmentation-audit/REPORT.md) separates cipher parsing
from word boundaries: perfect letters still give 15.3% word error on historical prose
and 28.5% on Petrarca development verse. A fixed-key reparse improved mean character
error by only 0.29 points and was rejected before consuming fresh passages.

A [training-only verse word model](experiments/word-segmentation-v2/REPORT.md) then
cut Petrarca development word error to **14.0%**. On fresh Villani passages it only
improved **28.4% to 27.3%**; modern VIT improved **8.5% to 8.4%**. The declared
3-point historical transfer threshold failed, so the old decoder stays the baseline.
Villani's chapter rubrics survived extraction (2.24% of reference words); this
deviation is archived, and the test was not silently replaced or selectively regraded.

The [v3 segmenter](experiments/word-segmentation-v3-fresh/REPORT.md) replaces the flat
unknown-word penalty with a letter-level spelling model. Most wrong spaces fell inside
words missing from the lexicon, so this targets them. On fresh passages from a new
author, Dino Compagni, perfect-letter word error fell **24.0% to 17.5%**. Modern ParTUT
fell **7.2% to 6.0%**. This passes the declared 3-point transfer threshold, the first
segmenter to do so. The 10% per-passage word gate is still not met.

[Round five](experiments/joint-recovery-v5/REPORT.md) put the v3 segmenter into full
ciphertext-only Naibbe recovery on eight new sealed passages (Dante and Compagni).
Letter error is unchanged at **5.6%**. Word error falls **45.8% to 41.5%** from segmentation
alone, and all 8 cases improve. Using v3 in the polish stage too does not help letters.
Remaining word error is mostly caused by letter errors.

[Round six](experiments/joint-recovery-v6/REPORT.md) tested longer ciphertext (about 20,900
letters) with a context reparse that refits the key. On four sealed historical cases (Dante and
Compagni), letter error fell to **1.83%** and word error to **26.8%**, from 3.85% / 34.1% for the
same decoder without the reparse. Its two modern cases turned out to be ISDT training sentences
and are excluded ([audit](experiments/partut-overlap-audit.json)). The 1% / 10% gate is not yet met.

A [language-identification control](experiments/language-id/REPORT.md) checks that the
pipeline does not simply assume Italian. It encrypted Latin, Old French, German, English and
Italian with Naibbe and decoded each under all five priors. The true language fit best
**5 of 5** times, by 0.75–1.33 bits per letter. This does not show that the Voynich text is
Naibbe-class or name its language.

A [rejection and fixed-key transfer screen](experiments/rejection-transfer-v2/REPORT.md)
now tests the missing "none of these" decision. On its first three fresh passage/key
pairs, English, Italian and Latin each rank first on both passages; the acceptance
rule admits Italian and Latin but rejects English because its fit score exceeds the
fixed ceiling. Transfer character errors are **6.91%, 5.85% and 9.37%**, respectively.
These controls use the five-language short-passage decoder, not round six. The report
records the negative controls and resource stop; the next priority is calibrated
rejection and affordable control searches, with the manuscript gate unchanged.

The [three development follow-ups](experiments/rejection-followups/REPORT.md) are
complete. A transfer-centered rule accepts **3/3** released genuine ciphers instead
of 2/3. Three new copying controls preserve every token frequency, retain **98.47–98.64%**
transfer coverage, and are all rejected without caps. On the released copying fixture,
incremental swap scoring is **31.1× faster** with the same winning swap and score;
peak worker memory falls from **1,694 to 331 MiB**. This is a search-step benchmark,
not a full-decoder speedup. All results are development on three source/key blocks;
fresh confirmation and manuscript readiness remain unestablished.

The new [24-panel image pilot](data/folios/object-pilot/REPORT.md) adds object groups,
evidence boxes and compound relations. Independent human review is pending.

The [Linear A track](docs/LINEAR_A.md) (branch `linear-a`, 2026-09-23) reverses the Voynich
problem: sign sounds are roughly known, the language is not. Five rounds tested lexicon
matching, tablet position, name lists, seven targeted probes and grammar profiles. Every
method was checked first on Linear B, which is Greek. The three lexical rounds found Greek in at
most 10% of Linear B samples against a 90% gate. No round-four probe reached p < 0.007. Round
five's Hittite profile match also appears for shuffled syllables, so it is an artefact. The track is closed. [docs/UNDECIPHERED.md](docs/UNDECIPHERED.md)
ranks other undeciphered scripts; the next candidate is Rongorongo.

## Documentation

| Page | For |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | What was tried, what was found, what it means |
| [docs/RESULTS.md](docs/RESULTS.md) | Every result in one table, with links |
| [docs/FOUR_SCRIPTS.md](docs/FOUR_SCRIPTS.md) | Cross-script write-up: Voynich, Linear A, Rongorongo, Proto-Elamite, and the known-answer rule |
| [docs/LINEAR_A.md](docs/LINEAR_A.md) | The Linear A track: five rounds, why none identified a language |
| [docs/UNDECIPHERED.md](docs/UNDECIPHERED.md) | Other undeciphered scripts and which ones these methods could test |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | Which file does what; which files are frozen |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Terms: BPC, CER, Naibbe, piece, role, gate, freeze, Linear B, entry word |
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md) | How a round is frozen, sealed, graded and reported |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Setup and every command |
| [docs/COMMIT_MAP.md](docs/COMMIT_MAP.md) | Old to new commit hashes after the one history rewrite |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | The full chronological record, append-only |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Research gates and status per work item |

## Setup

```sh
poetry install
.venv/bin/python -m unittest discover -s tests -v
```

Recovery rounds run on CPU in minutes. Raw corpora, fitted priors and sealed answers live in
the git-ignored `artifacts/` folder and are restored from pinned archives; see
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Rules of the road

- Methods are frozen and committed before sealed passages exist; `verify` enforces it.
- Each sealed passage is used once, then disclosed and excluded.
- Development text, evaluation text and Voynich text never mix; the Voynich final test is sealed.
- Files hashed by a freeze are never edited; new behaviour goes in a new module.
- Negatives are recorded with the same care as positives. No translation claims.

## Attribution

Naibbe cipher code and data: Michael A. Greshko (2025), *The Naibbe cipher: a substitution
cipher that encrypts Latin and Italian as Voynich Manuscript-like ciphertext*, Cryptologia,
modified MIT licence. Italian texts: Universal Dependencies treebanks (VIT is CC BY-NC-SA
3.0) and Wikisource transcriptions (CC BY-SA); manifests with revisions, hashes and licences accompany each
experiment. Manuscript scans: Yale Beinecke Library; see `data/folios/sources/`. Linear A and
Linear B data (SigLA via Navarre-AI, DĀMOS, TLHdig, LAMAN, Oracc): see
[docs/LINEAR_A.md](docs/LINEAR_A.md#data-and-licences).
