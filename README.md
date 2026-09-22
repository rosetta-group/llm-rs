# llm-rs: Voynich meaning-recovery research

Controlled experiments on whether computational methods can recover meaning from the
Voynich manuscript. The repository contains no translation. It contains a prediction track
(closed), corpus statistics, three image studies (parked), and an active cipher-recovery
track in which a solver is tested on Voynich-like ciphertext whose answers stay sealed
until grading.

**Start with [docs/OVERVIEW.md](docs/OVERVIEW.md)** for the story in plain English, then
[docs/RESULTS.md](docs/RESULTS.md) for every number with its record.

## Where things stand

| Question | Answer so far | Record |
|---|---|---|
| Is Voynich text predictable? | Yes, but shuffled and synthetic controls are predictable to the same degree; prediction cannot detect meaning | [prediction report](experiments/report-completed/REPORT.md) |
| Is it a simple cipher of a European language? | Not with its word spaces kept: adjacent word lengths cluster in Voynich and anti-cluster in Romance languages | [corpus statistics](experiments/language-comparison/REPORT.md) |
| Do pictures explain the text? | No association found beyond scribe hand and layout; some tests are unidentifiable | [image studies](experiments/image-domains/REPORT.md) |
| Can a solver break a Voynich-style cipher without its codebook? | Partly: about 94 letters in 100 on sealed 5,200-letter Dante passages; words about half wrong; pass mark not met | [round four](experiments/joint-recovery-v4/REPORT.md) |
| Has any Voynich word been read? | No. The manuscript's reserved test pages have never been scored | [research log](RESEARCH_LOG.md) |

Latest sealed recovery results (Naibbe cipher, codebook-free):

| Text | Round four | Round three | Round two | Round one |
|---|---:|---:|---:|---:|
| Dante, character error | **5.7%** | 10.5% | 10.3% | 33.5% |
| Dante, word error | **45%** | 56% | 60% | 87% |
| Modern Italian, character error | not run (test text exhausted) | 8.8% | 9.5% | 12.5% |

Pass mark: 1% character error and 10% word error. Round four repaired the solver's list of
candidate word pieces from the ciphertext alone; given the true list, the same solver reaches
0.5% character error, so the remaining gap is still in that list.

The new [24-panel image pilot](data/folios/object-pilot/REPORT.md) adds object groups,
evidence boxes and compound relations. Independent human review is pending.

## Documentation

| Page | For |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | What was tried, what was found, what it means |
| [docs/RESULTS.md](docs/RESULTS.md) | Every result in one table, with links |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | Which file does what; which files are frozen |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Terms: BPC, CER, Naibbe, piece, role, gate, freeze |
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
modified MIT licence. Italian texts: Universal Dependencies treebanks and Wikisource
transcriptions (CC BY-SA); manifests with revisions, hashes and licences accompany each
experiment. Manuscript scans: Yale Beinecke Library; see `data/folios/sources/`.
