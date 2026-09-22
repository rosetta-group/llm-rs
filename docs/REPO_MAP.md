# Repository map

Where things live and which files are frozen. Paths are relative to the repository root.

## Top level

| Path | What it is |
|---|---|
| `README.md` | Entry point: what this is, headline results, where to go next |
| `docs/` | Human-facing documentation (this folder) |
| `RESEARCH_LOG.md` | Chronological research record; long, append-only, every experiment and its limits |
| `RESEARCH_PLAN.md` | Research gates and status per work item |
| `voynich/` | Library code: data preparation, models, decoders, evaluation |
| `experiments/` | One driver script and one folder per experiment; protocols, results, reports |
| `tests/` | Unit tests (`python -m unittest discover -s tests`) |
| `data/folios/` | Archived Voynich scans and pixel descriptions (tracked, ~120 MB) |
| `artifacts/` | Large or private working files: raw corpora, fitted priors, sealed answers. Git-ignored, hashed in freezes |
| `training_run_outputs/` | Model weights and curves from the closed prediction track. Git-ignored |

## Library (`voynich/`)

| Module | Purpose | Frozen by |
|---|---|---|
| `data.py`, `windows.py`, `tokenization.py` | IVTFF parsing, page groups, token windows for the prediction track | — |
| `baselines.py`, `evaluate.py`, `character.py`, `context.py`, `matched.py` | Prediction baselines, GRU/transformer training, context and matched-context experiments | — |
| `corpora.py`, `sources.py` | Pinned public corpora and downloads with hash checks | rounds 1–3 (corpora) |
| `decipher.py` | Alphabet, normalisation, edit distance, the original annealing solver | all recovery rounds |
| `unknown_cipher.py` | Family-free solver of the first codebook-free benchmark | standard-method |
| `description_length.py` | Character prior and total description-length candidate scoring | standard-method, rounds 1–4 |
| `homophonic.py` | Nuhn-style key beam and Berg-Kirkpatrick & Klein HMM EM | standard-method, rounds 1–4 |
| `segmentation.py` | Lexicon word segmenter (Viterbi with word transitions) | segmentation, rounds 1–4 |
| `variable_units.py` | Variable-length key beam, iterated local search, annealing, piece induction | rounds 1–4 |
| `joint_segments.py` | Joint segmentation-and-decipherment EM (latent parses, role emissions) | rounds 1–4 |
| `joint_segments_v2.py` | Same EM with a candidate-lexicon override and usage pruning | rounds 2–4 |
| `lexical_polish.py` | Word-level polish of a unit key using the segmenter's local cost | rounds 3–4 |
| `lexicon_repair.py` | Concatenation test and complement admission for the candidate lexicon, iterated with EM | round 4 |
| `association_complex.py`, `image_domains.py`, `folio_description.py` | Image-association studies and pixel descriptions | — |

"Frozen by" means a committed `freeze.json` records the file's hash; changing the file
breaks `verify` for that round. Add new behaviour in a new module instead.

## Experiments (`experiments/`)

Each recovery round has a driver, a development folder and a fresh-evaluation folder.

| Round | Driver(s) | Development record | Fresh evaluation |
|---|---|---|---|
| Assisted decipherment | `decipherment.py`, `boundaries.py` | — | `decipherment/` |
| Segmentation | `segmentation.py` | `segmentation/` | same folder |
| Codebook-free v0 | `codebook_free.py` | — | `codebook-free/` |
| Standard methods | `standard_decipherment.py` (+ `_audit`, `_rebuild`, `_records`, `_report`), `historical_sources.py` | `standard-decipherment/development.json` | `standard-decipherment/` |
| Joint EM round one | `joint_development.py`, `joint_recovery.py` | `joint-development/` | `joint-recovery/` |
| Round two | `joint_development_v2.py`, `joint_recovery_v2.py`, `verse_sources.py` | `joint-development-v2/`, `verse-prior/` | `joint-recovery-v2/` |
| Round three | `joint_development_v3.py`, `joint_recovery_v3.py` | `joint-development-v3/` | `joint-recovery-v3/` |
| Round four | `joint_development_v4.py`, `joint_recovery_v4.py` | `joint-development-v4/` | `joint-recovery-v4/` |

Inside a fresh-evaluation folder: `freeze.json` (hashes, settings, commit), `results.json`
(per-case metrics), `REPORT.md` (what it means), `evaluated-records.tar.gz` (the sealed
challenge, predictions and answers, released after grading).

Other folders:

| Folder / file | Content |
|---|---|
| `report/`, `report-completed/`, `report-matched/` | Illustrated reports of the prediction track |
| `CHARACTERS.md`, `CONTEXT.md`, `LEARNING_CURVES.md`, `MATCHED.md`, `REPLICATIONS.md`, `RESULTS.md`, `CLOUD_RESULTS.md` | Prediction-track results tables |
| `*_PLAN.md`, `*-plan.json` | Fixed settings written before each prediction experiment |
| `language-comparison/`, `languages.py` | Corpus statistics across eleven languages |
| `association/`, `association-complex/`, `image-domains/` | Image studies with protocols and results |
| `method-benchmark/` | Combined benchmark report and methods note |
| `splits/` | Fixed Voynich page groups (folio and quire splits) |
| `sources.json`, `language-sources.json`, `decipherment-sources.json`, `segmentation-sources.json` | Pinned downloads with hashes and licences |
| `cloud-cleanup.json` | Audit of the deleted cloud resources |

## Tests (`tests/`)

One file per component. `test_joint.py` checks the joint EM against brute-force
enumeration; `test_polish.py` checks the lexical polish; `test_standard_decipherment.py`
checks the beam against exhaustive search and the HMM against exhaustive paths.

## What is not in git

`artifacts/` holds raw corpora (restorable from the pinned tarballs), fitted priors
(refittable from pinned texts, hashed in freezes), sealed answers and predictions (released
in `evaluated-records.tar.gz` after grading), and prediction-track model files.
