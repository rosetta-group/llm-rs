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
| `linear_a/` | Library code of the Linear A track; kept apart so no Voynich round imports or hashes it |
| `experiments/` | One driver script and one folder per experiment; protocols, results, reports |
| `tests/` | Unit tests (`python -m unittest discover -s tests`) |
| `data/folios/` | Archived Voynich scans and pixel descriptions (tracked, ~120 MB) |
| `artifacts/` | Large or private working files: raw corpora, fitted priors, sealed answers, Linear A sources. Git-ignored, hashed in freezes |
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
| `verse_word_model.py` | Training-only verse augmentation; fresh transfer threshold failed | word-segmentation-v2 |
| `unknown_words.py` | Letter n-gram unknown-word cost, Good–Turing rate, elision join | word-segmentation-v3 |
| `lexical_polish_v3.py` | Polish cost under the v3 segmenter (round five arm C; not adopted) | round five |
| `variable_units.py` | Variable-length key beam, iterated local search, annealing, piece induction | rounds 1–4 |
| `joint_segments.py` | Joint segmentation-and-decipherment EM (latent parses, role emissions) | rounds 1–4 |
| `joint_segments_v2.py` | Same EM with a candidate-lexicon override and usage pruning | rounds 2–4 |
| `lexical_polish.py` | Word-level polish of a unit key using the segmenter's local cost | rounds 3–4 |
| `lexicon_repair.py` | Concatenation test and complement admission for the candidate lexicon, iterated with EM | round 4 |
| `association_complex.py`, `image_domains.py`, `folio_description.py` | Image-association studies and pixel descriptions | — |

"Frozen by" means a committed `freeze.json` records the file's hash; changing the file
breaks `verify` for that round. Add new behaviour in a new module instead.

## Linear A library (`linear_a/`)

| Module | Purpose | Frozen by |
|---|---|---|
| `spelling.py` | Syllable tuples; Linear B spelling rules for alphabetic words | Linear A rounds 1–3 |
| `corpus.py` | Readable Linear A words from the pinned Navarre-AI collation | rounds 1–3 |
| `lexicons.py` | Candidate-language lexicons from Wiktionary (kaikki) extracts | rounds 1–3 |
| `matching.py` | Syllable edit distance and lexical match score against a phonotactic null | rounds 1–3 |
| `controls.py` | Known-answer samples: Linear B signal words mixed with Linear A-like noise | rounds 1–3 |
| `anchors.py`, `arithmetic.py` | Cretan toponym check; `ku-ro` total check | round 1 |
| `lexicons_v2.py` | Lexicons with a proper-name flag per form | round 2 |
| `contexts.py` | Word context labels (entry, logogram, header, other) for Linear A and DĀMOS | rounds 2–3 |
| `context_test.py` | Name-versus-position agreement statistic and permutation null | round 2 |
| `names.py` | Proper-name lexicons: Greek, LAMAN Anatolian, Oracc Levant and Babylonia | round 3 |
| `probes.py` | The seven round-four probes, including the grammar profile | — (protocol committed) |
| `tlhdig.py` | Word forms by language tag from TLHdig Beta 0.3 | — (protocol committed) |

`linear_a/person_role_control.py` supplies object-held-out structural classification, balanced
metrics, feature ceilings and block-label negatives; frozen by `linear-b-person-role`.

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
| Verse word model | `word_segmentation_v2.py`, `word_segmentation_fresh.py`, `word_segmentation_records.py`, `report_word_segmentation.py` | `word-segmentation-v2/development.json` | `word-segmentation-v2/` (perfect letters; extraction caveat) |
| Length scaling | `length_scaling.py` | `length-scaling/` (development) | — |
| Voynich suspects | `voynich_suspects.py` | — | `voynich-suspects/` (training pages) |
| Language-ID control | `language_id.py` | — | `language-id/` (5/5 correct) |
| Round five | `joint_recovery_v5.py` (paired A/B/C segmenter arms) | round-four development | `joint-recovery-v5/` |
| Word segmentation v3 | `word_segmentation_v3.py` (rubric-free extractor, diagnosis, unknown-word selection), `word_segmentation_v3_fresh.py` | `word-segmentation-v3/development.json` | `word-segmentation-v3-fresh/` (Compagni, ParTUT; transfer passed) |

Linear A rounds have one driver and one record folder each. The Linear B control plays the role
of the fresh evaluation.

| Round | Driver(s) | Record folder | Outcome |
|---|---|---|---|
| One: lexicon match | `linear_a_development.py`, `linear_a_round_one.py`, `linear_a_round_one_report.py` | `linear-a/` (also `PLAN.md`, the track plan) | control failed |
| Two: tablet position | `linear_a_damos_crawl.py` (DĀMOS download), `linear_a_context.py` | `linear-a-context/` | control failed |
| Three: name lists | `linear_a_names.py` | `linear-a-names/` | control failed |
| Scoping after round three | — | `linear-a-next/` (`BACKGROUND.md` research brief, `CORRESPONDENCES.md` exploratory, `SCOPE.md` round-five data) | — |
| Four: seven probes | `linear_a_probes.py` | `linear-a-probes/` (`lists.json`, `results-1.json` to `results-7.json`) | no probe passed |
| Five: TLHdig profiles | `linear_a_tlhdig.py` | `linear-a-tlhdig/` | artefact |
| Repair audit | `linear_a_audit.py` | `linear-a-audit/` | corrected profiles/trade controls fail; full-set correspondence exploratory p = 0.0001 |
| Sign-only entry endings | `linear_a_structure_v2.py` | `linear-a-structure-v2/` | 4/20 known-answer samples pass; gate failed; uncorrected driver archived in `linear-a-structure/` |
| Correspondence source audit | `linear_a_correspondence.py` | `linear-a-correspondence/` (12-pair evidence, literal inventories, frozen protocol and conditional nulls) | 6 strict pairs; onset-conditioned p = .1605, specific adaptation lead archived |
| Accounting-program feasibility | `linear_a_ledger.py` | `linear-a-ledger/` (opaque inputs, coverage, source review, protocol and follow-on benchmark plan) | not evaluable: 7 KN objects vs 33 A; no real-data fit |
| Source-checked account benchmark | `linear_a_account_benchmark.py` | `linear-a-account-benchmark/` (quantities, source labels, relative units, source reviews) | nine development accounts; two strict balances; no A score |
| Kinship feasibility | `linear_a_kinship.py` | `linear-a-kinship/` (known B controls, literal markers, graphic shapes and role review) | named vs unnamed relatives; no A kinship inference |
| Person-slot source audit | `linear_a_person_slots.py` | `linear-a-person-slots/` (43-row inventory, source decisions, rival interpretations, exact concordance) | 2 objects; edition-supported reading overlay; no kinship formula |
| Qi-tu-ne role contrast | `linear_a_qi_tu_ne.py` | `linear-a-qi-tu-ne/` (HT7 source inventory, 3 reviewed cases, rival assignments) | heading/count contrast confirmed; semantic class unresolved |
| Name/designation control | `linear_b_person_role.py` | `linear-b-person-role/` (38 source cases, anonymous features, held-out predictions, 199 negatives) | curated development; failed 90% gate; no Linear A scoring |

Rounds one to three each hold `PROTOCOL.md`, `sources.json` (hashes and licences),
`development-results.json`, `freeze.json`, control or test results and `REPORT.md`. Rounds four and
five hold `PROTOCOL.md`, their results and `REPORT.md`; five also has `sources.json`.

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
`test_linear_a.py` checks the Linear B spelling rules, the matcher and its null, `ku-ro` sums
and tablet context labels (`.venv/bin/python -m unittest tests.test_linear_a`).

`linear_a/probes_v2.py` repairs type profiles, unique sampling and Monte Carlo resolution;
`linear_a/audit.py` enforces source/code freezes and refuses output overwrite;
`linear_a/structure.py` preserves sign IDs and tests entry endings on unseen types.
`linear_a/correspondence.py` filters literal source readings and permutes endings while
preserving real stems, lengths and optionally final onsets; frozen by the correspondence audit.
`linear_a/ledger.py` parses commodity-specific integer vectors and learns fixed opaque-word
arithmetic programs; frozen by the accounting feasibility pilot.
The audit/structural tests run with
`.venv/bin/python -m unittest discover -s tests -p 'test_linear_a*.py' -v`.

## What is not in git

`artifacts/` holds raw corpora (restorable from the pinned tarballs), fitted priors
(refittable from pinned texts, hashed in freezes), sealed answers and predictions (released
in `evaluated-records.tar.gz` after grading), and prediction-track model files.
`artifacts/linear-a-sources/` holds the Linear A track's downloads: `navarre/` (Linear A corpus),
`kaikki/` (Wiktionary lexicons), `damos/` (5,932 Linear B documents), `names/` (LAMAN and Oracc),
`tlhdig/` (TLHdig Beta 0.3) and `peet/` (Peet 1927 scan). Each round's `sources.json` hashes them.

`linear_a/account_benchmark.py` performs exact unit arithmetic on source-annotated development
accounts. `linear_a/kinship_inventory.py` preserves ordered graphic slots and literal marker
spellings. Each has its own committed freeze; neither is a semantic recovery model.
