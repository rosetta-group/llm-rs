# Reproducing results

All commands run from the repository root with the project virtual environment. Python
3.12, PyTorch 2.8, Transformers 4.55, PEFT 0.17 were used; recovery rounds need only NumPy,
Numba and BeautifulSoup.

```sh
poetry install
.venv/bin/python -m unittest discover -s tests -v
```

Large working files live in `artifacts/` (git-ignored). Pinned texts restore from tarballs;
fitted priors refit deterministically; sealed challenge records unpack from each round's
`evaluated-records.tar.gz`.

## Cipher recovery rounds

Regrading an archived round (no new randomness):

```sh
.venv/bin/python -m experiments.historical_sources restore     # Novellino and Decameron pages
.venv/bin/python -m experiments.verse_sources restore          # Petrarch pages (round two onward)
.venv/bin/python -m experiments.joint_recovery_v4 verify       # frozen files unchanged and committed
.venv/bin/python -m experiments.joint_recovery_v4 evaluate     # regrades archived predictions
```

Rounds one to three regrade the same way with `joint_recovery`, `joint_recovery_v2` and `joint_recovery_v3`.

Running a new round follows [CONVENTIONS.md](CONVENTIONS.md), with a new `_v5` pair of drivers:

```sh
.venv/bin/python -m experiments.joint_development_v5            # development table
.venv/bin/python -m experiments.joint_recovery_v5 freeze         # then commit
.venv/bin/python -m experiments.joint_recovery_v5 prepare        # new sealed passages, new keys
.venv/bin/python -m experiments.joint_recovery_v5 solve          # ~6 min per case
.venv/bin/python -m experiments.joint_recovery_v5 evaluate
```

The ISDT modern test split is exhausted; a modern half needs a newly pinned corpus, recorded by
revision and hash like `experiments/language-sources.json`.

New passages draw new random keys, so a rerun reproduces the procedure, not the bytes.

Earlier recovery benchmarks (assisted decipherment, segmentation, codebook-free v0,
standard methods) keep their own drivers with the same stage names:

```sh
.venv/bin/python -m experiments.decipherment prepare|solve|evaluate
.venv/bin/python -m experiments.segmentation tune|prepare|solve|evaluate
.venv/bin/python -m experiments.codebook_free freeze|prepare|solve|evaluate
.venv/bin/python -m experiments.standard_decipherment develop|freeze|prepare|solve|segment_controls|evaluate
```

`standard_decipherment_records restore` unpacks that round's graded challenge;
`standard_decipherment_audit` checks it; `standard_decipherment_report` redraws its figures
(needs matplotlib).

## Rejection and fixed-key transfer screen

The [first attempt](../experiments/rejection-transfer/REPORT.md) stopped on refinement
caps. The [resource-repair attempt](../experiments/rejection-transfer-v2/REPORT.md)
keeps the same decoder and thresholds, with a larger refinement allowance and two
Numba threads per worker. These commands verify the existing records; they do not
fit a new key or open unused challenge answers:

```sh
.venv/bin/python -m experiments.rejection_transfer verify
.venv/bin/python -m experiments.rejection_transfer_v2 verify
.venv/bin/python -m experiments.verify_rejection_records rejection-transfer
.venv/bin/python -m experiments.verify_rejection_records rejection-transfer-v2
.venv/bin/python -m unittest tests.test_rejection_transfer tests.test_rejection_transfer_v2 -v
```

The replay verifier reads each committed `evaluated-records.tar.gz` directly. It
reconstructs graded ciphertext from pinned source sentences and saved seeds, checks
sealed-key hashes, repeats transfer with fixed keys, and recomputes decisions and
true-language character errors. Frozen external sources and priors must be present
at the paths in each `freeze.json`; the archive alone is not a self-contained runtime.

Historical stage order was `sources`, `freeze`, commit, `prepare`, `run`, `report`
under `experiments.rejection_transfer` or `experiments.rejection_transfer_v2`.
Creation stages refuse overwrites. Any future experiment needs a new driver, freeze,
and fresh keys/passages. Exclude the consumed source IDs from **both** attempts,
listed in `experiments/rejection-transfer-v2/released-source-ids.json`, as well as all
earlier training and graded material. Unrun cases were not graded or released.

### Three development follow-ups

The [follow-up report](../experiments/rejection-followups/REPORT.md) compares a
transfer-centered rejection rule, exact-frequency copying controls, and bounded
incremental refinement. It uses released examples only. The executed control freeze
is `freeze-v2.json`: the initial 50-sweep guard was corrected to 200 before any fit,
with time and proposal budgets unchanged. Both freezes remain in the record.

```sh
.venv/bin/python -m experiments.rejection_followups verify
.venv/bin/python -m experiments.rejection_followups_v2 verify
.venv/bin/python -m experiments.audit_rejection_followups
NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_rejection_followups -v
```

The audit checks the frozen files, regenerates copied passages from the released
positive ciphertexts and fixed seeds, checks exact token multisets, replays saved-key
transfers, and reproduces both decision rules and all 21 threshold-sensitivity points.
It does not repeat fitting or open unused answers. The archive stores working records;
restore them under `artifacts/rejection-followups/` and restore the earlier frozen
priors/sources before replay. Benchmark timings are machine-dependent: the committed
plan specifies two warmed repetitions per backend with two threads. Repeating timings
requires a new output location; do not overwrite the frozen benchmark record.

### Historical language coverage

The [coverage pilot](../experiments/language-coverage/REPORT.md) adds Old Catalan and
compares the original Latin/German models with models containing historical charters
and prose. Every model uses 400,000 training letters. Its three passage pairs use
different random keys and the existing transfer-centered rule.

```sh
.venv/bin/python -m experiments.language_coverage_sources download
.venv/bin/python -m experiments.language_coverage verify
.venv/bin/python -m experiments.language_coverage replay
NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_language_coverage -v
```

Restore `evaluated-records.tar.gz` under `artifacts/language-coverage/` without
overwriting existing records. The archive includes normalized model inputs and all
evaluated records. Rebuild each prior with `CharacterPrior.fit` on its `train` rows
in `partitions.json`; compare the probability hashes in `freeze.json`. Earlier frozen
sources and priors are also required by `verify`. The audit module additionally
rebuilds all eight priors and reproduces encryption; its no-clobber `audit.json` output
must be absent in a reproduction checkout. Do not overwrite the published audit.

Future fresh rounds must exclude the complete source groups in
`experiments/language-coverage/released-source-ids.json` as well as previously released
material. Catalan folio separation is within one chronicle, not an independent-author test.

### Old Czech and Old Occitan extension

The [eight-language extension](../experiments/language-expansion/REPORT.md) preserves
the six active models and adds two equal-budget priors. Each new language has one
fresh key and two different source works; the comparison requires 16 fits.

```sh
.venv/bin/python -m experiments.language_expansion_sources download
.venv/bin/python -m experiments.language_expansion verify
.venv/bin/python -m experiments.language_expansion replay
NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_language_expansion -v
```

Restore its evaluated-records archive under `artifacts/language-expansion/`, without
overwriting existing files. Rebuild priors from its `partitions.json` training rows
with `CharacterPrior.fit`, save them as `priors/{model}.npz`, and compare the frozen
probability hashes. Earlier coverage sources/models and their frozen dependencies
are still required. `audit_language_expansion` also re-extracts new source rows,
rebuilds models and replays encryption; its sealed outputs must be absent in a
reproduction checkout. Do not rerun `prepare` against already evaluated sources.

Future fresh rounds must exclude complete works/manuscripts in
`language-expansion/released-source-ids.json`, parallel versions, and all earlier
exclusions. Czech derivatives retain **CC BY-NC-SA 4.0**; Occitan derivatives retain
**CC BY 4.0**. Attribution and changes are recorded in the archive's `ATTRIBUTION.md`.

## Prediction track (closed; commands kept for the record)

Data and baselines:

```sh
.venv/bin/python -m voynich build voynich_transliterations/GC2a-n.txt
.venv/bin/python -m voynich baseline artifacts/data/gc --output artifacts/results/gc.json
.venv/bin/python -m experiments.baselines
```

Local training (Apple GPU or CPU; smoke config uses random weights):

```sh
.venv/bin/python run_fine_tuning.py experiments/smoke.json
.venv/bin/python -m voynich matrix mlx-community/Qwen3-1.7B-bf16 --contexts 64 --steps 400
.venv/bin/python run_fine_tuning.py experiments/generated/gc-outer-c64-s42.json
```

Comparisons and evaluation at other context lengths:

```sh
.venv/bin/python -m voynich compare artifacts/results/gc.json \
  training_run_outputs/gc-outer-c64-s42/adapted.json --reference-model copy \
  --output artifacts/results/qwen-vs-copy.json
.venv/bin/python -m voynich evaluate experiments/generated/gc-outer-c64-s42.json \
  --adapter training_run_outputs/gc-outer-c64-s42 --context 16 \
  --output artifacts/results/qwen-context16.json
```

Character models, context ablation, matched-context suite and replications:

```sh
.venv/bin/python -m experiments.characters report
.venv/bin/python -m experiments.context            # refuses to overwrite artifacts/context-ablation
.venv/bin/python -m experiments.matched            # see experiments/MATCHED_PLAN.md
.venv/bin/python -m experiments.replications
```

Reports (matplotlib and reportlab):

```sh
python -m experiments.research_report
python -m experiments.completed_report
python -m experiments.matched_report
python -m experiments.context_report
```

Each training run saves settings, selected layers, parameter counts, data and code hashes,
package versions, frozen and adapted scores, and adapter weights under `training_run_outputs/`.
Choose a new `output_dir` to rerun; directories are used once.

## Corpus statistics and images

```sh
.venv/bin/python -m experiments.discovery_sources     # pinned downloads with hash checks
.venv/bin/python -m experiments.languages
python -m experiments.discovery_report
.venv/bin/python -m experiments.association_complex
.venv/bin/python -m experiments.image_domains
.venv/bin/python -m experiments.download_folios       # Yale scans, hashed; see data/folios/README.md
```

## Segmentation audit

```sh
.venv/bin/python -m pip install -r experiments/audit_report_requirements.txt
.venv/bin/python -m experiments.report_segmentation_audit
.venv/bin/python -m experiments.verify_segmentation_audit
```

The first command installs plotting dependencies only. Verification rechecks source
text, deterministic ciphertext, scores, word predictions and the rejection decision
without rerunning cipher search. The [audit report](../experiments/segmentation-audit/REPORT.md)
explains how to rerun into a new directory. This is development data, not a fresh test.

## Training-only verse word model

The completed eight-passage test is released. Do not prepare another challenge from
the same source IDs or alter the frozen files to improve the recorded results.
With the prior baseline and training sources restored:

```sh
.venv/bin/python -m experiments.word_segmentation_records restore
.venv/bin/python -m experiments.word_segmentation_records verify
.venv/bin/python -m experiments.report_word_segmentation
```

The verifier checks archive/source hashes and the pre-evaluation commit, reconstructs
the exact passage selection, and regrades development and fresh saved predictions.
It does not run a new cipher test. Figures use the pinned matplotlib stack in
`experiments/audit_report_requirements.txt`. The historical source retains chapter
rubrics; see [the report](../experiments/word-segmentation-v2/REPORT.md) for the deviation.
The original model-selection and fresh pipeline commands were:

```text
word_segmentation_v2 develop
word_segmentation_v2 freeze
Commit the method before downloading fresh sources
word_segmentation_fresh prepare
word_segmentation_fresh solve
word_segmentation_fresh evaluate
```

These are historical stages, not commands to overwrite the released experiment.

## Word segmentation v3

Development only; reads the saved v2 predictions and the released Villani source
(restore it first with `word_segmentation_records restore`).

```sh
.venv/bin/python -m experiments.word_segmentation_v3 check
.venv/bin/python -m experiments.word_segmentation_v3 diagnose
.venv/bin/python -m experiments.word_segmentation_v3 verify
```

The fresh Compagni/ParTUT test is released; restore and check it with:

```sh
.venv/bin/python -m experiments.word_segmentation_v3_fresh restore
.venv/bin/python -m experiments.word_segmentation_v3_fresh verify
```

`verify` (v3) checks the committed freeze and refits the training-only unknown rate and
spelling model (about 25 s, 0.5 GB). See [the report](../experiments/word-segmentation-v3/REPORT.md).

## Object-and-relation pilot

```sh
.venv/bin/python -m experiments.describe_folio_objects
.venv/bin/python -m unittest tests.test_folio_objects tests.test_folio_description
```

The script validates reviewed observations and derives tables/galleries; it does
not detect semantic objects automatically. Changed inputs get a new `analysis_id`.
See [the pilot report](../data/folios/object-pilot/REPORT.md) for pending text masks,
independent reviewer forms and the agreement command. Do not use unfilled templates
as annotations or send the development gallery as a blind review packet.

## Linear A track (branch `linear-a`)

Sources are not in git. Download them into `artifacts/linear-a-sources/` from the URLs in each
round's `sources.json`. DĀMOS is crawled at one request per second with
`python experiments/linear_a_damos_crawl.py` (5,932 documents). The `verify` stage of rounds one
to three checks frozen code and sources against the recorded hashes and fails on a mismatch.
Rounds four and five have no `verify`; check their inputs against the four `sources.json` files
(rounds one, two, three and five) with `shasum -a 256`. Do not rerun the `sources` or `freeze`
stages: they rewrite committed records. `develop` rewrites `development-results.json`; compare it
with git. Tests:

```sh
.venv/bin/python -m unittest tests.test_linear_a
```

Round one ([report](../experiments/linear-a/REPORT.md)):

```sh
.venv/bin/python -m experiments.linear_a_development
.venv/bin/python -m experiments.linear_a_round_one verify
.venv/bin/python -m experiments.linear_a_round_one control
.venv/bin/python -m experiments.linear_a_round_one descriptive
.venv/bin/python -m experiments.linear_a_round_one_report
```

Round two ([report](../experiments/linear-a-context/REPORT.md)):

```sh
.venv/bin/python -m experiments.linear_a_context develop
.venv/bin/python -m experiments.linear_a_context verify
.venv/bin/python -m experiments.linear_a_context test
```

Round three ([report](../experiments/linear-a-names/REPORT.md)):

```sh
.venv/bin/python -m experiments.linear_a_names develop
.venv/bin/python -m experiments.linear_a_names verify
.venv/bin/python -m experiments.linear_a_names test
```

Rounds four and five ([probes](../experiments/linear-a-probes/REPORT.md),
[TLHdig](../experiments/linear-a-tlhdig/REPORT.md)). These drivers, like `control` and `test` above,
refuse to overwrite results, so move the committed files aside first:

```sh
.venv/bin/python -m experiments.linear_a_probes      # results-1.json to results-7.json
.venv/bin/python -m experiments.linear_a_tlhdig      # results.json
```

Round five's post-hoc checks were run interactively; their numbers are in its report only.
No round ran the `linear-a` stage, because no Linear B control passed the gate.

## Sources and licences

Pinned public texts and code, with revisions, hashes and licences, are listed in
`experiments/sources.json`, `language-sources.json`, `decipherment-sources.json`,
`segmentation-sources.json`, `standard-decipherment/sources.json`, `verse-prior/sources.json`
and `word-segmentation-v2/sources.json`. Linear A sources are in `linear-a/`, `linear-a-context/`,
`linear-a-names/` and `linear-a-tlhdig/sources.json`; licences in [LINEAR_A.md](LINEAR_A.md#data-and-licences).
Naibbe code and data: Michael A. Greshko (2025), *The Naibbe cipher*, Cryptologia,
modified MIT licence. Wikisource transcriptions: CC BY-SA. Universal Dependencies
treebanks: see each manifest; newly archived VIT is **CC BY-NC-SA 3.0**, with its
README and licence in the archive. Yale scans: see `data/folios/sources/yale-manifest.json`.
