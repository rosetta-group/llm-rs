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
.venv/bin/python -m experiments.joint_recovery_v3 verify       # frozen files unchanged and committed
.venv/bin/python -m experiments.joint_recovery_v3 evaluate     # regrades archived predictions
```

Rounds one and two regrade the same way with `joint_recovery` and `joint_recovery_v2`.

Running a new round follows [CONVENTIONS.md](CONVENTIONS.md), with a new `_v4` pair of drivers:

```sh
.venv/bin/python -m experiments.joint_development_v4            # development table
.venv/bin/python -m experiments.joint_recovery_v4 freeze         # then commit
.venv/bin/python -m experiments.joint_recovery_v4 prepare        # new sealed passages, new keys
.venv/bin/python -m experiments.joint_recovery_v4 solve          # ~5 min per case
.venv/bin/python -m experiments.joint_recovery_v4 evaluate
```

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

## Sources and licences

Pinned public texts and code, with revisions, hashes and licences, are listed in
`experiments/sources.json`, `language-sources.json`, `decipherment-sources.json`,
`segmentation-sources.json`, `standard-decipherment/sources.json` and `verse-prior/sources.json`.
Naibbe code and data: Michael A. Greshko (2025), *The Naibbe cipher*, Cryptologia,
modified MIT licence. Wikisource transcriptions: CC BY-SA. Universal Dependencies
treebanks: see each manifest. Yale scans: see `data/folios/sources/yale-manifest.json`.
