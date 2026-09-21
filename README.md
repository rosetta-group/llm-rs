# Voynich meaning-recovery research

The deliverable is a **validated decipherment-method benchmark** with positive controls
and honest Voynich negatives. Translation into English or Italian remains a long-term
motivation; it is not supported by current evidence.

Start with the [research record](RESEARCH_LOG.md), [current protocol](experiments/standard-decipherment/PROTOCOL.md),
and [fresh standard-method report with graphs](experiments/standard-decipherment/REPORT.md).
The [methods note](experiments/method-benchmark/METHODS_NOTE.md) connects all tracks.

- **Prediction track closed.** No further BPC sweeps, longer training, or larger-model
  comparisons. Reopening requires a named mechanism, falsifiable contrast, and fixed budget.
- **Fresh historical segmentation:** Novellino/Decameron training lowers Dante word error
  from 37.7% to 24.2% on the same new passages; modern error falls 7.4% → 6.0%.
  Historical segmentation still misses the 10% gate. Earlier results remain archived.
- **Published cipher comparators + MDL:** exact letters in 7/8 fresh substitution/homophonic
  controls; the remaining case has 0.63% character error. Historical substitution selection
  is repaired. Naibbe fails; the reserved Voynich mechanism test stays closed.
- **Image track parked:** no further catalogue study. Reopening needs masked labels from
  two annotators, agreement statistics and an identifiable design.
- **Broad image domains:** botanical, people/bathing, and celestial/diagrams now tested
  across 63 physical folio groups. Character text alone scores 72.2% balanced accuracy;
  hand/layout scores 94.4%; adding text does not improve it. [Report](experiments/image-domains/REPORT.md).
- **Complex image associations:** 123 descriptions; joint visual features, nonlinear
  models, same-page matching, and relational tests. None of 12 comparisons establishes
  an association. [Report and graphs](experiments/association-complex/REPORT.md).
  The earlier 59-label root-color pilot is preserved.
- **Earlier codebook-free recovery:** exact historical letters were found but not selected.
  The new selector repairs this substitution failure; variable-length recovery remains unresolved.
- **Cloud cleanup complete:** Runpod pod and attached temporary storage deleted on
  2026-09-21; no network volumes remain. [Audit record](experiments/cloud-cleanup.json).

All previous experiments and reports remain archived. The Voynich final test is sealed.
The old prediction monitor is paused. Training commands below document earlier methods;
they are not the active research plan.

## Archived benchmark commands

```sh
.venv/bin/python -m experiments.discovery_sources
.venv/bin/python -m experiments.benchmark_sources
.venv/bin/python -m experiments.segmentation tune
.venv/bin/python -m experiments.segmentation prepare
.venv/bin/python -m experiments.segmentation solve
.venv/bin/python -m experiments.segmentation evaluate
.venv/bin/python -m experiments.codebook_free freeze
.venv/bin/python -m experiments.codebook_free prepare
.venv/bin/python -m experiments.codebook_free solve
.venv/bin/python -m experiments.codebook_free evaluate
.venv/bin/python -m experiments.association freeze
.venv/bin/python -m experiments.association run
python -m experiments.benchmark_report
.venv/bin/python -m experiments.benchmark_verify
```

These are the stage order, not instructions to overwrite completed work. Preparation,
freezing, and prediction refuse existing outputs. Preserve this run; a fresh replication
needs an isolated copy with its prior output records archived. Source downloads are
hash-checked. Reports need matplotlib; computation uses CPU and NumPy. Case keys are
random: a new run is a replication, not a byte-identical recreation. Exact grading needs
the preserved local `artifacts/` inputs, frozen model, predictions, and evaluator files.
Reports publish aggregate metrics, provenance, and limitations without reference passages.

## Complex image-association extension

```sh
.venv/bin/python -m experiments.association_complex freeze
.venv/bin/python -m experiments.association_complex run
.venv/bin/python -m experiments.association_complex verify
python -m experiments.association_complex_report
```

This completed extension reuses existing human image descriptions and is exploratory.
The protocol protects whole folios, tests three model classes and four endpoints, and
corrects all 12 comparisons together. A fresh replication needs an isolated output
archive; existing frozen outputs are protected. Reports regenerate from saved scores.
A larger independently annotated image sample remains necessary; this is not raw-pixel analysis.

## Broad image-domain study

```sh
.venv/bin/python -m experiments.image_domains freeze
.venv/bin/python -m experiments.image_domains run
.venv/bin/python -m experiments.image_domains verify
python -m experiments.image_domains_report
```

This completed study uses conventional IVTFF illustration categories, including labels
and circular text on diagram pages. Frozen outputs are protected; use an isolated output
archive for a new replication. People/bathing cannot be validated across quires here:
all its training examples share one quire. Some domain/hand effects are unidentifiable.

## Setup

Run from the repo root. The existing Poetry lock contains the dependencies.
The implementation was checked with Python 3.12, PyTorch 2.8, Transformers 4.55,
and PEFT 0.17. An existing `.venv` works with the commands below.

```sh
poetry install
poetry run python -m unittest discover -s tests -v
```

## Data and baselines

```sh
.venv/bin/python -m voynich build voynich_transliterations/GC2a-n.txt
.venv/bin/python -m voynich baseline artifacts/data/gc --output artifacts/results/gc.json
.venv/bin/python -m experiments.baselines
```

The last command runs all eight baseline datasets. It downloads small, pinned public
text files listed in `experiments/sources.json`; existing files are hash-checked.
It does not download models. Results include per-page scores and A/B, section, and hand summaries.

1. **Manuscript splits.** `experiments/splits/folio-42.json` keeps both sides and all panels
   of a folio together, including the shared 85/86/Ros foldout. `quire-42.json` holds out
   whole quires. Splits are fixed across models.
2. **Representations.** GC2a uses v101; ZL3b uses EVA on matched paragraph pages.
   Dots and commas distinguish certain and uncertain spaces. Extended IVTFF glyph escapes
   count as one transcription character. Line and paragraph boundaries remain in the text.
3. **Controls.** Shuffle written forms within each line. Also fit the published Naibbe
   ciphertext and Timm–Schinner sample separately, with chronological blocks and unused
   boundary blocks. Those blocks are synthetic, not manuscript folios or independent generator runs.
4. **Preservation.** `corpus.json` retains raw loci, including labels and circular text.
   `documents.json` models running paragraph text only. Page metadata is inherited from
   the transcription; `$H` is its hand label, not a new attribution to a scribe.

## Local training

```sh
.venv/bin/python run_fine_tuning.py experiments/smoke.json
.venv/bin/python -m voynich matrix mlx-community/Qwen3-1.7B-bf16 --contexts 64 --steps 400
.venv/bin/python run_fine_tuning.py experiments/generated/gc-outer-c64-s42.json
```

The smoke model has random weights and checks the pipeline only. Each output directory
is used once. Choose a new `output_dir` to rerun an experiment.

The matrix writes 15 configs: outer, middle, and random layers for seeds 42–46.
It does not start those runs. `model_path` accepts a cached Hugging Face ID or a local
Transformers-compatible checkpoint. The cached Qwen3-1.7B BF16 checkpoint works on this Mac's
Apple GPU. Llama projections are also supported. No hosted API or paid compute is used.

`local_files_only` defaults to true. `device` selects MPS, CUDA, or CPU automatically.
Use `dtype: "float32"` for CPU and `"bfloat16"` for the tested Qwen/MPS setup.
Optional 4-bit loading requires CUDA and bitsandbytes. Logging stays local by default.
Legacy `HF_TOKEN` is read from the environment.

For 28-layer Qwen, four outer layers means `[0, 1, 26, 27]`; middle means `[12, 13, 14, 15]`.
All seven attention/MLP projections receive equal-rank adapters. Layer position is an
experimental choice; the code does not assume those layers contain only language-specific features.

Each run saves its settings, selected layers, parameter count, data/split/code hashes,
package versions, frozen scores, adapted scores, and adapter weights under `training_run_outputs/`.
`frozen_reference` can reuse a completed run's frozen evaluation after checking its settings and data.

For a learning curve, set `max_steps: 3000`, `eval_steps: 500`, `save_steps: 500`,
and `load_best_model_at_end: true`. Every validation checkpoint gets per-page scores under
`validation/`; `learning_curve.json` records the curve. The run root exports the selected
adapter, and `run.json` records both completed updates and the selected update.
The local 3,000-update experiments are reported in `experiments/LEARNING_CURVES.md`.

The illustrated [interim research report](experiments/report/REPORT.md) explains the
results, uncertainty and next experiments. Its [PDF](output/pdf/voynich-research-report.pdf)
and four figures use the frozen scores in `experiments/report/snapshot.json`.
Rebuild with `python -m experiments.research_report` in an environment containing
matplotlib and reportlab. This reads saved scores; it does not train or score the test set.
To replace the snapshot, refresh `experiments.learning_curves`, run the report builder
with `--capture`, and review its narrative for changes in experiment status.

The archived runner `python -m experiments.replications` selects the better
configuration. If its paired validation interval supports a gain over the copy baseline,
it runs four more seeds with that layer subset fixed, then three text controls at seed 42.
Each follow-up has the same 3,000-update budget. This command starts training and can take
several hours locally; completed results go to `experiments/REPLICATIONS.md`.

## Comparisons

The completed $10 Runpod comparison is specified in
[`experiments/CLOUD_PLAN.md`](experiments/CLOUD_PLAN.md). Its upload bundle excludes
final-test text and local credentials. Cloud runs use a separate environment and
are archived locally. The pod and its temporary storage have now been deleted.

```sh
.venv/bin/python -m voynich compare artifacts/results/gc.json \
  training_run_outputs/gc-outer-c64-s42/adapted.json --reference-model copy \
  --output artifacts/results/qwen-vs-copy.json
.venv/bin/python -m voynich evaluate experiments/generated/gc-outer-c64-s42.json \
  --adapter training_run_outputs/gc-outer-c64-s42 --context 16 \
  --output artifacts/results/qwen-context16.json
```

1. **Common targets.** Windows keep token IDs and score each target once. Padding,
   repeated context, and tokens containing unreadable `?` are excluded from loss.
   BPC includes scored line/boundary characters. Comparison rejects unequal page targets,
   even when they have the same length. A tokenizer that merges `?` with adjacent characters
   can mask extra characters; such scores cannot be directly paired with character baselines.
2. **Uncertainty.** `compare` reports a paired 95% bootstrap interval over folios
   (`--group quire` for the quire split). Positive delta favors the candidate. It describes
   page sampling uncertainty, not variation across training seeds.
3. **Context.** Evaluate the same adapter at 16, 64, and 256 tokens to isolate available
   context. Default stride is half the context, so targets receive varying preceding context,
   capped at the selected length. Use `--stride 1` for the full available context at every target;
   this costs more. Matrix budgets are matched within each context, not across context lengths.
4. **Units.** Token accuracy is comparable only for the same tokenizer. Compare BPC only
   on the same normalized text and masks. EVA and v101 BPC values are not directly comparable.
   Compare each model's gain within its own representation.

Scoring defaults to validation. `--split test` requires `--release-test` after model choices
are frozen. The test set has not been scored in the initial experiments.

## Learned symbol groups

```sh
.venv/bin/python -m voynich tokenizer artifacts/data/gc --output artifacts/tokenizers/gc-bpe64
```

This fits reversible BPE groups on training pages only and records those page IDs.
The fixed alphabet includes all 256 normalized transcription characters. To test the tokenizer
plumbing, set `tokenizer_path` and `tiny: true` in a smoke config. Replacing a pretrained model's
tokenizer would require training new embeddings; that experiment is not implemented.

## Evidence and remaining work

Small character models trained from scratch are described in
[`experiments/CHARACTER_PLAN.md`](experiments/CHARACTER_PLAN.md). Their local suite
completed all twelve runs without downloading weights and kept a 20 GiB free-disk reserve.
Run `.venv/bin/python -m experiments.characters report` to refresh completed results.

The [context plan](experiments/context-plan.json) evaluates saved GRUs on the same validation
targets at 8, 16, 32, 64, and 128 characters. Run `.venv/bin/python -m experiments.context`
once; it refuses to overwrite `artifacts/context-ablation`. The completed run used local
hardware, trained no models, and downloaded no weights. Its stride-one scores differ from
the earlier stride-64 evaluation. See [protocol, results, and reproduction](experiments/CONTEXT.md).

## Corpus statistics and controlled recovery

```sh
.venv/bin/python -m experiments.discovery_sources
.venv/bin/python -m experiments.languages
.venv/bin/python -m experiments.decipherment prepare
.venv/bin/python -m experiments.decipherment solve
.venv/bin/python -m experiments.decipherment evaluate
.venv/bin/python -m experiments.boundaries
python -m experiments.discovery_report
```

The source command downloads missing pinned text/code files and verifies SHA-256 hashes;
existing files are checked. Report generation needs matplotlib. Prepare/solve and the
boundary diagnostic refuse existing benchmark outputs: preserve this run and use an
isolated checkout for a new benchmark. New random keys produce a new run, not a byte-identical
reproduction. Original predictions and hidden keys are retained locally under
`artifacts/decipherment/`; public scores and passages are in `experiments/decipherment/`.

The old boundary diagnostic is an exploratory repair. The new lexicon benchmark uses
fresh passages, preserves that failed primary baseline, and still finds poor historical
word recovery; see the current benchmark report.

See `RESEARCH_PLAN.md` for research gates. Broad neural comparisons are closed. The image
pilot is implemented; independent visual re-annotation remains outstanding. No Voynich translator is implemented.

Sources: [IVTFF specification](https://www.voynich.nu/software/ivtt/IVTFF_format.pdf),
[ZL transcription mirror](https://github.com/karolrybak/voynichese),
[Greshko's Naibbe code and data](https://github.com/greshko/naibbe-cipher),
[Timm–Schinner generated sample](https://github.com/TorstenTimm/SelfCitationTextgenerator).
Naibbe data is credited to Michael A. Greshko (2025),
[The Naibbe cipher](https://doi.org/10.1080/01611194.2025.2566408).
Downloaded data and licenses remain under `artifacts/sources/`.

### Standard decipherment comparison

Current protocol: [standard-method benchmark](experiments/standard-decipherment/PROTOCOL.md).
Image modelling is parked pending independent annotations. Prediction sweeps remain closed.
The Voynich mechanism test is reserved until the Naibbe recovery gate passes.

```sh
.venv/bin/python -m pip install -r experiments/standard-decipherment/requirements.txt
.venv/bin/python -m experiments.historical_sources prepare
.venv/bin/python -m experiments.standard_decipherment develop
.venv/bin/python -m experiments.standard_decipherment freeze
# Commit the method and freeze before preparing evaluation cases.
.venv/bin/python -m experiments.standard_decipherment prepare
.venv/bin/python -m experiments.standard_decipherment solve
.venv/bin/python -m experiments.standard_decipherment segment_controls
.venv/bin/python -m experiments.standard_decipherment evaluate
```

Existing freezes refuse overwrite. `verify` checks the committed method and pinned inputs.
Raw prose, models and private answers live under ignored `artifacts/`; source manifests,
development results, protocols and aggregate evaluation results are tracked.

For the committed completed run, restore exact prose with
`python -m experiments.historical_sources restore`. With the earlier pinned UD corpora
and frozen modern segmenter available, `python -m experiments.standard_decipherment_rebuild`
recreates missing priors without tuning. `python -m experiments.standard_decipherment_records restore`
restores the exact graded challenge and predictions; its plaintext references are now disclosed.
`python -m experiments.standard_decipherment_audit` verifies the saved result. The report
regenerates with `python -m experiments.standard_decipherment_report` (matplotlib required).
Do not reuse disclosed source IDs as fresh tests. Source/data licenses accompany both archives.
