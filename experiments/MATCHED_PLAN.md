# Matched-context training

Status: authorized; fixed protocol for the 18-run local suite.

The goal is translation and recovery of meaning. This experiment closes a modeling
ambiguity before the next meaning-recovery benchmark; its scores are not translations.
The [research record](../RESEARCH_LOG.md) contains the goal, history, and next semantic milestone.

**Matched context:** train and evaluate with the same maximum preceding history.
**BPC:** bits per readable transcription character; lower is better.
**Interaction:** the gain from longer history on intact text minus that gain on shuffled text.

```text
For each seed in 42, 43, 44:
    For each history length in 8, 32, 128:
        Train a fresh GRU on intact Voynich
        Train the same initial GRU on within-line shuffled Voynich
        Evaluate each at its trained history length
Compare the fixed final-epoch scores
Measure the 8-to-128 gain within each text and its interaction
Keep final-test pages sealed
```

## Why this is needed

The [previous memory test](CONTEXT.md) shortened history only during evaluation.
Its GRUs had learned mostly from 65–128 preceding symbols. The Naibbe model was worse
at 32 symbols than at 8, so the curves could reflect a training mismatch as well as
information in the text. Fresh training at each history length removes that particular mismatch.

## Fixed design

| Setting | Value |
|---|---|
| Architecture | Character GRU, width 256, 3 layers, 1,316,608 parameters |
| Training | Random initial weights; all parameters trained; no pretrained models |
| Data | GC2a v101 and the existing within-line word shuffle |
| Histories | 8, 32, 128 characters; exact rolling history at each target |
| Seeds | 42, 43, 44; identical initial weights across conditions within each seed |
| Targets | One readable next character per example; `?` remains in context |
| Exposure | 125,566 training targets per pass × 20 passes = 2,511,320 per run |
| Batch | 256 training targets; 64 evaluation targets |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.01, gradient clip 1 |
| Primary checkpoint | Final epoch 20, with no early stopping or validation selection |
| Secondary checkpoint | Best validation epoch, labeled separately; unequal selected exposure is possible |
| Validation | Same 29 pages and 19,752 scored characters; 15 source folio groups |
| Hardware | Local Apple GPU, one worker at a time |
| Limits | 2 hours per run, 12 hours per suite, 20 GiB free-disk reserve, 1 GiB outputs |

Each target receives the available suffix of its own page, including boundaries.
Page starts have shorter histories; a beginning-of-page marker occupies one slot.
Hidden state resets for every training and validation example. Padding follows the
history, and the loss uses the last real state. No history crosses page boundaries.

The models see each readable training target once per epoch, in a seeded shuffled order.
The architecture, optimizer, updates, target exposure, and initial weights are matched.
Longer histories require more computation. Equal exposure does not guarantee convergence.

## Analysis fixed before launch

1. **Primary within-text contrast.** BPC at 8 minus BPC at 128, reported for each
   seed and dataset. Positive values favor longer context.
2. **Primary interaction.** Subtract the shuffled-text contrast from the intact-text
   contrast. Each contrast requires identical target hashes within its dataset.
   Across datasets, match source pages, folios, and character counts, not text hashes.
   Resample the corresponding source folios together: 2,000 bootstrap draws, seed 42.
3. **Replication.** Report every seed's interaction, its 95% folio interval, and the
   mean and range across the three seeds. Do not treat seeds × folios as independent
   manuscript samples. Treat consistent signs as exploratory evidence; a stronger
   result has a positive lower interval bound in every seed.
4. **Secondary diagnostics.** Show 32-character scores and training/validation curves.
   Best-epoch scores cannot replace the primary final-epoch comparison. A time-limited
   or failed run remains incomplete and is not silently compared at lower exposure.

A positive interaction would identify an order-related prediction effect under this
shuffle control. It would not distinguish syntax from repeated local sequences, nor
prove meaning. A null interaction would not prove meaningless text. Validation has
already guided earlier experiments; independent confirmation still needs held-out data.

## Execution and evidence

```sh
.venv/bin/python -m unittest discover -s tests -v
caffeinate -i .venv/bin/python -u -m experiments.matched suite
.venv/bin/python -m experiments.matched report
```

- `matched-plan.json`: machine-readable fixed settings.
- `artifacts/matched/manifest.json`: frozen input/code hashes, start time, and budget.
- `artifacts/matched/status.json`: current job and overall state.
- `artifacts/matched/*.log`: worker logs; `artifacts/matched.log`: parent log.
- `training_run_outputs/matched-*`: configurations, initial/final/best weights,
  per-epoch scores, target hashes, learning curves, and reload verification.
- [MATCHED.md](MATCHED.md) and `matched-results.json`: refreshed after each completed run.

The suite refuses existing output directories and shares the local GPU lock with
earlier suites. It does not download models, rent compute, or read test pages for scoring.
Interrupted runs are preserved for diagnosis; a retry requires reviewing the recorded state.

## Next meaning milestone

Prepare a verified ciphertext/plaintext benchmark using a known cipher such as Naibbe.
Separate recovery under a learned key from recovery under unseen keys; hold out source
passages and keep test plaintext unavailable during fitting. Measure recovered content
against the originals before considering English or Italian rendering. The existing
single Naibbe ciphertext sample alone is not such a benchmark.
