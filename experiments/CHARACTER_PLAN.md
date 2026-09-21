# Character models from scratch

Status: all 12 runs completed and saved weights verified on 2026-09-20.
See `CHARACTERS.md` and `report-completed/REPORT.md`.

Question: can a small model trained only on Voynich match the adapted Qwen?
This is a practical baseline comparison. Model architecture, training exposure,
context, and full-model versus adapter training differ; it does not isolate pretraining.

## Fixed design

| Setting | Transformer | GRU recurrent model |
|---|---:|---:|
| Trainable parameters | 1,272,448 | 1,316,608 |
| Width / layers | 128 / 6 | 256 / 3 |
| Input context / scored stride | 128 / 64 characters | 128 / 64 characters |
| Batch size | 16 | 16 |
| AdamW learning rate | 0.0003 | 0.0003 |
| Maximum epochs | 20 | 20 |

Use a fixed 256-character alphabet plus input-only page-start and padding IDs.
Exclude unreadable `?` targets. Never concatenate pages. Both models reset at every
window; targets get up to 128 preceding characters, varying with position in the window.
Vocabulary construction needs no validation or test text.

Train both models with seeds 42, 43, 44 on GC. Train both with seed 42 on GC shuffled
within lines, Timm, and Naibbe. These last two are single synthetic samples.
All runs use the existing folio splits; only training and validation pages enter a model.

Evaluate after every epoch. Keep the lowest-validation-loss checkpoint. Stop after
four epochs without improvement, at 20 epochs, or at the local time budget.
Report epoch 2 separately: Qwen saw approximately two passes, but token windows and
batching differ, so this is not exact exposure matching. Record actual characters seen.
Character models may train much longer; compare learning curves as well as selected scores.

Scores must match baseline and Qwen page IDs, character counts, and target hashes.
Reload saved weights and reproduce selected scores. Report paired folio uncertainty
for real/shuffled Voynich, and point scores for synthetic samples. Final test stays sealed.

## Cost, disk, and queue

- Local MPS only. No pretrained model downloads or cloud charges.
- The filesystem reported 55 GiB available at setup. Reserve at least 20 GiB free.
- Maximum character outputs: 1 GiB. One saved weight file per run; expected total below 100 MiB.
- Maximum 30 minutes training per run, six hours across the suite, plus evaluation/verification overhead.
- Wait on the existing replication-suite lock. Check that every Qwen job completed
  before using MPS. Refuse duplicate launches and existing output directories.
- Leave all existing experiments and downloaded models intact.

```text
Wait for the local Qwen suite to finish successfully
Train the two character models on real text and controls
Verify each saved checkpoint and refresh the comparison report
Review seed variation, learning curves, and shuffled-text results
Decide whether another pretrained family would answer an unresolved question
```

Run: `.venv/bin/python -u -m experiments.characters suite`.
State/logs: `artifacts/characters/`. Results: `experiments/CHARACTERS.md`.
Fixed machine-readable settings: `experiments/character-plan.json`.

Matching Qwen would show that prior language training is unnecessary for this level
of prediction on this split. Falling short would not prove that Qwen learned meaning;
optimization and architecture remain alternative explanations.
