# Bounded cloud comparison

Authorized on 2026-09-20: one Runpod GPU, up to four hours, $10 total.
Pod `6o8irlwqlzhsb1` ran from 12:26 to about 13:14 UTC, then was stopped.
The console showed $9.18 remaining from the $10 credit. The 50 GB stopped volume
still costs $0.014/hour until permanent cleanup is confirmed.

Both models completed 3,000 updates. Validation loss was 2.4771 bits/character
for 1.7B and 2.4689 for 8B. The paired gain interval includes zero, so the planned
shuffled-text cloud pair was skipped. See `CLOUD_RESULTS.md`.

The downloaded archive matches its remote SHA256. Both exported adapters and
selected scores match checkpoint 3,000. Final-test text was never uploaded.

The original watchdog process was alive, but its command detection accepted root
help for an unsupported CLI command. Manual shutdown used the verified
`runpodctl stop pod` syntax. Local detection now requires matching command usage;
two regression tests cover this failure. Do not rely on the old watchdog code.

## Question

Does a larger pretrained model improve Voynich prediction under matched data exposure?

Compare Qwen3-1.7B-Base with Qwen3-8B-Base. The existing chat-trained 1.7B results
remain a separate reference. Lower validation loss is not a translation claim.

## Fixed comparison

| Setting | 1.7B | 8B |
|---|---:|---:|
| Outer layers, zero-based | 0, 1, 26, 27 | 0, 1, 34, 35 |
| LoRA rank | 8 | 4 |
| LoRA alpha | 16 | 8 |
| Trainable parameters | 1,245,184 | 1,212,416 |
| Context / stride | 64 / 32 | 64 / 32 |
| Batch / accumulation | 1 / 2 | 1 / 2 |
| Seed | 42 | 42 |
| Learning rate | 0.0001 | 0.0001 |

The adapters differ by 2.6% in parameter count. Layer depth and model pretraining
also differ, so this is a practical model comparison, not an isolated causal test of size.
Pinned revisions are in `cloud-plan.json`. Locally verified identical token IDs,
target masks and character units on GC training/validation pages. The remote suite
rechecks both GC and shuffled data. No final-test text is uploaded.

```text
Verify the Runpod quote and account balance
Record the actual provisioning time
Arm a pod-side stop deadline before installing dependencies
Download the two pinned models
Benchmark both models within ten minutes
Select a shared 500-3,000 update budget from timing alone
Train both from fresh adapters and compare validation loss
If the paired interval supports the larger model and time remains:
    Train both on within-line shuffled text
Download and verify outputs
Stop the pod; remove temporary storage after preserving results
```

## Budget and stopping

- Preferred GPU: one L40S, 48 GB. Use an on-demand quote, no subscription.
- Accept at most $1.50/hour **including running storage**. Four hours then costs
  at most $6, leaving $3 reserved and another $1 margin within the $10 budget.
- The immutable deadline starts at provisioning, not at training.
- A separate pod-side watchdog invokes the installed, pod-scoped Runpod CLI.
- Training stops admitting work ten minutes before the deadline, allowing retrieval.
- The watchdog stops the GPU even if the local computer disconnects. Stopped volume
  storage still bills; export and remove the temporary pod/volume promptly.
- If setup or the benchmark fails, retrieve logs and stop early. Never rent a second
  GPU or add credits automatically.

## Package and run

`python -m experiments.cloud bundle` creates `artifacts/cloud-upload.tar.gz` from
an explicit file list. It includes code and train/validation data only. It excludes
local credentials, git history, unrelated files and the final-test text.

Extract under `/workspace/llm-rs` on an official Runpod PyTorch image. Use a Python
environment with CUDA-enabled PyTorch 2.8.0; verify the installed version rather than
assuming the template label is correct. Install `experiments/cloud-requirements.txt`
in that environment. Do not change the local training environment.

```sh
# Use the real provisioning timestamp and accepted total hourly quote.
python -m experiments.cloud arm --started-at TIMESTAMP --hourly QUOTE
nohup python -m experiments.cloud watchdog > artifacts/cloud/watchdog.log 2>&1 &
python -m pip install -r experiments/cloud-requirements.txt
nohup env HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME=/workspace/hf-cache \
  HF_HUB_CACHE=/workspace/hf-cache/hub python -u -m experiments.cloud suite \
  > artifacts/cloud/suite.log 2>&1 &
```

Before training, verify `artifacts/cloud/watchdog.json` and the live watchdog process.
The suite refuses an unarmed deadline, missing watchdog, or an existing suite state.
Check `artifacts/cloud/status.json` for progress. Benchmark output is explicitly
marked as a smoke run and uses one validation page; primary results use all 29 pages.
The shared training budget is selected before primary scores and keeps its own cosine schedule.
The official image had PyTorch 2.8.0+cu128 and a working L40S. Its initial download
configuration enabled an unavailable optional `hf_transfer` package. The retry uses
standard downloads and a persistent model cache; the initial setup logs are retained.

Results: `experiments/CLOUD_RESULTS.md`, `experiments/cloud-results.json`, per-page
scores, adapters, checkpoints and timing/memory measurements. Retrieve
`artifacts/cloud-results.tar.gz` before stopping when possible. `/workspace` must be
a persistent volume so the hard stop cannot erase unfinished results.

## Interpretation

Compare character-weighted losses on identical target hashes. Report paired folio
intervals and the difference between intact and shuffled gains. The initial cloud
comparison uses one training seed; repeat seeds require remaining time or another
explicit budget. Keep final-test scoring sealed until all choices are fixed.

Official references: [Runpod pod management](https://docs.runpod.io/pods/manage-pods),
[pod-scoped CLI](https://github.com/runpod/runpodctl),
[1.7B base](https://huggingface.co/Qwen/Qwen3-1.7B-Base),
[8B base](https://huggingface.co/Qwen/Qwen3-8B-Base).
