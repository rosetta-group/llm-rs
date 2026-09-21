# Cloud scaling results

Validation only. Positive paired differences favor the larger model.

| Model | Dataset | Updates | Selected | BPC |
|---|---|---:|---:|---:|
| Qwen/Qwen3-1.7B-Base | gc | 3000 | 3000 | 2.4771 |
| Qwen/Qwen3-8B-Base | gc | 3000 | 3000 | 2.4689 |

gc: larger-model gain 0.0083 bits/character; paired 95% folio interval [-0.0016, 0.0175].

Adapters use 1,245,184 parameters (1.7B) and 1,212,416 (8B).
Single seed, validation-selected checkpoints: exploratory evidence, not translation.

The interval includes zero. This run does not establish a reliable gain from the
larger model. The shuffled-text cloud follow-up did not meet its predeclared gate.
Both models beat the copy baseline (2.6985 bits/character) on these validation pages.
The final test remains sealed.

The GPU was stopped after archive and checkpoint verification. Runpod showed
$9.18 remaining from $10; temporary stopped storage still costs $0.014/hour.
Local repeat-seed and synthetic-control runs continue independently.

Next: finish those controls, strengthen the character baseline, and compare context
lengths on identical scored positions before paying for a larger model again.
