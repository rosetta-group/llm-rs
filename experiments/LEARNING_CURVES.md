# Qwen learning curves

Local hardware only. Validation pages only. Final test remains sealed.

Fresh adapters; seed 42; context 64; stride 32; rank 8; batch 1; accumulation 2.
Each run has a 3,000-update budget and its own cosine learning-rate schedule starting at 0.0001.
Validation and saving occur every 500 updates. The adapter exported at the run root is the best validation checkpoint.

Copy baseline: **2.6985 bits/character**. Lower is better.

| Run | Status | Update | Bits/character | Gain over copy, 95% interval |
|---|---|---:|---:|---|
| qwen-random-c64-s42-n3000 | complete | 500 | 2.7091 | -0.0105 [-0.0388, +0.0412] |
| qwen-random-c64-s42-n3000 | complete | 1000 | 2.5980 | +0.1006 [+0.0681, +0.1618] |
| qwen-random-c64-s42-n3000 | complete | 1500 | 2.5407 | +0.1578 [+0.1199, +0.2239] |
| qwen-random-c64-s42-n3000 | complete | 2000 | 2.5040 | +0.1945 [+0.1503, +0.2715] |
| qwen-random-c64-s42-n3000 | complete | 2500 | 2.4884 | +0.2101 [+0.1672, +0.2869] |
| qwen-random-c64-s42-n3000 | complete | 3000 **selected** | 2.4828 | +0.2158 [+0.1722, +0.2914] |
| qwen-outer-c64-s42-n3000 | complete | 500 | 2.7262 | -0.0277 [-0.0715, +0.0525] |
| qwen-outer-c64-s42-n3000 | complete | 1000 | 2.5996 | +0.0989 [+0.0572, +0.1765] |
| qwen-outer-c64-s42-n3000 | complete | 1500 | 2.5393 | +0.1592 [+0.1157, +0.2395] |
| qwen-outer-c64-s42-n3000 | complete | 2000 | 2.5052 | +0.1934 [+0.1482, +0.2750] |
| qwen-outer-c64-s42-n3000 | complete | 2500 | 2.4888 | +0.2097 [+0.1632, +0.2941] |
| qwen-outer-c64-s42-n3000 | complete | 3000 **selected** | 2.4819 | +0.2166 [+0.1694, +0.3002] |

Positive gain favors Qwen. Intervals resample 15 validation folio groups, not training seeds.
Selecting the best checkpoint on these same pages makes the intervals exploratory; they do not establish a final-test gain.
The older 400-update pilots used a shorter learning-rate schedule, so their endpoints are separate experiments.

Regenerate this report with `python -m experiments.learning_curves`.
