# Character-model results

Local models trained from random weights. Lower BPC is better.

| Model | Data | Seed | Parameters | Selected epoch | BPC | Epoch 2 BPC | Qwen BPC |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer | gc | 42 | 1,272,448 | 17 | 2.5592 | 2.7946 | 2.4819 |
| gru | gc | 42 | 1,316,608 | 16 | 2.3309 | 2.5087 | 2.4819 |
| transformer | gc | 43 | 1,272,448 | 14 | 2.5744 | 2.7972 | 2.4857 |
| gru | gc | 43 | 1,316,608 | 14 | 2.3332 | 2.5277 | 2.4857 |
| transformer | gc | 44 | 1,272,448 | 17 | 2.5634 | 2.7878 | 2.4845 |
| gru | gc | 44 | 1,316,608 | 17 | 2.3322 | 2.5063 | 2.4845 |
| transformer | gc-shuffle | 42 | 1,272,448 | 17 | 2.7060 | 2.8400 | 2.6202 |
| gru | gc-shuffle | 42 | 1,316,608 | 15 | 2.5159 | 2.6774 | 2.6202 |
| transformer | timm | 42 | 1,272,448 | 9 | 2.2863 | 2.4775 | 2.0170 |
| gru | timm | 42 | 1,316,608 | 8 | 2.0807 | 2.1917 | 2.0170 |
| transformer | naibbe | 42 | 1,272,448 | 15 | 1.8975 | 2.2488 | 1.8061 |
| gru | naibbe | 42 | 1,316,608 | 20 | 1.7366 | 1.8859 | 1.8061 |

Exact scored characters match the references. Training exposure, context, architecture,
and full-model versus adapter training differ. These are practical baselines, not a causal
estimate of pretraining's effect. Epoch 2 is an early reference, not exact Qwen exposure matching.
Each model selects its best validation epoch; uncertainty is exploratory. Final test remains sealed.
Timm and Naibbe are single synthetic samples, not independent generator replications.
