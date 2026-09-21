# Replication results

Selected layer subset: [0, 1, 26, 27]. Held fixed across all runs.
Five seeds vary initialization and data order. Each run selects its best validation checkpoint within 3,000 updates.
Controls use seed 42 only. The final test set remains sealed.

| Dataset | Seed | Selected update | Qwen bits/character | Best baseline | Baseline bits/character |
|---|---:|---:|---:|---|---:|
| gc | 42 | 3000 | 2.4819 | copy | 2.6985 |
| gc | 43 | 3000 | 2.4857 | copy | 2.6985 |
| gc | 44 | 3000 | 2.4845 | copy | 2.6985 |
| gc | 45 | 3000 | 2.4828 | copy | 2.6985 |
| gc | 46 | 3000 | 2.4818 | copy | 2.6985 |
| gc-shuffle | 42 | 3000 | 2.6202 | copy | 2.8644 |
| timm | 42 | 1500 | 2.0170 | copy | 2.0909 |
| naibbe | 42 | 2500 | 1.8061 | layout | 1.8683 |

Compare Qwen with the baseline within each row; different datasets have different target strings.
Timm and Naibbe are single published samples, not independent generator replications.
These remain validation experiments: checkpoint and layer selection used these pages.
This tests stability of the selected configuration, not a general claim about which layers encode language.
