# Pre-fit sweep-limit correction

Declared 2026-09-25 before any development control was fitted. The initial follow-up
freeze (`freeze.json`, committed at `cc1cc92`) is preserved but was not executed.
Preflight inspection found that the inherited settings request **30 random kicks**.
A kick usually needs one improving sweep followed by a sweep that confirms no further
improvement; a 50-sweep ceiling can stop before that schedule finishes. No new control
outcome was consulted to make this correction.

Use **200 sweeps**, retaining the **20 million proposal evaluations**, **1,200-second
refinement allowance**, **3,600-second fit allowance**, **12 aggregate fit-worker
hours**, five workers and two threads. All algorithms, priors, generator seeds,
copying inputs, comparison thresholds and declared benchmark results stay unchanged.
The full-score equivalence tests and numerical checks also stay unchanged.

Driver: `experiments.rejection_followups_v2`. Its `freeze-v2.json` is committed before
fitting. The first freeze remains a reproducible preflight record, not an experiment
with missing or selectively discarded results. Results are development evidence only.
