# Matched-context results

Completed: 18/18 runs. Primary score: epoch 20, exact context at every target.

[Protocol](MATCHED_PLAN.md) · [Project goal and research record](../RESEARCH_LOG.md)

| Data | Seed | Context | Final BPC | Best validation BPC (secondary) | Training characters |
|---|---:|---:|---:|---:|---:|
| gc | 42 | 8 | 2.6542 | 2.4689 | 2,511,320 |
| gc-shuffle | 42 | 8 | 2.8628 | 2.6369 | 2,511,320 |
| gc | 42 | 32 | 2.6153 | 2.3797 | 2,511,320 |
| gc-shuffle | 42 | 32 | 2.8286 | 2.5493 | 2,511,320 |
| gc | 42 | 128 | 2.5347 | 2.3277 | 2,511,320 |
| gc-shuffle | 42 | 128 | 2.7084 | 2.4988 | 2,511,320 |
| gc | 43 | 8 | 2.6626 | 2.4655 | 2,511,320 |
| gc-shuffle | 43 | 8 | 2.8437 | 2.6336 | 2,511,320 |
| gc | 43 | 32 | 2.6232 | 2.3814 | 2,511,320 |
| gc-shuffle | 43 | 32 | 2.8252 | 2.5599 | 2,511,320 |
| gc | 43 | 128 | 2.5469 | 2.3365 | 2,511,320 |
| gc-shuffle | 43 | 128 | 2.7245 | 2.5123 | 2,511,320 |
| gc | 44 | 8 | 2.6664 | 2.4612 | 2,511,320 |
| gc-shuffle | 44 | 8 | 2.8541 | 2.6288 | 2,511,320 |
| gc | 44 | 32 | 2.6248 | 2.3760 | 2,511,320 |
| gc-shuffle | 44 | 32 | 2.8263 | 2.5521 | 2,511,320 |
| gc | 44 | 128 | 2.5334 | 2.3309 | 2,511,320 |
| gc-shuffle | 44 | 128 | 2.7179 | 2.5019 | 2,511,320 |

## Does longer history help intact text more?

Positive extra gain favors intact Voynich. Each gain compares identical targets within its own text. The interaction resamples corresponding original/shuffled folios together, without pretending their strings are identical.

| Seed | Extra intact gain, 8 to 128 (BPC) | 95% folio interval |
|---:|---:|---|
| 42 | -0.0350 | [-0.0501, -0.0212] |
| 43 | -0.0035 | [-0.0256, 0.0199] |
| 44 | -0.0033 | [-0.0327, 0.0258] |

These are validation results, conditional on the fixed optimization budget. Twenty passes match exposure, not compute or convergence. Best-epoch scores are secondary and can have unequal exposure. A positive interaction identifies an effect of preserving order under these controls, not a translation. A null result does not imply meaningless text. Final-test pages remain sealed.
