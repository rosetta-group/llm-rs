# Voynich research plan

Updated: 2026-09-20

Recover and validate Voynich meaning, then express it in English or Italian.
Selective LLM adaptation and prediction comparisons are research methods toward that goal.
See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the current project overview, full experiment
history, live work, and the next meaning-recovery deliverable.

**Translation is the ultimate goal. Prediction is a diagnostic milestone.** There is no accepted English or Italian translation to use as ground truth.

## Terms

- **Residual predictability:** prediction gains beyond local spelling, copying, and layout baselines.
- **Currier A/B:** statistical text varieties, not proven distinct languages.
- **Quire:** a physical group of folded manuscript sheets.
- **Control:** data or a model that tests an alternative explanation for a result.

## Workflow

```text
Repair data preparation and training
Freeze evaluation splits and controls
Compare simple models with selective LLM adaptation
Use the completed matched-context comparison with its overfitting limitation
Build a verified known-plaintext benchmark and measure original-content recovery
Test connections between text and independently annotated images
Test constrained Voynich interpretations on unseen material
Use remaining prediction robustness checks when they resolve a specific uncertainty
```

## 1. Make the pipeline reliable

- Fix the missing comma in `fine_tuning.py`.
- Fix scrambling: training currently reads unchanged `spaced_text`.
- Pass `max_steps` to the trainer; it currently only limits dataset size.
- Preserve folio, line, paragraph, section, scribal hand, and Currier labels.
- Preserve uncertain glyphs and boundaries. Do not merge `.` and `,` silently.
- Keep token IDs through windowing; avoid decoding and tokenizing them again.
- Record data version, preprocessing, model revision, split IDs, and random seeds.

**Deliverable:** a reproducible dataset build and a short training smoke run.

## 2. Define evaluation before training

- Separate training, validation, and final test sets by folio; keep both sides together.
- Use validation to choose settings. Open the final test set only after choices are fixed.
- Add held-out-quire evaluation to test transfer beyond nearby pages.
- Fit learned tokenization and all data-derived rules on training data only.
- Score each target position once, even when context windows overlap.
- Report prediction loss and next-token accuracy, overall and by A/B and section.
- Compare tokenizations using loss per shared underlying transcription unit, not raw token perplexity.
- Repeat the selected comparisons with five seeds. Estimate uncertainty over folios or quires, not individual tokens.

**Deliverable:** fixed split manifests, scoring rules, and control-generation scripts.

## 3. Measure residual predictability

Completed on 2026-09-20. See `experiments/report-completed/REPORT.md` and
`output/pdf/voynich-completed-experiments.pdf` for the graphs and interpretation.
All model selection used validation; final-test scores remain sealed.

- Five local Qwen seeds averaged 2.4833 bits/character (seed SD 0.0017), versus
  copy at 2.6985. Random and outer layers were nearly equal in the initial comparison.
- Qwen also beat each control's baseline: shuffled GC 2.6202 versus 2.8644;
  Timm 2.0170 versus 2.0909; Naibbe 1.8061 versus layout at 1.8683.
  Prediction gains alone do not establish syntax or meaning.
- Three character GRU seeds averaged 2.3321 (SD 0.0012), without pretrained weights.
  The character transformer averaged 2.5656. The GRU trained for more passes and
  used different context and architecture; this is not a causal pretraining comparison.
- GRU control scores: shuffled GC 2.5159; Timm 2.0807; Naibbe 1.7366.
  Qwen wins on the Timm sample. All twelve character runs are complete and verified.
- Cloud base models scored 2.4771 (1.7B) and 2.4689 (8B). The paired gain interval
  includes zero. The GPU is stopped; no further paid comparison is planned.

Twenty-two selected runs were verified. Character outputs use about 62 MiB, with
no pretrained downloads. Temporary Runpod storage awaits deletion approval and
was last quoted at $0.014/hour. The local suite is complete and its progress monitor is paused.

The saved-GRU context ablations are complete: [CONTEXT.md](experiments/CONTEXT.md).
Matched-context retraining is complete: all 18 runs verified. Mean 8-to-128 gains were
0.1227 BPC intact and 0.1366 shuffled. No extra intact gain was established; all models
overfit by the fixed final epoch. See the [final report](experiments/report-matched/REPORT.md)
and [fixed protocol](experiments/MATCHED_PLAN.md).
The first controlled known-plaintext benchmark is complete; see
[the recovery report](experiments/decipherment/REPORT.md). Next, improve boundary recovery
on fresh passages and reduce supplied cipher assumptions. Transcription and quire checks
remain useful diagnostics; finishing every such check is not a prerequisite for that benchmark.
See `experiments/CHARACTER_PLAN.md` and `experiments/CLOUD_PLAN.md` for fixed settings.
Use the random tiny model for pipeline checks only.

| Comparison | Question |
|---|---|
| Frequency and short-context predictors | How much does local symbol structure explain? |
| Copy-and-modify predictor | How much does repetition explain? |
| Predictor using available line and paragraph position | How much does layout explain? |
| Frozen versus adapted LLM | Does adaptation improve prediction? |
| Outer versus middle or random layers | Does the selected layer location matter? |
| Contexts of 16, 64, and 256 tokens | Does longer context add useful information? |

Match trainable parameter counts and training budgets across layer selections. Use only information available at prediction time.

Run the same evaluation on intact text and words shuffled within each line. Keep vocabulary, line membership, and splits fixed. Include published copy-based generators and Naibbe ciphertext as additional controls.

**Success:** lower held-out loss than the strongest baseline, with uncertainty supporting a gain, stable across seeds and reasonable preprocessing choices. Test whether longer-context gains exceed those seen on controls.

**Interpretation:** a gain identifies structure to explain. It does not prove language or translation. Failure does not prove meaninglessness.

## 4. Check transcription and boundary assumptions

- Compare the current transcription with an independent one on matched pages.
- Compare transcription characters, composite symbols, and learned symbol groups.
- Test uncertain boundaries as separate, merged, and explicitly marked variants.
- Learn each representation on training data only.

**Success:** the main model comparison survives these choices. Otherwise, report which choice creates the effect.

## 5. Seek independent evidence of meaning

Test whether text predicts observable image properties within the same section and scribal hand. For example, use independently annotated leaf counts or diagram shapes.

- Control for text length and layout.
- Shuffle image-text pairings within matched groups.
- Hold out pages from annotation-driven model selection.
- Treat plant identities and medical uses as hypotheses, not labels known to be correct.

**Success:** reproducible text-image association beyond the matched controls. This supports a limited association, not a translated sentence.

## 6. Test a route to English or Italian

The first verified recovery benchmark is complete: six Dante passages, two hidden keys,
and three supplied cipher conditions. Spaced substitution recovered all normalized words;
restricted Naibbe recovered 99.44% of letters with its structural codebook supplied.
Space-free word recovery remains poor (60–62% error after a secondary diagnostic).
See [methods, limitations, and graphs](experiments/decipherment/REPORT.md).

Improve segmentation using independent development material, then score fresh passages.
Keep evaluation originals hidden from method selection and declare every supplied hint.
Gradually reduce known cipher structure. This remains a method check, not a Voynich reading.

Compare input-layer adaptation with adaptation of both early and late layers. Use TransformerLens for targeted activation interventions. Add SAEs only for a specific question that simpler interventions cannot answer.

Apply candidate Voynich mappings consistently across unseen passages. Test repeated forms and independent image associations. Record competing interpretations and contradictions. Fluent output, feature names, and round-trip consistency are insufficient evidence on their own.

**Success:** a fixed interpretation makes independent predictions that alternatives fail. English or Italian is the output language; neither is assumed to be the manuscript's source language.

## Deliverables and progress

1. Dataset audit, fixed splits, and working controls.
2. Baseline and selective-adaptation results table.
3. Context, transcription, and boundary robustness report.
4. Controlled content recovery alongside prediction studies, then independent Voynich grounding.

After each milestone, report: completed work, measured results, blockers, and the next experiment. During active work, give brief updates when findings or decisions change. Do not start long or paid training runs without an agreed compute budget.

## Implementation status

| Work | Status |
|---|---|
| IVTFF parsing, metadata, uncertain readings, grouped splits | Implemented and tested |
| Raw token windows, target masks, real optimizer-step limits | Implemented; character and learned-BPE smoke runs passed |
| Frequency, n-gram, layout, copy baselines | Measured on validation |
| Shuffled lines, boundaries, EVA transcription, quire holdout | Baseline comparisons measured |
| Published Naibbe and Timm–Schinner controls | Pinned samples imported and baseline-scored |
| Local Qwen adapters and layer controls | Implemented; initial seed-42 experiments recorded in `experiments/RESULTS.md` |
| Longer primary runs | Both 3,000-update runs complete; outer 2.4819 and random 2.4828 validation BPC |
| Five training seeds and neural controls | Complete: five intact seeds and three controls; `experiments/REPLICATIONS.md` |
| Larger base-model comparison | Complete: 8B gain over 1.7B is 0.0083 BPC, with folio interval crossing zero; GPU stopped |
| Character models from scratch | Twelve runs complete: GRU mean 2.332 BPC; training exposure differs from Qwen |
| Exact context ablation | Thirty evaluations complete: 8–128 characters, fixed GRU weights, stride 1; `experiments/CONTEXT.md` |
| Training matched to short histories | Complete: 18 verified runs; no established extra intact context gain, with later overfitting; `experiments/report-matched/REPORT.md` |
| Historical/modern language statistics | Complete: eleven pinned corpus samples, two Voynich transcriptions, adjacency controls; `experiments/language-comparison/REPORT.md` |
| Known-plaintext recovery | First controlled benchmark complete; spaced substitution exact, space-free word recovery still poor; `experiments/decipherment/REPORT.md` |
| Text-image association | Proposed independent grounding experiment; not implemented yet |
| SAEs, translation claims, and Linear B | Deferred until earlier evidence supports the next experiment |

`README.md` contains runnable commands. `experiments/results.json` preserves the initial numerical results.
Voynich model scores use validation; corpus statistics use training pages. Controlled
recovery uses separate hidden source passages. Voynich final test scoring remains sealed.

## Key reading

- [Gaskell & Bowern, 2022: Gibberish after all?](https://ceur-ws.org/Vol-3313/paper4.pdf) Meaningless text can reproduce language-like statistics.
- [Bowern & Gaskell, 2022: Enciphered after all?](https://ceur-ws.org/Vol-3313/paper6.pdf) Encoding can make meaningful text unusually predictable.
- [Steckley & Steckley, 2024: Subtle Signs of Scribal Intent](https://dspace.ut.ee/items/f57c5030-067e-4872-942f-f9c153e49747) Layout affects token distributions.
- [Greshko, 2025: Naibbe cipher and code](https://github.com/greshko/naibbe-cipher) Reversible Latin/Italian ciphertext provides a relevant control.
- [Parisel, 2026: Currier distinction](https://arxiv.org/html/2604.25979v2) Preprint; motivates separate A/B evaluation.
- [Parisel, 2026: Positional and directional constraints](https://arxiv.org/html/2604.19762v2) Preprint; tests combinations of structural properties against generators.
- [Rozanova & Temerev, 2026: A Glyph Is Not a Letter…](https://arxiv.org/html/2608.17096v1) Preprint; challenges assumed symbol and boundary units.

These sources motivate experiments. Their findings have not been independently reproduced in this repo.
