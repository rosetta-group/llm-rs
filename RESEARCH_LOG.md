# Recovering meaning from the Voynich manuscript

Updated: 2026-09-21. This is the current project overview and research record.
Older reports preserve earlier experiments and may contain superseded next steps.

## Overall goal

Recover and validate the meaning of Voynich text, then express it in English or Italian.
Prediction, linguistic statistics, and model interpretation are tools toward that goal.
They are not the final product, and a better prediction score is not a translation.
Linear B remains a later target after the meaning-recovery method has been validated.

**Meaning recovery:** an interpretation of the source that can be checked against independent evidence.
**BPC:** bits per normalized transcription character; lower means better prediction.
**Validation:** pages used to develop and compare methods.
**Final test:** reserved pages that have not been scored or used to choose models.

The initial hypothesis was to adapt an LLM's language-dependent layers while preserving
its higher-level knowledge. We have not established a clean separation between those
functions or shown that selective adaptation recovers meaning. TransformerLens and sparse
autoencoders remain possible diagnostic tools, not sources of verified word meanings.

## The route to translation

```text
Build reliable text preparation and evaluation
Use prediction experiments to identify useful models and unresolved assumptions
Record the completed matched-context comparison and its limits
Run controlled recovery with verified ciphertext and hidden original passages
Improve word recovery on fresh passages, then reduce supplied cipher assumptions
Test which parts of the method can work without paired Voynich plaintext
Ground Voynich interpretations in independent evidence
Render supported meanings in English or Italian, with uncertainty and alternatives
```

We do not know an accepted Voynich passage in English or Italian to use as a label.
There may be multiple interpretations compatible with the same text statistics.
The project therefore needs evidence outside next-character prediction: verified
recovery on known examples, consistent mappings on unseen passages, and independent
text/image associations where suitable annotations exist.

Success is not a plausible English paragraph. A proposed reading must use consistent
rules, account for repeated forms, and make checkable predictions beyond the material
used to invent it. We must also record contradictions and competing readings.

## What we are doing now

**Completed: matched-context training, all 18 local runs verified.** Each model trained
and evaluated with 8, 32, or 128 preceding characters. We compared intact and within-line shuffled
Voynich, with three training seeds. Model size, initial weights within a seed, optimizer,
updates, and scored-character exposure were matched. Primary results use the fixed final
epoch, not whichever checkpoint looks best on validation.

- Question: does extra history help intact text more than its shuffled control after
  the model learns to operate at that history length?
- Why: the previous memory test changed history only at evaluation; the Naibbe curve
  showed that shortening history can disrupt a model's learned operating conditions.
- Budget: local Apple GPU only; no model downloads or new cloud rentals. Twenty passes
  per run; two-hour run cap and twelve-hour suite cap. Preserve 20 GiB free disk.
- State and results: [fixed protocol](experiments/MATCHED_PLAN.md),
  [completed-run table](experiments/MATCHED.md),
  [machine-readable results](experiments/matched-results.json), and
  [illustrated final report](experiments/report-matched/REPORT.md).
- Runtime details: `artifacts/matched/status.json`, `artifacts/matched.log`, and
  `training_run_outputs/matched-*`. Failed or incomplete runs stay labeled as such.

This comparison tested whether extra history benefits intact text more than its shuffled control. It cannot produce a word
translation or prove the manuscript contains language. Completing every possible
linguistic-statistics experiment is not a prerequisite for the known-plaintext benchmark.

Early diagnostic: the first intact 8-character run reached 2.4689 validation BPC at
epoch 6 but finished at 2.6542 at epoch 20. The training loss kept falling. This is
evidence of overfitting in that run, not evidence against meaning. We retain the fixed
final-epoch comparison and show the full curves and secondary best-epoch scores.

Completion: **18/18 runs verified**. Each model completed 20 passes, 9,820 updates,
and 2,511,320 scored training targets. Checkpoint reload errors were zero. Frozen input
hashes, saved scores, and equal exposure were checked; the suite released its training lock.
The suite took 4.92 hours, produced about 276 MiB of model/run outputs, and finished
within its original deadline with 56.8 GiB free. No new cloud compute or models were used.

| Seed | Intact gain (BPC) | Shuffled gain (BPC) | Extra intact gain | 95% source-folio interval |
|---|---:|---:|---:|---|
| 42 | 0.1195 | 0.1544 | -0.0350 | [-0.0501, -0.0212] |
| 43 | 0.1157 | 0.1192 | -0.0035 | [-0.0256, +0.0199] |
| 44 | 0.1330 | 0.1362 | -0.0033 | [-0.0327, +0.0258] |

**Finding:** longer history helped both versions, without an established extra benefit
for intact Voynich. Mean gains were 0.1227 BPC intact and 0.1366 shuffled. One seed
favored shuffled text; two were inconclusive. The mean interaction was −0.0139 BPC.

**Limit:** all 18 models peaked on validation between passes 5 and 9 and deteriorated
by pass 20. Equal exposure does not equalize convergence or overfitting. These results
do not measure semantic content or imply that the manuscript is meaningless. The folio
intervals do not include every source of training/design uncertainty. Final-test pages remain sealed.

The prediction suite is finished and its monitor is paused. No follow-up experiment
was started. The proposed next meaning milestone remains fresh-passage word recovery
with fewer supplied cipher assumptions, described below.

A reporting error stopped the suite after run five: bootstrap and training seeds used
the same dictionary field. The report now records both separately. A regression test
passes; the original runner and manifest are archived in `artifacts/matched/repairs/`.
A recorded manifest amendment permits this reporting/resume repair. The five completed
models were verified and retained; training resumed under the original suite deadline,
with unchanged training code, data, settings, and exposure.

**Completed alongside training: language statistics and blind decipherment.** Both used
local CPU work. The statistics compare 148 Voynich training pages against eleven pinned
historical/modern corpus samples. The recovery benchmark tests six hidden Dante passages
under two random keys and three supplied cipher conditions. Neither uses Voynich test pages.
See the [statistics report](experiments/language-comparison/REPORT.md) and
[decipherment report](experiments/decipherment/REPORT.md), with five figures in total.

## What we tried and found

Voynich model scores below are validation results. New corpus statistics use training
text; controlled decipherment has separate hidden original passages. Different texts, transcriptions, and evaluation
window settings are not interchangeable. Follow the linked reports for exact configurations,
page scores, uncertainty, and limitations.

| Experiment | Result | What it tells us | Record |
|---|---|---|---|
| Repair the original pipeline | Fixed data scrambling, optimizer-step limits, target masks, grouped splits, and saved-checkpoint verification | Later model comparisons use auditable data and scoring | [Implementation plan](RESEARCH_PLAN.md), [tests](tests/) |
| Frequency, spelling, layout, and copying | GC frequency 4.2450 BPC; best copy baseline 2.6985; copy character accuracy 44.93% | Local regularity explains substantial predictability | [Initial results](experiments/RESULTS.md) |
| Boundaries, transcription, and quire holdout | Copy: merged GC 2.6499; separated GC 2.6441; quire split 2.7815; ZL 2.0691 | Representation and split matter; these different target strings cannot be directly ranked | [Baseline matrix](experiments/RESULTS.md) |
| Short Qwen layer pilots, 400 updates | Frozen 4.3505; outer 2.9373; middle 3.1549; random 2.8783 | Adaptation helped, but all pilots lost to copying; outer layers were not uniquely effective | [Pilot settings and scores](experiments/RESULTS.md) |
| Longer Qwen training, 3,000 updates | Outer 2.4819; random 2.4828 | More training beat copying; no clear outer-versus-random layer advantage | [Learning curves](experiments/LEARNING_CURVES.md) |
| Five Qwen training seeds | Mean 2.4833 BPC; sample SD 0.0017 | The selected configuration's prediction gain was stable across these seeds | [Replications](experiments/REPLICATIONS.md) |
| Qwen on shuffled and synthetic controls | Shuffled 2.6202 vs copy 2.8644; Timm 2.0170 vs copy 2.0909; Naibbe 1.8061 vs layout 1.8683 | Beating simple baselines also happens on controls; it is not a meaning detector | [Control results](experiments/REPLICATIONS.md) |
| Cloud Qwen 1.7B-Base versus 8B-Base | 2.4771 vs 2.4689; gain 0.0083, folio interval [-0.0016, 0.0175] | This one-seed comparison did not establish a reliable larger-model gain; the gated cloud follow-up was not run | [Cloud results](experiments/CLOUD_RESULTS.md) |
| Small character models trained from scratch | Three-seed GRU mean 2.3321; character transformer mean 2.5656 | A small model can outperform these Qwen runs on prediction; training exposure and architecture differ, so this does not isolate pretraining's effect | [Twelve character runs](experiments/CHARACTERS.md) |
| Character models on controls | GRU: shuffled 2.5159, Timm 2.0807, Naibbe 1.7366; Qwen remains better on Timm | The model ranking varies by control | [Character control results](experiments/CHARACTERS.md) |
| Saved-GRU memory test, 30 evaluations | Intact mean: 2.5482 BPC at 8 characters, 2.34898 at 64, 2.32857 at 128; 64 captures 90.7% of the measured 8-to-128 gain | Extra history helps; shuffled text benefits almost as much. Naibbe worsens at intermediate histories, exposing a training-mismatch question | [Context report and graphs](experiments/CONTEXT.md) |
| Learned symbol groups | Training-only BPE and tiny-model plumbing checks passed | Infrastructure exists; no pretrained-tokenizer replacement or semantic result was claimed | [Tokenizer usage](README.md#learned-symbol-groups) |
| Matched-context retraining | All 18 runs verified; mean 8-to-128 gain 0.1227 intact vs 0.1366 shuffled | No established extra intact gain; all runs overfit by the fixed final epoch | [Final report](experiments/report-matched/REPORT.md) |
| Historical/modern corpus statistics | v101: 25,723 forms, mean length 3.865, adjacent-length correlation +0.218; EVA: mean 4.975, correlation +0.169 | Length clustering survives transcription and boundary checks; genre/layout confounds prevent language identification | [Statistics and graphs](experiments/language-comparison/REPORT.md) |
| Blind substitution recovery | 100% normalized letters and words with spaces; 100% letters without spaces | Hidden-key recovery works with known Italian and a supplied simple cipher family | [Recovery benchmark](experiments/decipherment/REPORT.md) |
| Restricted Naibbe recovery | 99.44% letters with structural codebook supplied | A positive control under strong declared assistance, not unknown-cipher discovery | [Recovery benchmark](experiments/decipherment/REPORT.md) |
| Word-boundary failure and secondary repair | Primary space-free word error 99.32%; independently calibrated penalty reduces it to 60.33% for substitution and 61.53% for Naibbe | Correct letters do not guarantee correct words; secondary reuse is exploratory | [Boundary diagnostic](experiments/decipherment/boundary-diagnostic.json) |

The frozen [completed-experiments report](experiments/report-completed/REPORT.md) and
[PDF](output/pdf/voynich-completed-experiments.pdf) cover the Qwen, cloud, and character
experiments. The [earlier report](experiments/report/REPORT.md) is an interim snapshot.
The exact-context test used stride 1; earlier GRU reports used stride 64. That change
explains why its 128-character reference differs from the earlier 2.3321 mean.

## What we know, and what remains open

1. **Predictive structure exists in these transcriptions.** Both simple copying and
   learned models predict better than character frequency. This does not identify
   a source language, cipher, word boundary, or meaning.
2. **Larger language models are not yet the bottleneck we have demonstrated.** The
   8B comparison gave an uncertain gain, while a small GRU predicted better under
   a different training budget. We have no evidence that another large model will translate Voynich.
3. **Our initial layer hypothesis is unconfirmed.** Outer and random layer adaptations
   were nearly equal after longer training. No activation intervention or SAE study
   has established a language-only subset of layers or a Voynich concept mapping.
4. **Controlled content recovery now works under declared assistance.** The spaced
   substitution benchmark recovered its normalized Italian originals exactly. Known
   language and cipher structure are substantial hints; word segmentation remains poor.
5. **Voynich meaning remains unvalidated.** No verified Voynich word, passage translation,
   or controlled image association has been produced. Corpus resemblance is not a language label.

The validation set contains 29 pages from 15 folio groups, with an uneven A/B mix.
The GC representation is v101; the independent ZL transcription is EVA. Timm and Naibbe
are each one published sample, split into chronological blocks, not independent
generator replications. Baseline robustness was measured; the corresponding full
neural transcription/quire robustness matrix has not been run.

## Next meaning milestone: reliable recovery with fewer hints

**Known plaintext:** original text whose relationship to the ciphertext is independently verified.
The first controlled benchmark is complete: the solver used unrelated modern Italian
frequencies and received no paired original passages or true letter keys. The Naibbe
condition supplied its structural codebook. Predictions were frozen before grading.
The original boundary failure and the secondary calibration are both preserved.

```text
Improve boundary recovery using independent development material
Freeze the method and score fresh, unseen original passages
Check original content, not just letter accuracy or fluent output
Reduce supplied Naibbe structure and test ambiguity explicitly
Test which recoverable assumptions are defensible for Voynich
```

Do not tune further against the six disclosed Dante passages and call that blind progress.
A new recovery result needs fresh evaluation passages. A useful target is whole-word and
passage recovery with source spaces removed. English or Italian rendering must preserve
recovered content; it must not hide a wrong decoding. These are proposed next experiments,
not additional training or cloud jobs launched by this study.

For Voynich, independent anchors could include repeatable text/image relationships
tested within section and hand, or mappings that predict unseen passages consistently.
Plant identities, medical uses, and source-language guesses remain hypotheses rather
than training labels known to be correct. We will use TransformerLens or SAEs only
when a concrete intervention can test a specific mapping or mechanism.

## Evidence, resources, and maintenance

- [README](README.md): setup and commands. [Research plan](RESEARCH_PLAN.md): methods and evidence requirements.
- `experiments/*-plan.json`: fixed machine-readable settings; `experiments/*-results.json`: numerical results.
- [Language sources](experiments/language-sources.json) and [Naibbe sources](experiments/decipherment-sources.json): immutable revisions, hashes, attribution, and licenses; about 210 MiB downloaded.
- `experiments/splits/`: fixed page groups. `experiments/sources.json`: pinned downloads and hashes.
- `artifacts/data/`: prepared text and metadata. `artifacts/results/`: baseline page scores.
- `training_run_outputs/`: model configs, training curves, weights, scores, and run status. These large files are git-ignored.
- [Literature and source links](RESEARCH_PLAN.md#key-reading): motivation, not independently reproduced findings.
- The old Runpod GPU was stopped and its results downloaded and verified. The last
  recorded balance was $9.18 from $10; this is not a current balance. Temporary stopped
  storage was last quoted at $0.014/hour and its deletion is still pending. No new cloud run is authorized by this local suite.
- The matched-context suite is complete and its progress monitor is paused.
  [Final verification](experiments/report-matched/verification.json) records checkpoint, exposure, hash, and resource checks.

After each experiment, add its question, fixed settings, result, limitations, and
implication for meaning recovery here. Preserve old reports and failed attempts.
Distinguish planned, running, completed, and verified work. Do not turn a prediction
milestone into a translation claim, or keep expanding the statistics track indefinitely.
