# Voynich research plan

> **How to read this file.** Research gates and the status of each work item, kept since the
> project began; numbered sections 1–6 are the original plan with status notes added. The
> current state is summarised in [docs/OVERVIEW.md](docs/OVERVIEW.md) and
> [docs/RESULTS.md](docs/RESULTS.md); conventions for new rounds are in [docs/CONVENTIONS.md](docs/CONVENTIONS.md).

Updated: 2026-09-25

**Historical-language coverage pilot completed, 2026-09-25.** The
[three-pair pilot](experiments/language-coverage/REPORT.md) adds Catalan and broadens
Latin/German. The expanded set ranks the correct language first on both passages in
3/3 cases, but accepts only Latin (1/3); all three true-language omissions reject.
German transfer CER improves 45.69% → 12.48%, Latin 12.37% → 6.69%; Catalan is 9.27%.
Twenty-four fits used 1.386 worker-hours, with no caps. The feasibility target failed.
Post-run correct-plaintext scores support key-recovery development on these released
cases before more language additions or a larger confirmation. Keep the current
thresholds; confirmation needs new source groups, including another Catalan work.
The existing manuscript gate is unchanged. This is a small coverage pilot, not the
90-block rejection confirmation below.

**Voynich priority, 2026-09-25: prepare a costed fresh rejection study.** The
[three authorized development follow-ups](experiments/rejection-followups/REPORT.md)
are complete. The transfer-centered candidate accepts 3/3 released genuine ciphers
instead of 2/3; three exact-frequency copying controls are rejected without caps
at 98.47–98.64% coverage. Incremental swap scoring is 31.1× faster on the released
copying fixture with the same winning swap and score. These are development results
from three shared source/key blocks, not a population error-rate estimate.

The next [confirmation proposal](experiments/rejection-followups/CONFIRMATION_PLAN.md)
requires independent fresh sources/keys, all five languages, multiple generators and
predeclared statistical gates. Its 1,800 fits project to about 123 worker-hours at
current copying-control cost; that is not a guaranteed runtime or an agreed run
budget. Audit source availability and work limits on every released input class
before freezing any new study. The 20-million-proposal limit is especially restrictive
on large mutation inventories. No fresh confirmation or manuscript run was started.

The [original screen](experiments/rejection-transfer-v2/REPORT.md) remains inconclusive
on caps, with 2/3 positives accepted and one copying control inconclusive. Its
thresholds and outcomes were not rewritten. The translation gate and reserved
manuscript test remain unchanged; the [first capped attempt](experiments/rejection-transfer/REPORT.md)
also stays separate.

**Linear A track closed (2026-09-23).** Five rounds on branch `linear-a` tested whether these
methods can tell which language Linear A is. Lexical tests found Greek in at most 10% of Linear B
control samples against a 90% gate; round five's Hittite profile match also holds for shuffled
syllables. No language was identified or ruled out. See [docs/LINEAR_A.md](docs/LINEAR_A.md).

**Next candidate: Rongorongo.** Its language (Old Rapa Nui) is known and its corpus (about
15,000 glyphs) is above the Naibbe recovery threshold, so a known-answer control can be built.
Ranking of other scripts: [docs/UNDECIPHERED.md](docs/UNDECIPHERED.md).

**Latest scope change:** the user requested raw folio images and reproducible image
descriptions. The [image archive and Pixel Layout v1](data/folios/README.md) provide
213 scans, 228 panel descriptions, reviewed foldout crops and versioned `analysis_id`
records. This reopens acquisition/description, while confirmatory text/image modelling
remains parked. A later method should describe objects and relations with explicit
evidence regions; it still needs independent masked evaluation before meaning claims.

**Current deliverable: a validated decipherment-method benchmark with positive controls
and honest Voynich negatives.** Translation is a long-term motivation, not a reachable
claim on current evidence. See [the current CPU plan](experiments/METHOD_BENCHMARK_PLAN.md),
[benchmark results](experiments/method-benchmark/REPORT.md), and [research record](RESEARCH_LOG.md).

**Prediction track formally closed.** No more broad BPC experiments, model scaling, or
longer training. Any exception needs a named mechanism, falsifiable contrast, fixed
budget, and a statement of which interpretation the result could reject. Unfinished
robustness checks below are archived possibilities, not a queue to execute.

## Terms

- **Residual predictability:** prediction gains beyond local spelling, copying, and layout baselines.
- **Currier A/B:** statistical text varieties, not proven distinct languages.
- **Quire:** a physical group of folded manuscript sheets.
- **Control:** data or a model that tests an alternative explanation for a result.

## Workflow

```text
Archive completed prediction experiments
Tune segmentation on non-Dante development text
Freeze, then grade fresh hidden passages
Remove cipher-family and codebook hints; report positive controls and failures
Test existing visual annotations with held-out folios and layout controls
Publish the validated benchmark and its limits; require new evidence before translation
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

## 3. Measure residual predictability — closed

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
  includes zero. The pod and attached storage are deleted; no further paid comparison is planned.

Twenty-two selected runs were verified. Character outputs use about 62 MiB, with
no pretrained downloads. Runpod pod and attached non-network volumes were deleted on 2026-09-21;
the network-volume inventory is empty. See `experiments/cloud-cleanup.json`. The local suite is complete and its progress monitor is paused.

The saved-GRU context ablations are complete: [CONTEXT.md](experiments/CONTEXT.md).
Matched-context retraining is complete: all 18 runs verified. Mean 8-to-128 gains were
0.1227 BPC intact and 0.1366 shuffled. No extra intact gain was established; all models
overfit by the fixed final epoch. See the [final report](experiments/report-matched/REPORT.md)
and [fixed protocol](experiments/MATCHED_PLAN.md).
The first controlled known-plaintext benchmark is complete; see
[the recovery report](experiments/decipherment/REPORT.md). Fresh-passage lexicon segmentation and codebook-free recovery now have separate frozen
protocols in `experiments/segmentation/` and `experiments/codebook-free/`. Transcription
and quire prediction checks are deferred unless they test a specific mechanism.
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

## 4. Check transcription and boundary assumptions — conditional only

- Compare the current transcription with an independent one on matched pages.
- Compare transcription characters, composite symbols, and learned symbol groups.
- Test uncertain boundaries as separate, merged, and explicitly marked variants.
- Learn each representation on training data only.

**Success:** the main model comparison survives these choices. Otherwise, report which choice creates the effect.

## 5. Seek independent evidence of meaning

**Started and measured:** the Grove/Stolfi pharmaceutical-label pilot uses explicit light/dark
root descriptions, not guessed plant identities. With 59 examples across six training
folios, text improves balanced accuracy by 1.45 points over length/layout controls;
p=0.348, so no association is established. See [frozen protocol](experiments/association/PROTOCOL.md)
and [report](experiments/association/REPORT.md). The larger herbal-page study still needs
independent visual annotations; missing part mentions cannot serve as negative labels.


Test whether text predicts observable image properties within the same section and scribal hand. For example, use independently annotated leaf counts or diagram shapes.

- Control for text length and layout.
- Shuffle image-text pairings within matched groups.
- Hold out pages from annotation-driven model selection.
- Treat plant identities and medical uses as hypotheses, not labels known to be correct.

**Success:** reproducible text-image association beyond the matched controls. This supports a limited association, not a translated sentence.

**Complex extension completed:** [joint and nonlinear associations](experiments/association-complex/REPORT.md)
adds 35 descriptors and pairwise co-mentions on 123 plants, same-page description retrieval,
and relational alignment. Three model classes × four endpoints yield no reliable gain
under whole-folio holdout and 999 within-page permutations; all 12 Holm p-values are 1.0.
A synthetic interaction control passes. This is exploratory reuse of existing descriptions,
not direct pixel modelling or independently blinded annotation. The next richer data need
is standardized morphology/spatial-relation annotation with writing hidden and fresh folios.

**Broad image domains completed:** [domain-level study](experiments/image-domains/REPORT.md)
compares botanical, people/bathing, and celestial/diagram categories on 63 training folio
groups. Character text alone scores 72.22% balanced accuracy; hand/layout controls score
94.44%; adding text does not improve them. People/bathing occupies one quire; other
conditional domain tests are unidentifiable because hand and domain are aligned. Existing
illustration metadata supplies dominant domains, not independent object-presence labels.
Next, link nearby text to independently annotated objects within matched domains, including
people/stars in zodiac diagrams and people/pipes/pools in biological scenes.

## 6. Test a route to English or Italian

The first verified recovery benchmark is complete: six Dante passages, two hidden keys,
and three supplied cipher conditions. Spaced substitution recovered all normalized words;
restricted Naibbe recovered 99.44% of letters with its structural codebook supplied.
Space-free word recovery remains poor (60–62% error after a secondary diagnostic).
See [methods, limitations, and graphs](experiments/decipherment/REPORT.md).

The new lexicon benchmark was tuned only on non-Dante Italian and scored 24 fresh passages.
Modern word error is 6.1%; historical word error remains 39.9%. The improvement gate passed;
the declared <10% error gate on both corpora failed. See [segmentation report](experiments/segmentation/REPORT.md).
The codebook-free benchmark supplied only ciphertext and a known Italian prior. Its first
decoder failed; the standard one-letter comparators recovered substitution and homophonic
controls but cannot express Naibbe's units. A joint segmentation-and-decipherment EM now
recovers Naibbe to about 12% (modern) and 15% (historical) character error on development
passages of 5,200–10,400 letters, and nothing recovers it below about 2,600 letters. The gate
is not met. Round one on four sealed 5,200-letter passages gave 12.5% CER modern and 33.5% Dante.
Round two, with usage pruning of the piece lexicon and a prior including Petrarca's verse, gave
9.5% modern and 10.3% Dante on four new sealed passages. Round three added a word-level polish
and gave 8.8% modern and 10.5% Dante. Round four repaired the candidate piece lexicon from the
ciphertext (an oracle with the true lexicon reaches 0.5%) and gave **5.7% Dante** on four new
sealed passages; word error 45%; the gate still fails. Modern test text is exhausted; a modern
half needs a newly pinned corpus. See the [round-four report](experiments/joint-recovery-v4/REPORT.md),
[round three](experiments/joint-recovery-v3/REPORT.md), [round two](experiments/joint-recovery-v2/REPORT.md)
and [round one](experiments/joint-recovery/REPORT.md).

TransformerLens or SAEs are deferred unless a concrete mechanism warrants an intervention. No new layer or BPC sweep is planned.

Apply candidate Voynich mappings consistently across unseen passages. Test repeated forms and independent image associations. Record competing interpretations and contradictions. Fluent output, feature names, and round-trip consistency are insufficient evidence on their own.

**Success:** a fixed interpretation makes independent predictions that alternatives fail. English or Italian is the output language; neither is assumed to be the manuscript's source language.

## Deliverables and progress

1. Dataset audit, fixed splits, and working controls.
2. Baseline and selective-adaptation results table.
3. Context, transcription, and boundary robustness report.
4. Frozen decipherment-method benchmark, fresh positive controls, negative results, and an image-association pilot.
5. Reproducible report suitable for assessment as a methods contribution; publication is not guaranteed.

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
| Larger base-model comparison | Complete: 8B gain over 1.7B is 0.0083 BPC, with folio interval crossing zero; pod/storage deleted |
| Character models from scratch | Twelve runs complete: GRU mean 2.332 BPC; training exposure differs from Qwen |
| Exact context ablation | Thirty evaluations complete: 8–128 characters, fixed GRU weights, stride 1; `experiments/CONTEXT.md` |
| Training matched to short histories | Complete: 18 verified runs; no established extra intact context gain, with later overfitting; `experiments/report-matched/REPORT.md` |
| Historical/modern language statistics | Complete: eleven pinned corpus samples, two Voynich transcriptions, adjacency controls; `experiments/language-comparison/REPORT.md` |
| Known-plaintext recovery | First controlled benchmark complete; spaced substitution exact, space-free word recovery still poor; `experiments/decipherment/REPORT.md` |
| Lexicon segmentation | Fresh 24-passage evaluation complete: 6.1% modern / 39.9% historical word error |
| Codebook-free recovery | Complete: modern substitution passes; historical selection and broader/Naibbe controls fail; `experiments/codebook-free/REPORT.md` |
| Joint segmentation + EM | Four fresh rounds complete: round one 12.5% CER modern / 33.5% Dante; round two 9.5% / 10.3% with pruning and a verse prior; round three 8.8% / 10.5% with a lexical polish; round four 5.7% Dante with lexicon repair; gate not met; oracle ceiling 0.5%; `experiments/joint-recovery-v4/REPORT.md` |
| Text-image association | Original pilot preserved; complex 123-description extension complete, 12 corrected tests, no established gain; synthetic nonlinear control passes |
| Broad image domains | Complete: 63 folio groups, three domains; text adds no gain over hand/layout; cross-quire and identifiability limits documented |
| Linear A track | Closed 2026-09-23: five rounds; Linear B used as the known-answer control; no language identified; `docs/LINEAR_A.md` |
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

## Superseding priorities: 2026-09-21

The [standard-method protocol](experiments/standard-decipherment/PROTOCOL.md) supersedes
older next-step suggestions. Archive committed (`49a4522`); use total description length
including key and residual ambiguity costs; compare published homophonic search; build
historical training/development from Novellino and Decameron, with Dante held for evaluation.
Freeze in Git before new grading. CPU only. Image association modelling is parked pending independent
masked annotations from two annotators. No further catalogue study. The reserved Voynich
shuffle mechanism remains gated on Naibbe recovery. Write up methods and bounded negatives;
do not claim a translation or a general impossibility result.

### Completed standard-method comparison

Frozen at `752c3fa` before fresh grading; [report](experiments/standard-decipherment/REPORT.md).
Seven of eight substitution/homophonic controls have exact selected letters; the eighth
has 0.63% CER. MDL fixes the historical substitution selection failure. Non-Dante historical
prose lowers fresh Dante WER from 37.7% to 24.2%, still above the 10% gate. Naibbe and
variable-length recovery remain unsuccessful. No Voynich mechanism run is earned by this
result. Published one-letter model failures cannot reject every variable-length cipher.
The [methods note](experiments/method-benchmark/METHODS_NOTE.md) now includes the comparison.
Image association modelling remains parked; image description is reopened above.
No further decoder run is scheduled; any decoder revision needs
new development evidence, a new committed freeze and source IDs disjoint from this release.


### 2026-09-22: object-and-relation description pilot

The [24-panel pilot](data/folios/object-pilot/REPORT.md) extends image descriptions to
visible objects and relations. Tables include `analysis_id`, evidence regions,
uncertainty and physical-folio/foldout groups. The descriptions are a single AI
observer's development annotations. Text masking, two independent human reviews and
agreement estimates remain pending; no association model or Voynich word claim is
licensed by this pilot. The previous pixel-only method remains available for comparison.


### 2026-09-23: segmentation audit completed

[Audit](experiments/segmentation-audit/REPORT.md): perfect-letter word error is 15.3%
on historical prose, 6.9% modern and 28.5% Petrarca development verse. A true
cipher-piece inventory oracle reaches 0.38–0.65% character error but still fails
historical word recovery. The one targeted fixed-key reparse change improves mean
CER by only 0.29 points, below its predeclared 1-point minimum; it is rejected.
Fresh author/corpus evaluation was conditional on selection and was not run.

Next: declare a bounded word-segmentation development comparison using training-only
historical prose and verse; freeze before any new grading. Do not add observed
missing development words directly to a lexicon. Any selected decoder must then be
compared with the baseline on identical new passages and keys, including a new
historical author and a newly pinned modern corpus. The recovery gate and the
closed Voynich mechanism test remain unchanged.

### 2026-09-23: training-only verse word comparison completed

The [word-model comparison](experiments/word-segmentation-v2/REPORT.md) implements
the preceding next step for perfect-letter segmentation. Three weights were tried;
weight 1 won development selection and was committed before fresh source preparation.
Petrarca development WER fell from 28.49% to 14.04%, but fresh Villani only improved
28.37% to 27.35%, below the declared 3-point transfer threshold. VIT modern WER was
8.49% to 8.38%. The candidate is not promoted to cipher recovery; no paired cipher
evaluation was earned. The historical source also retained chapter rubrics (107
words), a documented deviation from prose-only extraction. Frozen outputs are kept.

Next bounded work: test paragraph-encoded rubric removal in a new extractor version,
then diagnose missing forms versus incorrect splits on existing development text.
Declare and freeze any later candidate before new grading; exclude the released
Villani/VIT source IDs. No further weight sweep, paid model, Naibbe test or Voynich
mechanism run follows from this result. The image track still needs independent
text-masked human annotations.
