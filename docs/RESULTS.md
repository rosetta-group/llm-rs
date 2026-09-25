# Results at a glance

Every frozen or verified result, grouped by track. Numbers are as reported in the linked
record; "sealed" means the answers were hidden until predictions were saved. Nothing here
scores Voynich's reserved test pages.

Metrics: **BPC** bits per character (prediction; lower is better). **CER / WER** character
and word error rate (recovery; lower is better). See the [glossary](GLOSSARY.md).

## Prediction on Voynich text (validation pages; track closed)

| Experiment | Result | Reading | Record |
|---|---|---|---|
| Baselines | frequency 4.245 BPC; best copy baseline 2.699 | Local repetition explains a lot of predictability | [RESULTS.md](../experiments/RESULTS.md) |
| Qwen 1.7B adapters, 5 seeds | 2.483 BPC (SD 0.002) | Stable gain over copying | [REPLICATIONS.md](../experiments/REPLICATIONS.md) |
| Outer vs random layers | 2.4819 vs 2.4828 | No layer-specific effect | [LEARNING_CURVES.md](../experiments/LEARNING_CURVES.md) |
| Qwen 8B vs 1.7B (cloud) | 2.469 vs 2.477; interval crosses zero | Scale did not help measurably | [CLOUD_RESULTS.md](../experiments/CLOUD_RESULTS.md) |
| Character GRU from scratch, 3 seeds | 2.332 BPC | Beats the pretrained model | [CHARACTERS.md](../experiments/CHARACTERS.md) |
| Controls (shuffled, Timm, Naibbe) | gains of the same size on all | Prediction is not a meaning detector | [report](../experiments/report-completed/REPORT.md) |
| Context ablation, fixed GRU | 64 chars give 91% of the 8→128 gain; shuffled text gains too | History helps regardless of order | [CONTEXT.md](../experiments/CONTEXT.md) |
| Matched-context retraining, 18 runs | intact gain 0.123 vs shuffled 0.137 BPC | No extra benefit for intact text | [report](../experiments/report-matched/REPORT.md) |

## Corpus statistics (training pages)

| Experiment | Result | Reading | Record |
|---|---|---|---|
| Word-length adjacency, 11 languages | Voynich +0.218 (v101), +0.169 (EVA); languages −0.18 to +0.07; controls ≈ 0 | Rules out spaced substitution of a Romance language; does not identify the text | [report](../experiments/language-comparison/REPORT.md) |

## Text–image association (parked)

| Experiment | Result | Reading | Record |
|---|---|---|---|
| Root-colour pilot, 59 labels | +1.45 points over controls, p = 0.35 | No association established | [report](../experiments/association/REPORT.md) |
| Complex features, 123 descriptions, 12 tests | no corrected signal | Same | [report](../experiments/association-complex/REPORT.md) |
| Broad domains, 63 folio groups | text 72% balanced accuracy; hand/layout 94%; no gain from adding text | Domain and production are entangled | [report](../experiments/image-domains/REPORT.md) |
| Folio scans and pixel descriptions | 213 scans, 228 panels archived | Data only; no inference | [data/folios](../data/folios/README.md) |
| Object–Relation pilot | 24 panels; 2 with figures in linked basins, 5 with vessels beside botanical groups | Single AI development annotations; independent agreement pending | [pilot](../data/folios/object-pilot/REPORT.md) |

## Cipher recovery on sealed Italian passages (active)

Positive controls first, then the codebook-free Naibbe rounds. "Letters" is passage length.

| Round | Setup | Result | Record |
|---|---|---|---|
| Historical language coverage | 3 fresh historical source/key pairs; 8 models, 400,000 letters each; 24 fits | expanded true-language rankings 3/3 on both passages, acceptance 1/3 (Latin); omissions rejected 3/3; German transfer CER 45.69% → 12.48%, Latin 12.37% → 6.69%, Catalan 9.27%; no caps, 1.386 worker h; feasibility target failed | [report](../experiments/language-coverage/REPORT.md) |
| Rejection development follow-ups | 3 released source/key blocks; 3 new frequency-preserving copying pairs × 5 priors | candidate positives 3/3 vs original 2/3; stronger copies rejected 3/3 at 98.47–98.64% coverage, no caps; 15 fits / 1.028 worker h; development only | [report](../experiments/rejection-followups/REPORT.md) |
| Bounded incremental refiner benchmark | 4 fixtures × 3 backends × 2 warmed runs; old objective retained | copying-fixture swap batch 4.817 → 0.155 s (31.1×); RSS 1,694 → 331 MiB; every winning pair and exact score match; not an end-to-end speedup | [benchmark](../experiments/rejection-followups/benchmark.json) |
| Rejection + fixed-key transfer (resource repair) | 3 of 10 planned source/key blocks, 45 fits; old five-language decoder | positives accepted 2/3; shuffle rejected 3/3; absent language rejected 3/3; copy rejected 2/3 plus 1 inconclusive; 8.884 fit-worker h; stopped on refinement caps | [report](../experiments/rejection-transfer-v2/REPORT.md) |
| Rejection + fixed-key transfer (first attempt) | first English positive, 5 fits | English wins both rankings; wrong-language refinement caps make positive and absent-language decisions inconclusive | [report](../experiments/rejection-transfer/REPORT.md) |
| Assisted | substitution with spaces; Naibbe with codebook | 100% letters and words; 99.4% letters | [report](../experiments/decipherment/REPORT.md) |
| Segmentation | frozen lexicon segmenter on exact letters, 24 fresh passages | WER 6.1% modern, 39.9% historical | [report](../experiments/segmentation/REPORT.md) |
| Verse word model | exact letters, 4 fresh Villani + 4 VIT passages; paired baseline | Villani WER 28.37% → 27.35%; VIT 8.49% → 8.38%; misses 3-point transfer threshold; historical rubrics retained in error (2.24% of words) | [report and caveat](../experiments/word-segmentation-v2/REPORT.md) |
| Context admission of rare pieces (dev) | 20,800 letters, RESPACING 17 and 9 | CER 1.79% → 6.24% and 3.30% → 9.99%; 832–1,597 admitted, 2–3% true: rejected | [report](../experiments/length-scaling-v5/REPORT.md) |
| Post-reparse pruning (dev) | 20,800 letters, 3 dev texts | CER 1.79% → 1.63% (not adopted); spurious pieces 178–200 → 3–6, missing unchanged 33–38 | [report](../experiments/length-scaling-v4/REPORT.md) |
| Decoder at RESPACING 9 (dev) | same texts, 75% of letters paired | mean CER 3.30% (vs 1.79% at 17); missing pieces 52–56 | same |
| Word segmentation v4 (historical spelling) | dev, then fresh Sacchetti (4 passages); modern half invalid | dev prose 8.65% → 6.80%, verse 20.68% → 14.95%; Sacchetti 8.82% → 6.87% (−1.95, threshold 2): not promoted | [report](../experiments/word-segmentation-v4-fresh/REPORT.md) |
| Naibbe near-duplicate grid | 90 encryptor settings × 2 seeds | reaches EVA 8.33% only at RESPACING 9 (best 9.36%); never v101 12.2%; hapax types 55–62% vs Voynich 71% | [report](../experiments/naibbe-near-duplicates/REPORT.md) |
| **Round six (sealed)** | ciphertext-only cases × ~20,900 letters; **valid: 4 historical (2 Dante, 2 Compagni)**; 2 ParTUT cases invalid (ISDT-train leakage) | historical CER 3.85% → **1.83%**, WER 34.1% → **26.8%**; 4/4 improve; gate 0/4 | [report](../experiments/joint-recovery-v6/REPORT.md) |
| Context reparse + key refit (dev) | square-root baseline plus two reparse/refit rounds; 10,400 / 20,800 letters | mean CER 2.45% / **1.79%** (baseline 3.19% / 3.25%); modern 20,800 1.31% CER, 12.8% WER; glue-test candidate collapses | [report](../experiments/length-scaling-v3/REPORT.md) |
| Length-aware lexicon (dev) | thresholds scaled linearly (primary) or by √length; 10,400 / 20,800 letters | linear 4.78% / 5.14% (fails); √ 3.19% / 3.25% vs unscaled 3.39% / 4.18% | [report](../experiments/length-scaling-v2/REPORT.md) |
| Length scaling (dev) | round four on 5,200 / 10,400 / 20,800-letter prefixes of 3 dev texts | mean CER 5.62% → 3.39% → 4.18%; spurious pieces 32–75 → 295–408 at 20,800; rule not met | [report](../experiments/length-scaling/REPORT.md) |
| Voynich transcription suspects | training pages, v101 and EVA, 25k-token natural controls | near-hapax 10.4% / 6.8% of tokens vs 0.9–2.6% natural; 447/2,878 lines differ by ≥ 2 words | [report](../experiments/voynich-suspects/REPORT.md) |
| Language-ID control | 5 Naibbe ciphertexts (Latin, Old French, German, English, Italian) × 5 equal priors | true language ranks first 5/5; median margin 1.28 bits/letter; Italian smallest (0.75) | [report](../experiments/language-id/REPORT.md) |
| Round five (v3 segmenter in Naibbe) | ciphertext only, 8 sealed cases (4 Dante, 4 Compagni), paired arms | CER 5.63% unchanged; WER 45.81% → 41.46% segmentation only (8/8 improve), 41.69% with v3 polish; gate failed | [report](../experiments/joint-recovery-v5/REPORT.md) |
| v3 unknown-word model | exact letters, 4 fresh Compagni + 4 ParTUT passages; paired baseline | Compagni WER 24.01% → 17.46%; ParTUT 7.23% → 6.05%; transfer threshold passed; 10% word gate not met | [report](../experiments/word-segmentation-v3-fresh/REPORT.md) |
| Codebook-free v0 | annealing, mean score, 1,200–1,800 letters | modern substitution ok; historical selection fails; Naibbe 300%+ CER | [report](../experiments/codebook-free/REPORT.md) |
| Standard methods | MDL selection, Nuhn beam, BK&K HMM, prose prior | 7/8 substitution and homophonic controls exact; Naibbe still 300%+ | [report](../experiments/standard-decipherment/REPORT.md) |
| Joint EM, round one | latent parses, role emissions, 5,200 letters | CER 12.5% modern, 33.5% Dante; WER 58% / 87% | [report](../experiments/joint-recovery/REPORT.md) |
| Joint EM, round two | + usage pruning, + Petrarch verse in prior | **CER 9.5% modern, 10.3% Dante**; WER 54% / 60% | [report](../experiments/joint-recovery-v2/REPORT.md) |
| Joint EM, round three | + lexical polish after the round-two decoder | CER 8.8% modern, 10.5% Dante; WER 54% / 56%; polish gains 0.5–1.5 points paired | [report](../experiments/joint-recovery-v3/REPORT.md) |
| Joint EM, round four | + lexicon repair before refinement; four Dante passages (modern test text exhausted) | **CER 5.7% Dante**; WER 45%; parse agreement 90% → 93% on every case | [report](../experiments/joint-recovery-v4/REPORT.md) |

Development findings that shaped the rounds (development text only):

| Finding | Evidence | Record |
|---|---|---|
| Search, not scoring, blocked recovery | true key scores best under every objective; only EM reaches it | [round one dev](../experiments/joint-development/REPORT.md) |
| Length is a hard limit | 77% CER at 1,300 letters, 4.6% at 2,600, 3.2% at 5,200 with true splits | same |
| Splits and letters must be learned jointly | fixed splits with 14% errors collapse EM to 77% | same |
| Pruning helps, one pass suffices | 14.2% → 10.4% prose, 12.2% → 9.4% modern | [round two dev](../experiments/joint-development-v2/REPORT.md) |
| Verse prior fixes verse | Petrarch dev 53% → 9.5%; no regression on prose | same |
| Remaining error is in mis-parsed tokens | 0.5–1.0% letter error inside correct parses; ~12% of letters in wrong parses; confirmed on fresh text (0.5–1.4%, 10–14%) | [round three dev](../experiments/joint-development-v3/REPORT.md), [fresh](../experiments/joint-recovery-v3/REPORT.md) |
| Polish weight 1 is the only safe setting | higher weights help prose, hurt verse; token re-parsing and warm EM restarts rejected | [round three dev](../experiments/joint-development-v3/REPORT.md) |
| The lexicon, not the model, limits parsing | with the true piece lexicon the same EM reaches 97% agreement and 0.5% CER | [round four dev](../experiments/joint-development-v4/REPORT.md) |
| Two lexicon defects, both repairable from ciphertext | spurious whole pieces are all bigram concatenations (count ≈ expected from the halves); missing pieces are rare-letter pieces below the threshold; repair 10.9% → 5.0%, 9.3% → 5.5%, 9.5% → 6.2% | same |
| Lower thresholds and global admission fail | candidate threshold 3: 12.0%; complements for every token: 17.2% | same |
| Perfect letters do not fix historical word boundaries | development WER 15.3% historical prose, 6.9% modern, 28.5% Petrarca | [segmentation audit](../experiments/segmentation-audit/REPORT.md) |
| Fixed-key context reparse rejected | paired mean CER 4.97% → 4.67%, below the predeclared 1-point improvement; WER 37.3% → 36.0%; no fresh evaluation | same |
| Extra spaces are missing word forms | 83–90% of extra spaces fall inside lexicon-missing words; adding them (oracle) gives historical WER 2.29% | [diagnosis](../experiments/word-segmentation-v3/REPORT.md) |
| Letter-level unknown-word cost fixes much of it | development WER 15.27% → 8.65% prose, 28.49% → 20.68% verse, 6.89% → 5.61% modern | same |
| Training-only verse words help held-out Petrarca | WER 28.49% → 14.04%, historical prose 15.27% → 14.92%, modern unchanged at 6.89%; weight 1 selected before fresh sources fetched | [word model](../experiments/word-segmentation-v2/REPORT.md) |

## Linear A (branch `linear-a`; track closed; summary in [LINEAR_A.md](LINEAR_A.md))

Each method had to find Greek in Linear B (DĀMOS Knossos) at Linear A's size before Linear A was
run. Rounds one to three failed that control. Rounds four and five found no lead that survived
their controls. So no Linear A language result exists.

| Round | Method | Result (Linear B control: Greek identified) | Record |
|---|---|---|---|
| Anchors | Linear B Cretan toponyms in Linear A; `ku-ro` totals | 2 of 14 found, chance 0.025; 8 of 37 totals exact | [report](../experiments/linear-a/REPORT.md) |
| One | lexicon match, Linear B spelling, 8 languages | 10% of draws; gate 90%; failed | [report](../experiments/linear-a/REPORT.md) |
| Two | lexicon match plus name–position agreement | 0% of 20; failed | [report](../experiments/linear-a-context/REPORT.md) |
| Shared words (exploratory) | Linear A types also in Linear B | 94 against 72.7 by chance (max 88); no gate | [report](../experiments/linear-a-context/REPORT.md) |
| Three | entry words against 4 proper-name lists | 0% of 20; failed | [report](../experiments/linear-a-names/REPORT.md) |
| Four | 7 probes: Egyptian place names, Keftiu names, gods, trade words, profiles, spelling rules, role transfer | no probe below p = 0.007; `-re`/`-ru` → `-ro` 12 pairs vs 3.1 (p = 0.0099, the floor of 100 null runs); Levant name-profile lead gone after length matching | [report](../experiments/linear-a-probes/REPORT.md) |
| Five | length-matched profiles vs TLHdig Hittite, Luwian, Palaic, Hurrian, Hattic, Akkadian | gate passed (Linear B → Greek 20 of 20; Linear A → Hittite 20 of 20) but shuffled syllables give Hittite 20 of 20 too; artefact of *o*/*u* and syllable frequencies | [report](../experiments/linear-a-tlhdig/REPORT.md) |

## Proto-Elamite (number systems, validated on proto-cuneiform)

| Round | Method | Control (proto-cuneiform) | Proto-Elamite | Record |
|---|---|---|---|---|
| One | two-sign tablets solved for the N14/N01 ratio | 10 recovered (p = 0.001); 6 missed; gate failed | not run | [report](../experiments/proto-elamite/REPORT.md) |
| Two | joint search, 3 latent systems | sexagesimal and capacity values exact; gate passed | N14 = 10 (11 tablets) and 6 (9 tablets) recovered blind | [report](../experiments/proto-elamite-joint/REPORT.md) |
| Three | object signs by strict arithmetic label | 15 tablets labelled; gate failed | not run | [report](../experiments/proto-elamite-signs/REPORT.md) |
| Four | object signs by held-out system label | barley on capacity tablets, p = 0.018 | grain signs M288/M036/M297 on 10 of 11 capacity vs 1 of 8 counting tablets, p = 0.0012; people/animal signs p = 0.55 | [report](../experiments/proto-elamite-signs-loo/REPORT.md) |
| Five (exploratory) | constant quantity ratios between consecutive entries, 46 pairs | no control available | only survivor is a ratio of 1 (repeated amounts); no exchange rate found | [report](../experiments/proto-elamite-ratios/REPORT.md) |

## Operational

| Item | Status | Record |
|---|---|---|
| Cloud compute | pod and volumes deleted 2026-09-21; $8.89 balance at check | [audit](../experiments/cloud-cleanup.json) |
| Voynich final test pages | never scored | all reports |
| Reserved Voynich mechanism test | closed until a recovery gate is met | [protocol](../experiments/joint-development/PROTOCOL.md) |
