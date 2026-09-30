# Voynich research: from a better decoder to a testable language comparison

Research synthesis · 25 September 2026, updated 30 September 2026

This project tests whether a method can recover known text from a Voynich-like cipher before trusting it on the manuscript.
**The main progress is a stronger, more auditable benchmark: recovery has improved, eight languages can now compete, the system can reject an answer, and the language screen has now been confirmed twice on fresh sources and run once on the manuscript's non-reserved pages, where it reached no decision—no Voynich language or translation has been established.**

**Naibbe:** the published cipher family used to generate our known-answer controls; it is a hypothesis-testing tool, not an established explanation of Voynich.
**Character / word error:** edit distance divided by the reference length; insertions, deletions and substitutions count.
**Fixed-key transfer:** decoding a second passage without changing the key learned from the first.
**Excess:** a model's decoded-text score, in bits per letter, minus its score on separate calibration text; lower is better.

## What was done

- Reviewed the earlier prediction, image, segmentation and cipher-recovery results, including corrections to invalid test cases.
- Added language rejection and fixed-key transfer tests, preserving two attempts that stopped at compute limits.
- Completed the three follow-ups: revised decision calibration, stronger copying controls and a faster key-refinement implementation.
- Broadened historical Latin and German, added Catalan, then added Old Czech and Old Occitan.
- Preserved frozen protocols, source licenses, input hashes, predictions, failed gates and replayable evaluated records. The latest suite contains 194 passing tests.
- 26–30 September (section 8): fixed the rare-piece bottleneck with whole-token admission; confirmed the screen on fresh sources twice (13/24 with narrow priors, then 16/24 with rebuilt Latin, German and Catalan priors, 0 false acceptances); found and fixed a transfer-scoring bug; tried four further decoder changes without gain; tested the Voynich-like pairing setting; ran the screen once on the non-reserved Voynich pages.

## Why it was done

Recovering plausible text is insufficient if a solver must always name a language, succeeds only on familiar material, or can change its key for every passage. The new experiments test those weaknesses explicitly rather than treating a favorable score as a decipherment.

## 1. The starting point was stronger than a stalled result suggested

The first review needed a correction: the project had already advanced beyond the roughly 5.6% character-error result. The longer-passage decoder achieved **1.83% character error and 26.8% word error** on four valid historical Italian cases, each about 20,800 letters. Its paired baseline gave 3.85% and 34.1%. The two modern cases were excluded after a training-overlap audit; they are not part of that headline. [Round six](../experiments/joint-recovery-v6/REPORT.md), [overlap audit](../experiments/partut-overlap-audit.json).

That is substantial progress on a known cipher, but it misses the project's manuscript-readiness gate of **1% character error and 10% word error**. The remaining errors have identifiable causes: missing cipher pieces, imperfect key recovery, and word segmentation. Even correct letters did not solve historical word boundaries: a separate segmenter improved fresh Compagni word error from 24.0% to 17.5%. [Full recovery account](VOYNICH.md), [segmentation test](../experiments/word-segmentation-v3-fresh/REPORT.md).

The other tracks explain why decoding became the priority:

| Track | Concrete finding | Consequence |
|---|---|---|
| Prediction | A small character model reached 2.332 bits/character; gains also appeared on shuffled and synthetic controls | Predictability alone cannot establish meaning; this track was closed. [Record](../experiments/report-completed/REPORT.md) |
| Corpus statistics | Adjacent word-length correlation was +0.218 in v101 and +0.169 in EVA | This challenges simple substitution accounts that preserve ordinary word boundaries; it does not name a language. [Record](../experiments/language-comparison/REPORT.md) |
| Images | No reported association survived the relevant hand/layout controls; some comparisons were unidentifiable | Image-based decoding remains unsupported; the newer object annotation pilot awaits human review. [Studies](../experiments/image-domains/REPORT.md), [pilot](../data/folios/object-pilot/REPORT.md) |
| Cipher mechanism | Heavy letter pairing could match EVA's near-duplicate rate, but not v101's; another statistic still disagreed | Successful Naibbe recovery does not establish that Voynich uses Naibbe. [Record](../experiments/naibbe-near-duplicates/REPORT.md) |

The useful change in direction was to test reliability around the decoder, while keeping those limitations visible.

## 2. A decoder needs a tested “none of these” answer

We introduced the **fixed-key transfer test**:

```text
Freeze language models, decision thresholds and compute limits
Encrypt two passages with one hidden key
Fit each candidate model to the first ciphertext
Seal each fitted key
Decode the second ciphertext without updating the key
If a compute limit was reached: report inconclusive
Otherwise: accept only if both rankings and all evidence gates agree
Repeat with shuffled, copying and missing-language controls
```

The initial attempt stopped when wrong-language searches reached their limits. A separately frozen resource repair used fresh inputs, but also stopped early: **three of ten planned source/key blocks** were completed. English, Italian and Latin each ranked correctly on both passages; only Italian and Latin passed the original acceptance rule. There were eight conclusive negative decisions and one inconclusive copying decision, all sharing those three source blocks. A timeout was never counted as a successful rejection. [First attempt](../experiments/rejection-transfer/REPORT.md), [repaired screen](../experiments/rejection-transfer-v2/REPORT.md).

English revealed a specific problem. Its transfer error was 6.91%, and its transfer excess was 0.374, below the 0.50 ceiling. It failed solely because its fitting excess was 0.584. The threshold was mixing variation between genuine source passages with reconstruction error. This became a development question; the original failed decision stayed in the record.

## 3. The three follow-ups addressed different weaknesses

1. **Decision calibration.** Removing only the fitting-score ceiling changed acceptance from **2/3 to 3/3** on the released genuine cases. The candidate retained matching language winners, minimum margins of 0.25 bits/letter, transfer excess at most 0.50, at least 95% token coverage, and compute-limit checks. These were reused development cases, not independent confirmation.

2. **Stronger negatives.** Earlier mutation controls introduced unfamiliar pieces, making rejection partly easy. New copying controls preserved every token's frequency and the entire token inventory while increasing local repetition. All **three** were rejected without caps, despite **98.47–98.64%** transfer coverage. Each control received its own full search under all five language models.

3. **Cheaper search.** Incremental refinement rescored only the character windows affected by a proposed key change, with full rescoring for close decisions. On the released copying fixture, one exhaustive swap batch fell from **4.817 to 0.155 seconds: 31.1× faster**. Peak worker memory fell from **1,694 to 331 MiB**. The winning swap and full score matched. This is a search-step benchmark, not a 31× speedup of the complete decoder.

All three results, including the preserved inconclusive mutation case, are in the [follow-up report](../experiments/rejection-followups/REPORT.md). They made further controlled comparisons practical; they did not establish a population false-positive rate.

## 4. Historical source coverage changed the answer

The first expansion added **Catalan**, broadened Latin with historical charters, and broadened German with Middle High German prose. All models used exactly **400,000 training letters** and **20,000 calibration letters**. Three fresh pairs compared the original five-language set with six expanded candidates. Eight models were needed because both original and broad Latin/German versions were tested. [Protocol and results](../experiments/language-coverage/REPORT.md).

| True source | Original model: transfer character error | Expanded model: transfer character error | Expanded acceptance |
|---|---:|---:|---|
| Latin | 12.37% | **6.69%** | Passed |
| German | 45.69% | **12.48%** | Failed: transfer excess too high |
| Catalan | No candidate | **9.27%** | Failed: fitting margin and transfer excess |

The original candidate set ranked Old French first on both passages in every case, but rejected all three. The expanded set ranked the correct language first in all three cases. Removing each true language also produced rejection in all three. **Better ranking was real; the three-case acceptance target still failed.**

Post-run diagnostics separated source coverage from recovery: the correct German and Catalan transfer plaintexts had excesses of 0.190 and 0.150, within the unchanged ceiling. Their decoded versions failed. On these examples, improving the recovered key remained a concrete next target.

## 5. Czech and Occitan brought the active comparison to eight languages

At the user's request, we added **Old Czech** from DIAKORP/HistCorp and **Old Occitan** from COMETA. Czech adds historical medical material and the first Slavic candidate in this cipher comparison. Occitan adds another medieval Romance candidate alongside Catalan and Old French. The [source manifest](../experiments/language-expansion/sources.json) records exact works, licenses, downloads and hashes.

Each new model uses 100,000 letters from each of four training works. Two other works supply calibration; two further works supply the 5,200-letter fitting and transfer passages. The six existing models reproduce exactly. The test ran **16 fits** with no caps. [Full report](../experiments/language-expansion/REPORT.md).

| New language | Fit / transfer winner | Transfer character error | Transfer excess | Acceptance |
|---|---|---:|---:|---|
| Old Czech | Czech / Czech | **5.54%** | 0.339 | Passed |
| Old Occitan | Occitan / Occitan | **15.19%** | 0.525 | Failed: ceiling is 0.500 |

Both missing-language controls rejected. Occitan's 0.025 miss remains a failure; no threshold was relaxed. Its correct transfer plaintext scored at **−0.139 excess**, so decoding errors added about **0.664 bits per letter**. That points to recovery quality on this case, rather than an inability of the model to recognize the correct text.

The active candidates are now **Latin, German, Old French, English, Italian, Catalan, Old Czech and Old Occitan**. Eight candidate languages do not mean eight equally validated historical models. The source collections differ in period, genre and spelling; Czech accents collapse in the shared cipher alphabet, Catalan still comes from one chronicle, and the latest extension did not run fresh retention tests for the previous six languages.

## 6. What the combined evidence supports

1. **The benchmark is stronger.** Language ranking, acceptance, character recovery and word recovery are now separate measured outcomes. Czech can pass a language-acceptance control at 5.54% character error while still missing the much stricter 1%/10% manuscript-readiness gate.

2. **Small pilots remain small.** The three-case historical-coverage pilot and two-case extension use different candidate sets and source conditions. Their successful rankings must not be pooled into a claimed general accuracy rate. The three follow-up blocks were development, and the earlier capped screen remains inconclusive.

3. **The evidence is inspectable.** The latest audit rebuilt eight priors, re-extracted 2,538 source rows, reproduced four encryptions, and replayed 16 transfers. Checks found no selected challenge eight-word overlaps against the audited references or previously released passages. Source-work exclusions and evaluated archives are published with the reports. Blinding was procedural on one machine, not an independent evaluation. [Audit](../experiments/language-expansion/audit.json), [reproduction](REPRODUCE.md).

4. **The next experiment should improve recovery, then confirm it on fresh material.** German, Catalan and Occitan now provide released cases where correct text scores well but reconstructed text does not. A useful improvement must reduce actual reconstruction errors without winning merely by weakening the decision rule.

## 7. The research decision of 25 September, and what came of it

```text
Develop key recovery on the released failure cases        -> whole-token admission, 3/5 -> confirmed
Keep the existing thresholds fixed during that comparison  -> kept; a scoring bug was found instead
Audit unused works, parallel versions and compute          -> 92 works pinned and released over two rounds
Freeze a new confirmation protocol                          -> twice: 443098f, 7d56dd1
Test new keys and works, including the earlier six          -> 13/24 then 16/24, 0 false acceptances
Reconsider manuscript readiness only after the gates pass  -> the language screen ran once; the recovery gate is still unmet
```

The 90-block design was replaced by two 24-block confirmations (576 fits each, 44–48 worker-hours on one machine), which were affordable and decisive. Section 8 has the details.

## 8. Update, 26–30 September 2026

**Decoding cost:** transfer excess of the decoded text minus that of the true text under the same prior; the gap that decoding errors add.

1. **The screen is confirmed, at the threshold.** The first fresh confirmation failed at 13/24 because three priors were too narrow for other genres of their own language: the true plaintext itself scored 0.52–1.40 bits over calibration. Rebuilt Latin, German and Catalan priors passed the second at 16/24 with 0 false acceptances in 72 wrong-language, omitted-language and negative decisions ([round two](../experiments/key-recovery-confirmation-v2/REPORT.md)). This state is tagged `checkpoint/2026-09-28-v2-pass`.

2. **Recovery, not the priors, now limits sensitivity.** Seven of the eight remaining failures would pass with the true key; decoding adds a median 0.34 bits per letter. Supplying the true lexicon halves that; supplying true letters alone barely helps. The residual error is a systematic split point (`l`+`chdy` read as `lch`+`dy`) that four cipher-internal signals all get wrong ([oracles](../experiments/recovery-oracles/REPORT.md), [attempts](../experiments/lexicon-admission-development/REPORT.md)).

3. **A scoring bug hid inside the safety margin.** The transfer score charged one length code per unreadable gap, about 0.13 bits per letter. Fixed in `voynich/rejection_v3.py` without touching frozen files. Rescored honestly, the second confirmation gives 20/24, but one released Catalan block is then accepted as Occitan when Catalan is omitted; a 0.45 ceiling or a Catalan–Occitan decision group restores a clean record on all released rounds. Neither is confirmed on fresh text ([rescore](../experiments/transfer-length-rescore/REPORT.md)).

4. **At the Voynich-like pairing setting** the confirmed rule accepts 13/24 and the candidate rule 19/24 with no false acceptance, but five negatives hit the work limit ([RESPACING 9](../experiments/respacing9-development/REPORT.md)).

5. **On the manuscript, the screen cannot conclude.** Pre-registered and run once on four blocks of non-reserved pages: three blocks exceed the work limit (keys of 652–815 units against 356 for real ciphertext), the fourth reads only 91.7% of its second passage. Nothing is accepted. Every manuscript passage scores within 0.1 bits per letter of its own shuffle and frequency copy, where a real cipher passage separates from its shuffle by more than a bit ([report](../experiments/voynich-language-screen/REPORT.md)). Under this cipher family and these eight languages, the token order of these pages carries no usable structure. Other mechanisms, languages and pairing settings are untested; the reserved pages were not read.

**Next decision, not taken.** A second manuscript run would need a higher work limit and a coverage-tolerant rule, both re-confirmed on known-answer controls first; the sources for another fresh confirmation are limited, since 92 works are now released. Longer passages and other cipher mechanisms remain the untested levers.

## Reading map

- [Detailed Voynich account](VOYNICH.md): the full method history, segmentation work, corrections and manuscript statistics.
- [Results table](RESULTS.md): numerical outcomes across all tracks.
- [Research log](../RESEARCH_LOG.md): chronological experiment record.
- [Reproduction instructions](REPRODUCE.md): commands, archives, frozen dependencies and source licenses.
- The other undeciphered-script branches are separate research efforts. This synthesis and merge concern the Voynich work completed in this task.
