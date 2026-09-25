# Historical language coverage pilot

Pattern: **language coverage test**. Compare fixed candidate sets on ciphertext with known historical sources.

The expanded system accepted 1/3 true languages; omitting the true language produced 3/3 conclusive rejections.

**Transfer:** decode a second passage with the first passage’s sealed key.
**Excess:** decoded bits per letter minus that model’s separate calibration score.
**CER:** character edit distance divided by the 5,200 plaintext letters; gaps count as errors.

## What was done

- Audited and pinned Latin charters, Middle High German prose and Old Catalan.
- Compared five original languages with six expanded candidates on three fresh passage pairs.
- Refit all eight required models on exactly 400,000 letters each; calibration used 20,000 letters each.
- Completed 24/24 fits in 1.386 aggregate fit-worker hours.
- Rebuilt 8 priors and replayed 6 encryptions and 24 fixed-key transfers.

## Why it was done

The earlier five-language pilot had one passage per language and mixed modern and historical sources. This tests whether broader historical coverage improves identification without forcing an incorrect label when a language is unavailable.

```text
Freeze equal-budget models, sources and decision rule
Encrypt three independent-key passage pairs
Fit models on the first passage and seal each key
Transfer fixed keys to the second passage
Compare baseline, expanded and true-language-omitted decisions
```

## Results

| True source | Baseline accepted | Expanded accepted | Expanded fit / transfer winners | Omitted accepted |
|---|---|---|---|---|
| catalan | none | none | catalan / catalan | none |
| german | none | none | german / german | none |
| latin | none | latin | latin / latin | none |

Feasibility target: **not passed**. Stop reason: `None`.

| Source | Model | Fit excess | Transfer excess | Coverage | Fit CER | Transfer CER |
|---|---|---:|---:|---:|---:|---:|
| catalan | catalan | 0.607 | 0.860 | 98.77% | 6.21% | 9.27% |
| german | german | 1.446 | 2.382 | 97.86% | 40.40% | 45.69% |
| german | german_broad | 0.436 | 0.745 | 97.86% | 8.81% | 12.48% |
| latin | latin | 1.656 | 1.986 | 98.03% | 9.17% | 12.37% |
| latin | latin_broad | 0.309 | 0.429 | 97.91% | 4.94% | 6.69% |

Broad models also have different calibration entropies. The following raw scores help separate that change from better text prediction; a smaller excess alone is not proof of improved recovery.

| Source | Model | Calibration bits/letter | Decoded transfer bits/letter | True transfer plaintext bits/letter |
|---|---|---:|---:|---:|
| catalan | catalan | 2.354 | 3.214 | 2.504 |
| german | german | 2.795 | 5.176 | 4.596 |
| german | german_broad | 3.204 | 3.949 | 3.394 |
| latin | latin | 2.567 | 4.553 | 4.931 |
| latin | latin_broad | 2.494 | 2.923 | 2.206 |

True-plaintext scores are post-run diagnostics only. They did not select models, thresholds or challenge passages.

Decision reasons (unchanged thresholds):

- catalan / baseline: transfer_excess. Fit margin 0.689; transfer margin 0.820.
- catalan / expanded: fit_margin, transfer_excess. Fit margin 0.148; transfer margin 0.735.
- catalan / omitted: transfer_excess. Fit margin 0.333; transfer margin 0.263.
- german / baseline: transfer_excess. Fit margin 0.461; transfer margin 0.426.
- german / expanded: transfer_excess. Fit margin 0.549; transfer margin 1.211.
- german / omitted: transfer_excess. Fit margin 0.714; transfer margin 0.849.
- latin / baseline: transfer_excess. Fit margin 0.646; transfer margin 0.493.
- latin / expanded: all gates passed. Fit margin 0.453; transfer margin 1.064.
- latin / omitted: transfer_excess. Fit margin 0.260; transfer margin 0.342.

## Interpretation and next target

1. **Coverage changes the answer.** The baseline ranks Old French first on both passages of all three cases, but rejects all three. The expanded system ranks the correct language first on both passages in all three. A winning language label alone would have been misleading under the original candidate set.

2. **Recovery improves materially.** German transfer CER falls from 45.69% to 12.48%; Latin from 12.37% to 6.69%. These error reductions are independent of entropy normalization. Catalan transfer CER is 9.27%.

3. **The acceptance target fails.** Only Latin passes. German transfer excess is 0.745, above 0.50. Catalan has transfer excess 0.860 and a fitting margin of 0.148, below 0.25. Coverage exceeds 97% for all true models, and no cap explains these failures. All three omitted-language cases reject; that is a paired pilot result, not a population false-acceptance estimate.

4. **Work on key recovery next.** Post-run scores of the correct German and Catalan transfer plaintexts have excesses 0.190 and 0.150, both below 0.50. The recovered text adds about 0.56 and 0.71 bits/letter respectively. Use these now-released cases for recovery development, retain the thresholds, and require fresh source groups for confirmation. A second Catalan author and unseen medical prose remain specific coverage gaps. Adding more language names is lower priority.

The interpretation above describes this executed pilot only; it is not a newly tuned decision rule.

## Source audit and limits

1. **Latin is broader, not medically validated.** [LLCT](https://universaldependencies.org/treebanks/la_llct/) contains Tuscan legal charters from AD 774–897. The broad prior mixes 200,000 Aquinas letters with 200,000 charter letters; this does not represent fifteenth-century medical Latin.

2. **German uses historical surface forms.** [ReM 2.1](https://doi.org/10.5281/zenodo.13982324) supplies 191 prose documents from 1050–1350. Training includes medical and religious texts; the challenge is M113 (St. Trudperter Hohes Lied) and M172 (Prager Predigtentwürfe). Numeric work groups keep manuscript variants together. Dialect, genre and date remain mixed.

3. **Catalan remains within one work.** [HisCat](https://doi.org/10.5281/zenodo.5615759) is the thirteenth-century Llibre dels Fets. Complete folios separate training, calibration and challenge, but author and genre do not change. This is why Catalan was selected over using a modern news corpus; Occitan remains untested.

4. **Small, deliberately filtered pilot.** Three keys cannot establish sensitivity or false-acceptance rates. Overlap filtering removes formulaic chunks and makes some passages non-contiguous. The baseline also has 400,000 training letters, so compare it with the expanded system here, not directly with the older 606,976-letter pilot. The expanded system changes both models and candidate count; no old-language retention claim is justified.

| New source | Training-pool letters | Calibration letters after filtering | Challenge letters after filtering |
|---|---:|---:|---:|
| catalan | 427,576 | 50,144 | 49,385 |
| german | 2,774,932 | 315,731 | 551,320 |
| latin | 922,989 | 27,148 | 27,154 |

Final checks found zero eight-word overlaps for every selected challenge passage against all reference collections, including joins between retained chunks. Texts use a Latin-letter normalization with explicit historical glyph expansions. Neither this alphabet nor these successful/failed controls identify Voynich’s language or validate the cipher family.

An additional post-run audit found 0 eight-word matches against the earlier released challenge inventory. This check selected no replacement passages.

## Reproduction and records

- [Protocol](PROTOCOL.md), [source manifest](sources.json), [freeze](freeze.json), [results](results.json), [audit](audit.json), [post-run plaintext diagnostics](plaintext-diagnostics.json), [archive hashes](archive.json), [released source groups](released-source-ids.json).
- `python -m experiments.language_coverage_sources download` restores hash-pinned raw corpora.
- `python -m experiments.language_coverage verify` checks code, sources, models and committed freeze.
- `python -m experiments.language_coverage replay` repeats all completed transfer scores and decisions.
- `python -m experiments.audit_language_coverage` additionally rebuilds priors and reproduces encryption; its sealed output must not already exist.
- `evaluated-records.tar.gz` contains normalized model inputs, evaluated plaintext/ciphertext, keys, predictions and logs. Raw ReM and LLCT downloads and binary priors remain in artifacts; probability hashes support rebuilding priors.
- New-source attribution: Timo Korkiakangas, Flavio Massimiliano Cecchini and Marco Passarotti (LLCT); ReM 2.1 creators listed in archived Zenodo metadata; Afra Pujol i Campeny and Marieke Meelen (HisCat). LLCT/ReM text CC BY-SA 4.0; HisCat CC BY 4.0. Existing corpus licenses remain recorded in the prior source manifests.

**No Voynich text or reserved manuscript test was used.**
