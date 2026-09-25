# Old Czech and Old Occitan extension

Pattern: **language coverage test**. Compare six and eight candidate languages on two fresh historical-source ciphertext pairs.

The eight-language system accepted 1/2 correct languages; removing the true language gave 2/2 conclusive rejections.

**Transfer:** decode a different work with the fitted key held fixed.
**CER:** character edit distance divided by the 5,200 reference letters.
**Excess:** decoded bits per letter minus that model’s calibration score.

## What was done

- Added Old Czech (DIAKORP/HistCorp) and Old Occitan (COMETA); six existing models reproduce exactly.
- Trained each new model on 400,000 letters across four works; calibration uses 20,000 letters across two other works.
- Compared Latin, German, Old French, English, Italian, Catalan, Old Czech and Old Occitan.
- Completed 16/16 fits in 0.966 aggregate worker-hours; 194 tests passed.
- Re-extracted 2538 source rows, rebuilt eight priors, replayed four encryptions and 16 transfers.

## Why it was done

Czech supplies the first Slavic candidate and historical medical material. Occitan tests a medieval Romance alternative close to the existing Catalan and Old French candidates.

```text
Freeze sources, models and existing rule; commit
Encrypt two independent-key pairs
Fit eight models; seal keys; decode different works
Compare six, eight and true-language-omitted candidate sets
Open answers once; preserve all failures
```

## Results

| True language | Six-language accepted | Eight-language fit / transfer winners | Eight-language accepted | True language omitted |
|---|---|---|---|---|
| czech | none | czech / czech | czech | none |
| occitan | none | occitan / occitan | none | none |

The frozen feasibility target **failed**. Stop reason: `None`.

| True model | Fit CER | Transfer CER | Fit excess | Transfer excess | Coverage | Fit / transfer margin |
|---|---:|---:|---:|---:|---:|---:|
| czech | 4.79% | 5.54% | 0.165 | 0.339 | 99.19% | 0.555 / 1.247 |
| occitan | 12.06% | 15.19% | 0.191 | 0.525 | 97.44% | 0.489 / 1.035 |

Gate reasons (unchanged thresholds):

- czech / baseline: fit_margin, transfer_margin, transfer_excess.
- czech / expanded: all gates passed.
- czech / omitted: fit_margin, transfer_margin, transfer_excess.
- occitan / baseline: transfer_excess.
- occitan / expanded: transfer_excess.
- occitan / omitted: transfer_excess.

## Post-run diagnostic

| Source | Calibration bits/letter | True transfer bits/letter | True transfer excess | Recovered transfer bits/letter |
|---|---:|---:|---:|---:|
| czech | 3.219 | 3.267 | 0.048 | 3.558 |
| occitan | 3.416 | 3.277 | -0.139 | 3.941 |

These true-plaintext scores were calculated after grading, never used to choose models or thresholds. A low true-plaintext excess with a high recovered-text excess points to key recovery; a high true-plaintext excess also exposes source/genre mismatch.

## What the result supports

1. **Czech is a useful addition in this control.** It wins on both works and passes acceptance with 5.54% transfer character error. The earlier six-language set ranks Old French first but correctly rejects it.

2. **Occitan remains a failed acceptance case.** It wins both rankings, but 15.19% transfer character error raises excess to 0.525, above the frozen 0.500 ceiling. The small 0.025 miss is still a failure; neither the threshold nor the passage was changed.

3. **The next bottleneck is recovery on these examples.** Correct Occitan transfer plaintext scores at −0.139 excess, so recovery adds 0.664 bits per letter. Develop key recovery on these now-released cases, then require new works and keys for confirmation. The two omitted-language rejections are controls, not an estimated false-acceptance rate.

## Source interpretation and limits

1. **Czech is historical, but spelling is normalized.** [DIAKORP](https://wiki.korpus.cz/doku.php/en:cnk:diakorp) transcribes historical forms. The selected work dates range from 1350–1400 to 1492; the fixed alphabet additionally removes accents and merges j/i and k/c. Training includes Lékařství neznámého františkána; testing uses Hvězdářství krále Jana and the 1492 travel account. No modern Czech newspaper material is used.

2. **Occitan supplies different works.** [COMETA](https://zenodo.org/records/15300719) provides medieval Provence/Languedoc transcriptions, manually corrected after handwriting recognition. The two challenge manuscripts are NAF 11151 and Harley 7403; their opening excerpts are not a dedicated medical test. Multiple copies of Honorat and Philomena are kept out of other roles. Verse, prose, spelling and untagged foreign quotations remain mixed.

3. **Two cases do not validate eight languages generally.** Each new language has one independent key, with fit and transfer from different works. No fresh retention controls for the existing six languages were run. A correct winner is weaker than passing the fixed acceptance rule. No reserved Voynich text was used and no Voynich language was identified.

4. **Source independence is filtered and finite.** No selected challenge passage has an eight-word overlap with the audited references or earlier released passages. Filtering can remove material and create non-contiguous excerpts; different works can still share genre or translated traditions. Procedural blinding on one computer is not an independent evaluation.

## Selected works

| Language | Work / manuscript | Role | Source date label | Selected letters |
|---|---|---|---|---:|
| czech | Cesta z Čech do Jeruzaléma a Egypta (`diakorp1-1492`) | challenge | 1492 | 5,200 |
| czech | Hvězdářství krále Jana (`diakorp14-1440-1460`) | challenge | 1440--1460 | 5,200 |
| czech | Lékařství neznámého františkána (`diakorp27-1440-1460`) | train | 1440--1460 | 100,000 |
| czech | Pasionál muzejní (`diakorp39-1350-1400`) | train | 1350--1400 | 100,000 |
| czech | Překlad proroků Izaiáše, Jeremiáše, Daniela (`diakorp46-1380-1400`) | calibration | 1380--1400 | 10,000 |
| czech | Pulkavova Kronika králů českých, (`diakorp47-1400`) | train | 1400 | 100,000 |
| czech | Řeči besední (`diakorp50-1389-1401`) | train | 1389--1401 | 100,000 |
| czech | O svatém Jeronýmovi knihy troje (`diakorp72-1410`) | calibration | 1410 | 10,000 |
| occitan | Roman de Flamenca (`BmC-34`) | train | medieval (COMETA; manuscript dates not homogenized) | 100,000 |
| occitan | Robert of Sicily; Libre de vicis et de vertutz; Barlam et Josaphas (`Français_1049`) | train | medieval (COMETA; manuscript dates not homogenized) | 100,000 |
| occitan | Vida de santa Doucelina (`Français_13503`) | train | medieval (COMETA; manuscript dates not homogenized) | 100,000 |
| occitan | Roman de Philomena (P) (`Français_2232`) | calibration | medieval (COMETA; manuscript dates not homogenized) | 10,000 |
| occitan | Chanson de la Croisade contre les Albigeois (`Français_25425`) | train | medieval (COMETA; manuscript dates not homogenized) | 100,000 |
| occitan | Nicodemus; fifteen signs; Cross; dietetics; repentance; doctrinal (`Harley_7403`) | challenge | medieval (COMETA; manuscript dates not homogenized) | 5,200 |
| occitan | Arbitral sentences; Mulomedicina; recepta del vi (`NAF_11151`) | challenge | medieval (COMETA; manuscript dates not homogenized) | 5,200 |
| occitan | Vida de sant Honorat (M) (`NAF_6195`) | calibration | medieval (COMETA; manuscript dates not homogenized) | 10,000 |

Each challenge work contributes one passage. Date labels come from corpus metadata; COMETA manuscript dates are not claimed to be a single contemporaneous window.

## Records and reproduction

- [Protocol](PROTOCOL.md), [sources](sources.json), [freeze](freeze.json), [results](results.json), [audit](audit.json), [diagnostics](plaintext-diagnostics.json), [archive hashes](archive.json), [released works](released-source-ids.json).
- Freeze commit: `4b23333b1ee614f6a815c325598f583c8f570a72`.
- `python -m experiments.language_expansion_sources download` restores pinned new inputs.
- `python -m experiments.language_expansion verify` checks the committed freeze and input hashes.
- `python -m experiments.language_expansion replay` repeats all transfer predictions and decisions.
- `python -m experiments.audit_language_expansion` rebuilds models and encryption; sealed audit outputs must not already exist.
- The archive contains model partitions, evaluated ciphertext/answers/predictions, logs and attribution. Earlier frozen dependencies remain required; see [reproduction instructions](../../docs/REPRODUCE.md).

## Attribution

Czech: Karel Kučera and Martin Stluka (2011), DIAKORP v5; Eva Pettersson and Beáta Megyesi (2018), HistCorp. [Distribution and license](https://sprakbanken-clarin.lingfil.uu.se/histcorp/readme/czech-readme-diakorp): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Occitan: Marinus Wiedner (2025), COMETA v1, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Normalized excerpts are adaptations; the original source licenses continue to apply. Earlier-language source licenses and authors remain in the previous manifests and the archived ATTRIBUTION.md.
