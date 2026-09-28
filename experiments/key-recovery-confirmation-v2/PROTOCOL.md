# Second fresh confirmation: broadened Latin, German and Catalan priors, 24 blocks

Pattern: **sealed known-answer confirmation**. New works, new keys, one run, every failure kept.

Declared 2026-09-28, before any key or passage is drawn. The user approved the downloads. Development on the released round one gave 19/24 with no false acceptance. Round one failed on sensitivity
(13/24) because three priors were too narrow for other genres of their own language
([round one](../key-recovery-confirmation/REPORT.md)). This round tests the rebuilt priors
([development](../key-recovery-v2-development/PROTOCOL.md)).

**Block:** one hidden letter key, two 5,200-letter passages from two different works (fit, transfer).
**Transfer excess:** decoded bits per letter on the transfer passage with the sealed fit key, minus
the model's calibration score.
**A / B:** the fitter with / without whole-token admission and context reparse, as in round one.

## What will be done

```text
Freeze code, the eight priors, pinned passages and this protocol; commit and push
For each of 24 blocks (3 rounds; each round has one block per language, in a fixed order):
    draw key and encoder seeds from the system RNG; pick at random which passage is fitted
    encrypt both passages; derive two negatives: token shuffle, frequency copy
    fit all three inputs under all eight priors (A and B); seal keys; decode transfer passages
Open answers once; score, report, archive; release every used work
```

## Fixed components

1. **Candidates:** `latin_broad2`, `german_broad2` and `catalan_broad2` (400,000 training letters over
   many works; calibration from two held-out works), plus the five unchanged priors of the
   language-expansion freeze (`4b23333`). Hashes and calibration entropies in
   [priors.json](../key-recovery-v2-development/priors.json).
2. **Decoder, rule, negatives and caps:** identical to round one (`443098f`). `decide_transfer`
   unchanged: both margins at least 0.25, transfer excess at most 0.50, coverage at least 0.95, same
   winner on fit and transfer, no cap.

## Sources

Pinned in [sources.json](sources.json), built by `experiments/key_recovery_v2_sources.py`. Every role
comes from a sha256 order of author/text groups, never from a model score. No test work, author or
text family appears in any prior's training or calibration, in round one, or in earlier experiments.

| Language | Test works (passage order) |
|---|---|
| Latin | Abelard, Historia calamitatum; Annales Bertiniani (Hincmar); Legenda aurea; Poggio, Facetiae; Dante, Monarchia; Salimbene, Cronica |
| Italian | Masuccio, Novellino; Libro di Sidrach; Bruni, Guerra punica; Leonardo, Frammenti; second passages of Sidrach and Leonardo |
| Catalan | Eiximenis, Regles; Perellós, Purgatori; Art de bé morir; Paris e Viana; Alcanyís, Regiment; Filla del rey d'Hongria |
| Old French | Gui de Bourgogne; Lai de l'Ombre; Benoît, Troie; Otinel; Villehardouin; Floovant |
| English | Russell; Anderson; Cather; Keynes; Dewey; Du Bois (Project Gutenberg, 1903–1922) |
| German | ReM: Litanei; Pilatus; Iwein; Himelrîche; Gottfried, Tristan; Der wilde Mann |
| Czech | Chelčický, O trojiem lidu; Milíč prayer book (diakorp29); Hus, Dcerka; DIAKORP 1552, 1580, 1585 |
| Occitan | Daurel e Beton; Girart de Rossilho (Oxford); Esquerrier; Breviari d'amor; Costumas de Seix; Jaufre |

Known limits, fixed now:
- **Italian block 3 shares works** with blocks 1 and 2: only four eligible Italian works had a clear
  licence. Its passages start at least 50,000 filtered letters later. Reported separately.
- **Czech blocks 2 and 3 are later** (1552–1585) than the Czech prior (1350–1460); Hus uses
  modernised spelling.
- **Occitan and Latin test texts include uncorrected OCR** from public-domain editions (Breviari,
  Daurel, Jaufre); German test texts are verse, which the new German prior now covers.

## Endpoints

Positives are the 24 true-language decisions; negatives are the 48 derived inputs.

1. **Safety (all must hold):** no wrong language accepted on any positive; no acceptance on any
   positive with its true language omitted; no negative accepted.
2. **Sensitivity:** A accepts the true language in at least 16 of 24 blocks.
3. **Completeness:** all 24 blocks run and no input is inconclusive.

The confirmation passes only if all three hold. A against B is reported but is not a pass condition:
round one already confirmed on fresh text that A accepts more true languages than B (13 against 8),
and this round tests the priors. Every language is reported separately.

## Budget and stop rule

576 fits. At the measured 300–330 s per fit that is about 50 fit-worker hours, about 10–11 wall
hours on five workers. Worker budget 150 hours; 900 s reserved per unfitted input-model pair
before each block. If the reservation would exceed the budget, the run stops and the remaining
blocks are inconclusive. A safety failure does not stop the run.

## What a pass would and would not mean

A pass supports using A with these eight priors to rank and accept candidate languages for this
cipher family. It does not open the reserved Voynich pages: the manuscript recovery gate (1% CER and
10% WER per case) is unchanged, and no Voynich language or word follows from known-answer controls.
