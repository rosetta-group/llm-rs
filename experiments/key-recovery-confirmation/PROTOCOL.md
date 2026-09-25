# Fresh confirmation of whole-token admission: 24 blocks, eight languages

Pattern: **sealed known-answer confirmation**. New works, new keys, one run, every failure kept.

Declared 2026-09-25, before any key or passage is drawn. The user approved 3 blocks per language and
new downloads for Latin, Old French, English, Italian and Catalan. Development evidence:
[five cases and controls](../key-recovery-development/REPORT.md).

**Block:** one hidden letter key, two 5,200-letter passages from two different works (fit, transfer).
**Transfer excess:** decoded bits per letter on the transfer passage with the sealed fit key, minus
the model's calibration score.
**A / B:** the fitter with / without whole-token admission and context reparse. B is an intermediate
state of A, so both come from the same fits.

## What will be done

```text
Freeze code, priors, pinned passages and this protocol; commit
For each of 24 blocks (3 rounds; each round has one block per language, in a fixed order):
    draw key and encoder seeds from the system RNG; pick at random which passage is fitted
    encrypt both passages; derive two negatives: token shuffle, frequency copy
    fit all three inputs under all eight priors (A and B); seal keys; decode transfer passages
Open answers once; score, report, archive; release every used work
```

## Fixed components

1. **Candidates:** the eight priors and calibration entropies of the language-expansion freeze
   (`4b23333`), unchanged.
2. **Decoder:** `experiments.key_recovery_development.fit_both`. Admission threshold
   $6 + \log_2(\text{types tested})$ bits; 3 admission rounds; 2 reparse rounds, width 128.
3. **Rule:** `decide_transfer` unchanged. Both margins at least 0.25 bits per letter, transfer excess
   at most 0.50, token coverage at least 0.95, same winner on fit and transfer, no cap.
4. **Negatives:** token shuffle as `rejection_transfer_v2.prepare`; frequency copy as
   `rejection_development.frequency_copy` (copy probability 0.8, window 50). Copy-mutate is left out:
   all 24 of its released development fits reached the frozen 20,000,000-proposal limit in both arms,
   so it cannot conclude.
5. **Caps:** 3,600 s for B and 7,200 s for A per fit. A cap makes that input inconclusive; no rerun.

## Sources

Pinned in [sources.json](sources.json) (125 raw files with sha256; partitions sha256 `d49734c9…`),
built by `experiments/key_recovery_confirmation_sources.py`. Works were fixed by date, genre, length
and licence, never by a model score. None is a training, calibration or earlier challenge work.
Every 80-word chunk sharing an 8-word shingle with earlier text or another selected work is removed;
each passage is the first 5,200 remaining letters. Blocks pair passages (1, 2), (3, 4), (5, 6).

| Language | Works (passage order) |
|---|---|
| Latin | Einhard, Vita Karoli; Navigatio Brendani; Liutprand, Antapodosis; Gesta Francorum; Petrus Alfonsi, Disciplina clericalis; Isidore, Etymologiae IV (medicine) |
| Italian | Fioretti; Passavanti, Specchio; Giamboni, Vizi e virtudi; Fiore di virtù; Caterina, Lettere; Bernardino, novellette |
| Catalan | Muntaner; Desclot; Llull, Llibre de les bèsties; Curial e Güelfa; Tirant lo Blanc; Pere IV, Crònica |
| Old French | Erec et Enide; Roman de Thèbes; Meraugis; Cléomadès; Huon de Bordeaux; Livre du roi Dancus |
| English | Joyce, Dubliners; GUM: Warhol news, walking essay, theropod article, Herrick interview, history textbook |
| German | ReM: Speculum ecclesiae; Berliner Evangelistar; Jenaer Martyrologium; Augsburger Stadtbuch; Wiggert psalm fragments; Nürnberger Urkunden |
| Czech | DIAKORP: Svatovítský rukopis; Traktáty a modlitby; Život Krista Pána; Životy svatých otců; Mastičkář; Praktika testamentu |
| Occitan | COMETA: Leys d'amors; Français 13504; Arsenal 6355; NAF 11180; second passages of Leys d'amors and Français 13504 |

Known limits, fixed now:
- **Occitan block 3 shares works** with Occitan block 1: only four clean COMETA works exist. Its second
  passages start at least 50,000 filtered letters later. It is reported separately.
- **Genre shifts:** Old French is mostly verse; English mixes a novel with short GUM documents; German
  includes a martyrology and charters. Some passages contain short Latin insertions (at most 2.8% of
  words in the Czech and German passages checked).
- **Catalan against Occitan** was the development case that failed on margin (0.24 < 0.25). It is
  not changed.

## Endpoints

Positives are the 24 true-language decisions; negatives are the 48 derived inputs.

1. **Safety (all must hold):** no wrong language accepted on any positive; no acceptance on any
   positive with its true language omitted; no negative accepted.
2. **Sensitivity:** A accepts the true language in at least 16 of 24 blocks.
3. **Paired improvement:** A accepts more true languages than B.
4. **Completeness:** all 24 blocks run and no input is inconclusive.

The confirmation passes only if all four hold. Every language is reported separately. With 48
negatives and none accepted, the one-sided 95% upper bound on the negative acceptance rate is
$1 - 0.05^{1/48} = 6.1\%$; blocks share languages and sources, so this bound is indicative only.

## Budget and stop rule

576 fits (24 blocks × 3 inputs × 8 priors). At the measured 328 s per fit that is about 52
fit-worker hours, about 11 wall hours on five workers. The worker budget is 150 hours; before each
block, 900 s is reserved per unfitted input-model pair. If the reservation would exceed the budget,
the run stops and the remaining blocks are inconclusive. A safety failure does not stop the run.

## What a pass would and would not mean

A pass supports using A to rank and accept these eight candidate languages for this cipher family.
It does not open the reserved Voynich pages: the manuscript recovery gate (1% CER and 10% WER per
case) is unchanged, and no Voynich language or word follows from known-answer controls.
