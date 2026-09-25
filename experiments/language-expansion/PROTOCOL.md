# Old Czech and Old Occitan: eight-language extension

Pattern: **language coverage test**. Test two new historical candidates against the six-language system.
The preregistered feasibility target is correct acceptance on both fresh pairs, and conclusive rejection when each true language is omitted.

**Transfer:** decoding a second passage using only the first passage's sealed key.
**Excess:** decoded bits per letter minus a separate calibration score for that model.

## What will be done

- Add Old Czech and Old Occitan to Latin, German, Old French, English, Italian and Catalan.
- Preserve the broad Latin/German models and all six earlier models' exact inputs.
- Train each new order-five character model on 400,000 letters: 100,000 from each of four works. Calibrate on 20,000 letters: 10,000 from each of two different works.
- Use two distinct held-out source works per new language, 5,200 letters for fitting and 5,200 for transfer. One independent random key per language; 16 fits total.
- Compare six-language baseline, eight-language expansion and seven-language true-language-omitted decisions on the same predictions.

## Why

Old Czech adds the first Slavic candidate and period medical material. Old Occitan adds a medieval Romance competitor to Catalan and Old French using a separately licensed corpus.

```text
Pin sources and source-work roles
Exclude earlier released passages and overlapping eight-word spans
Freeze models, settings, code and thresholds; commit
Encrypt two new passage pairs using independent hidden keys
Fit eight models per pair; seal all fitted keys
Decode second passages without updates
Open answers once; evaluate included and omitted language decisions
Archive results and release complete source groups for future exclusion
```

## Fixed source choices

The machine-readable roles and exact titles are in `sources.json`. Selection used dates, genres and text availability, never model scores.

1. **Czech:** DIAKORP v5, distributed by HistCorp in Zenodo record 10013189, CC BY-NC-SA 4.0. Training: medicine (1440–1460), Pasionál (1350–1400), Pulkava chronicle (1400), discourses (1389–1401). Calibration: translated prophets (1380–1400), Jerome (1410). Challenge: Hvězdářství krále Jana (1440–1460, includes medicine), travel to Jerusalem/Egypt (1492). Exclude all other DIAKORP dates, modern editorial metadata and omission markers. Czech source dates are corpus metadata, not new manuscript dating.
2. **Occitan:** Marinus Wiedner, COMETA v1, Zenodo 15300719, CC BY 4.0. Training manuscripts: Français 1049, Français 13503, BmC 34, Français 25425. Calibration: NAF 6195 and Français 2232. Challenge: NAF 11151 and Harley 7403. Other copies of Honorat and Philomena are unused, so parallel versions cannot leak between roles. Remove the French editorial heading in Français 1049, join explicit line-end hyphenation, and remove omission markers. Historical verse and prose are mixed. The challenge is taken from the start of each filtered manuscript; it is not a guaranteed medical excerpt.
3. **Representation:** unchanged 21-letter Naibbe alphabet and historical normalization. Czech accents are removed, j→i, k→c, w→uu; this collapses real Czech distinctions. DIAKORP is transcribed, not diplomatic spelling (the provider's documentation overrides HistCorp's generic header). COMETA expands abbreviations and distinguishes u/v; manually corrected HTR can retain errors. Foreign quotations are not consistently tagged in the plain text exports.
4. **Overlap:** whole works separate new training, calibration and challenge. Filter challenge eight-word matches against both languages' complete train/calibration pools, earlier legacy train/calibration, and previous historical collections. Exclude prior released challenge shingles before selecting any new model or test text. Recheck final joins. Filtering can make excerpts non-contiguous. No score-driven replacement is permitted.

## Frozen algorithm and decision

Use `rejection_followups_v2.fit_key` unchanged, four restarts and the previous EM configuration, incremental refinement with 200 sweeps, 20,000,000 proposals, batch 512, 30 kicks. Fit cap 3,600 seconds, refinement cap 1,200 seconds, four workers with two Numba threads, 12 aggregate fit-worker hours. Reserve a complete worst-case bundle before starting each pair. Any cap makes the affected comparison inconclusive; no threshold tuning or extra restart after looking at results.

Use `decide_transfer` unchanged: same winning language for fit and transfer; both margins at least 0.25 bits/letter; transfer excess at most 0.50; token coverage at least 0.95; no caps. The true language's entire candidate is removed for the omitted test. Report CER (edit distance / 5,200) for each true model and all scores, margins, gate reasons and timings.

The solver receives ciphertext and the eight priors, with the shared alphabet and encoder family. It receives no source language, work identity, plaintext, encoder seed, key or trace. Procedural blinding is on one machine, not an independent evaluator; the source curator necessarily sees plaintext during preparation.

## Scope and stop rule

Exactly two new-language cases. No fresh old-language retention claim, prevalence estimate or Voynich inference follows from this pilot. Previously evaluated cases remain development only. No reserved Voynich pages are used. Preserve failed gates. Post-run true-plaintext scores may diagnose model mismatch versus key error, but cannot change this run.

## Source notes

The Vokabulář editions were inspected but not imported because their terms restrict redistribution. Padeřov Bible ground truth was downloaded during source discovery but is not used. DIAKORP provides broader licensed period coverage. Unused downloads are outside the frozen input manifest.
