# Historical language coverage: three-pair exploratory pilot

This is a coverage test of known Naibbe ciphertext, not a Voynich language search.
Freeze this protocol, code, source partitions, models and thresholds before encryption.

## Sources and representation

- Latin: LLCT2 Tuscan legal charters, AD 774–897, UD revision
  `df63d06c5a7788b457f50bf37526146e9225c27d`, CC BY-SA 4.0.
  Source: <https://universaldependencies.org/treebanks/la_llct/>.
  This broadens practical writing, not medical Latin or fifteenth-century coverage.
- German: ReM 2.1, <https://doi.org/10.5281/zenodo.13982324>, CC BY-SA 4.0,
  1050–1350. Restrict to prose (`classCode=P`). Use normalized surface `norm`,
  never lemma or modern German translation. Group manuscript variants by numeric M ID.
- Catalan: HisCat Llibre dels Fets, <https://doi.org/10.5281/zenodo.5615759>,
  CC BY 4.0, thirteenth-century chronicle. Strip POS annotations and folio markers.
  This single work cannot establish generalization to a new author or genre.
- Occitan deferred: an accessible historical Catalan corpus fills the planned Romance
  comparison without substituting a modern news corpus for a medieval source.

New text uses the existing Naibbe alphabet and normalization (`j→i`, `k→c`,
`w→uu`, accents removed), plus explicit `ſ→s`, `ß→ss`, `æ→ae`, `œ→oe`, `ð→d`.
These tests concern normalized text; the encoding is not a historical spelling claim.

Reconstruct whole Latin charters before splitting. For Latin and German assign
numeric work/document groups by SHA-256 bucket modulo 10: 0 challenge, 1 calibration,
2–9 training. Catalan uses the first 80% of complete folios for training, next 10%
for calibration, last 10% for challenge. Keep recto/verso together.
Split documents into 80-word chunks. Exclude calibration chunks sharing any normalized
eight-word sequence with training; exclude challenge chunks matching training or
calibration. Include cross-chunk windows in reference documents. Exclude overlap between
the two challenge passages and check both against all legacy training/calibration texts.
Check challenge joins against all three new source collections as well. Discard a chunk
if joining it to the retained text creates an eight-word match; verify full passage joins.
Removed chunks can make passages non-contiguous; record this limitation.

## Models and comparison

Fit eight order-5 character models, each on **exactly 400,000 letters**:
the existing five languages, Latin-broad, German-broad and Catalan.
Broad models contain 200,000 letters of the original source plus 200,000 of the new
historical corpus. Their entropy calibration similarly mixes 10,000 + 10,000 letters;
other models use 20,000. No challenge scores enter training or calibration.

Compare two fixed systems using the same predictions:

1. Baseline: original Latin, German, Old French, English, Italian, refit at 400,000 letters.
2. Expanded: replace Latin/German with their broad models and add Catalan; six languages.

The baseline is a matched-budget comparison, not the old 606,976-letter result.
Do not rank two Latin or two German models as separate languages, or pick whichever
model looks better on the challenge. For each true language, also remove that entire
language from the expanded candidate list and reapply the decision rule.

## Challenge and decisions

Three independent random keys: one Latin-charter pair, one historical-German-prose pair,
one Catalan pair. Each pair has a 5,200-letter fitting passage and a different
5,200-letter transfer passage, with disjoint source groups. Encrypt with the same key,
independent random seeds, and standard Naibbe spacing (RESPACING=17).

Use the existing transfer-centered development rule unchanged: fit and transfer winner
agree; each winning margin ≥0.25 bits/letter; transfer excess ≤0.50 bits/letter;
token coverage ≥95%; no compute cap. Seal fitted keys before any transfer.
No transfer refitting. A compute cap means inconclusive, never successful rejection.

Primary feasibility target: all three expanded positives correctly accepted and all three
omitted-language cases rejected, with no caps. Also report raw rankings, margins,
coverage, excess and true-prior character error. Continue all planned cases despite
decision failures to describe this fixed small pilot; stop on a compute cap or budget.
This is not a false-positive-rate estimate; omitted decisions reuse positive predictions.
There are no newly generated copying or shuffle controls in this pilot.

## Resources and reproducibility

Reuse the verified bounded incremental refiner: 200 sweeps, 20 million evaluations,
512 proposals per batch, 30 kicks, 1,200-second refinement cap, 3,600-second fit cap.
Four worker processes, two Numba threads each. At most 12 aggregate fit-worker hours,
reserving the worst-case eight-fit bundle before starting a pair. Three pairs × eight
models = 24 fits. Checkpoint every result without overwriting. Record all cap reasons.

```text
Audit sources and test extraction, splits and omission logic
Fit equal-budget models and freeze hashes, settings and thresholds
Commit the freeze
Encrypt and seal all three passage pairs
Fit eight models per pair and seal keys
Transfer each fixed key to the second passage
Evaluate baseline, expanded and omitted-language decisions
Replay transfers and archive inputs, predictions and provenance
```

Keep all earlier experiments unchanged. Do not access Voynich reserved text.
Report corpus period, genre, author and preprocessing limits even if every case passes.
