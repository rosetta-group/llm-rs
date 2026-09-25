# Etruscan round three: repair helps the inputs; abstention mostly selects names

Run 2026-09-24 against the local pre-run [freeze](freeze.json), under the unchanged
[protocol](PROTOCOL.md). **Both methods fail the continuation gate.** This is an
internal evaluation on known vocabulary, not an independent decipherment result.

## What was done

- Audited source labels and a deterministic sample of texts; repaired empty-first-row
  dictionary loss and `kh`/`ch` spelling duplicates. Quarantined ambiguous word
  divisions and damaged records. Details: [AUDIT.md](AUDIT.md).
- Evaluated context alone and context plus endings, with abstention calibrated in
  inner folds. Related forms stayed in one outer fold. The old 49-word test and
  connected relatives supplied no seed, calibration, or evaluation labels.
- Completed five outer folds, 99 shuffled-label controls per method, and 1,000
  paired family-bootstrap replicates. Saved predictions, thresholds, splits and
  controls in [results.json](results.json). Seven focused regression tests pass.

## Why

The previous negative result used a dictionary loader that silently discarded
usable labels. This follow-up tests a repaired, more conservative pipeline and
asks whether refusing weak predictions yields useful accuracy beyond names.

## Results

427 labelled types, 303 blocking families, 260 ETP texts, 1,154 tokens.

| Measure | Context | Context + endings |
|---|---:|---:|
| Accepted predictions | 72 / 427 | 192 / 427 |
| Coverage | 16.9% | 45.0% |
| Correct among accepted | 61 / 72 | 164 / 192 |
| Accepted precision | 84.7% | 85.4% |
| 95% family-bootstrap interval | 74.1–92.4% | 78.2–91.4% |
| Majority guess on exactly those accepted items | 84.7% | 81.2% |
| Advantage over that majority guess | 0.0 points | 4.2 points |
| Non-NAME calls correct | 1 / 4 | 10 / 15 |
| Full balanced accuracy, including abstained items' raw guesses | 32.5% | 28.1% |
| Full-classifier shuffled-label p | 0.01 | 0.01 |
| Continuation gate | **fail** | **fail** |

Balanced accuracy averages class recalls; ordinary precision here counts correct
accepted calls. The two metrics answer different questions. Shuffled-label means
were 19.7% and 20.2% full balanced accuracy. These p-values establish neither the
reliability of a particular accepted meaning nor independent historical evidence.

### Why the apparently good 85% is insufficient

The endings model accepts 177 NAME calls (154 correct) and 15 OTHER calls (10
correct). It accepts **zero KIN, NUM, or LIFE calls**. For example, `puia` is
confidently called NAME, while `clen`, `lupuce`, and `svalce` are called OTHER.
These are held-out known words, not proposed new translations.

Endings expand the accepted subset substantially, but the frozen test does not
show an improvement in full class discrimination: the balanced-accuracy difference
is **−4.4 points**, with a paired family-bootstrap interval of **−18.8 to +2.3**.
This does not prove endings are intrinsically unhelpful; it fails the preregistered
criterion for calling this implementation an improvement.

The endings model misses two gate conditions: >=5-point advantage over the same-
subset majority guess (actual 4.2), and >=70% precision on non-NAME calls (actual
66.7%). Context also misses the coverage and non-NAME support requirements.
The near misses do not justify changing the thresholds after seeing the results.
More fundamentally, the accepted labels provide no positive kinship, numeral, or
life/death identifications.

## What the old model does on the repaired test

The unchanged M2 predictor reaches **49.8% balanced accuracy** on these new grouped
splits, but only **47.8% ordinary accuracy**. Its KIN precision is 8/84 (9.5%),
NUM precision 2/31 (6.5%), and LIFE precision 5/32 (15.6%). This descriptive bridge
again shows why balanced accuracy alone was an unsuitable permission to publish
individual meanings. It is not a controlled estimate of the dictionary repair's
effect: corpus, labels and evaluation splits changed together.

## What this changes about round two

The old loader used the first row of each normalised spelling, even if empty.
Later real labels for `larth`, `vel`, `arnth` and `lautni` were silently lost.
Consequently, “not labelled by this loader” did not always mean “not glossed in
ETP”, and low labelled-neighbour coverage was partly a software/data-handling
problem. `lautni`, one of round two's purported fresh items, already has an ETP
gloss in row 869. The original run remains reproducible, but that interpretation
needs qualification. The wider scholarly coverage limit was not established by
the old experiment alone.

This new run fixes that handling in a separately versioned module. Earlier frozen
modules, sources, scores and reports remain untouched. Suspicious upstream suffix
flags (`apa`, `ati`) remain unresolved and are explicitly excluded.

## Decision and limits

```text
Keep the source audit, loader repair, and abstention implementation
Stop this tested classification pipeline under its frozen continuation rule
Do not generate unknown-word meaning claims
Require independent labels and reviewed word divisions before a new validation
```

The result supports Claude's practical decision to withhold predictions. It does
not support declaring Etruscan research exhausted. This study has only four NUM
types, ten LIFE types, and sixteen KIN types; it aggressively filters to cleaner
known vocabulary, has no new Latin benchmark, and uses conservative groups that
can merge unrelated words. It remains an exploratory follow-up on familiar data.

## Validation and reproduction

- Seven tests cover row-order repair, conflicting/uncertain labels, family-disjoint
  folds, same-family neighbour leakage, evidence-free abstention, ending features,
  and calibration support/quality.
- All code and source hashes matched the pre-run freeze. The final result records
  that freeze's SHA-256. The run refuses modified inputs and existing results.
- Evaluation outputs are retained; consult [PROTOCOL.md](PROTOCOL.md) for commands.
  Reproduction in a separate copy requires retaining the freeze while moving only
  the saved `results.json` out of the runner's output path. No further fitting or
  parameter changes were made after this result.
