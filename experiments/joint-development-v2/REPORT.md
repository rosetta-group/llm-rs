# Codebook-free Naibbe recovery, round two: development record

Development results on development text only. The [protocol](PROTOCOL.md) fixes the two changes
tested here; the fresh evaluation is reported in `../joint-recovery-v2/REPORT.md` once run.

Two levers from round one were tested. Pruning the piece lexicon by expected usage lowers
character error by about a quarter on every text. Adding Petrarca's verse to the prior turns
historical verse from unrecoverable into recoverable at the same level as prose, without
hurting prose or modern text.

**Usage pruning:** after a first joint EM, drop candidate pieces whose expected use under the parse
posteriors is below a threshold, rebuild the parses, run EM again.
**Verse prior:** the order-5 character prior refitted on modern ISDT, Novellino/Decameron prose and
Petrarca's Canzoniere training poems (weight 3), Dante excluded.
**Oracle EM:** the mapping search with the true segmentation supplied, a diagnostic separating the
prior's adequacy from segmentation quality.

## What was done

- Added `voynich/joint_segments_v2.py`: a joint EM that accepts a candidate lexicon and reports
  expected piece usage, plus `prune_and_rerun`. Round one's module is untouched and still verifies.
- Pinned Petrarca's Canzoniere from Wikisource (366 poems, 41,844 training words) with revisions
  and hashes in `../verse-prior/sources.json`; every fifth poem is development.
- Measured pruning thresholds, a second pruning round, and old versus verse priors on
  5,200-letter development passages of prose, modern text and Petrarca.
- Wrote a reproducible driver (`experiments/joint_development_v2.py`) whose table is `results.json`.

## Why

Round one left segmentation at 85–91% agreement and Dante at 30–37% error against 10–15% for
prose. The first is a lexicon problem: too many spurious candidate pieces survive the frequency
threshold. The second is a prior problem: the prose prior assigns Dante 3.5 bits per letter
where prose costs 2.8.

## Usage pruning

Exploratory runs, old prior, 5,200 letters unless stated.

| Text | No pruning | Usage ≥ 1 | Usage ≥ 3 | Usage ≥ 6 | Second round at 3 |
|---|---:|---:|---:|---:|---:|
| historical prose | 14.2% (87.1% agreement) | 13.6% | **10.4%** (89.8%) | 11.7% | 10.8% |
| modern | 12.2% (88.5%) | | **9.4%** (90.4%) | | 10.0% |
| historical prose, 10,400 letters | 15.3% (84.6%) | | **11.3%** (87.5%) | | 10.7% |

1. **Threshold 3 is the sweet spot.** It removes about 60% of candidate pieces (646 to 273 on
   historical prose) and raises segmentation agreement by three points. Threshold 6 over-prunes.
2. **One round is enough.** A second pass changes fewer than ten pieces and moves error by
   under a point in either direction; at 10,400 letters it helps slightly, at 5,200 it hurts slightly.

## Verse prior

Petrarca development poems, 5,200 letters, no pruning.

| Prior | Bits per letter on the true text | Oracle EM | Oracle EM + refine | Codebook-free final |
|---|---:|---:|---:|---:|
| prose only (round one) | 3.50 | 76.2% | 76.4% | 53.4% |
| prose + verse, weight 1 | 2.95 | 13.3% | 3.7% | 12.9% |
| prose + verse, weight 3 | 2.85 | 10.5% | 3.6% | **13.1%** |

Regression on the other texts with the weight-3 prior: historical prose 14.4% (was 14.0%),
modern 11.9% (was 12.2%). Segmentation agreement is identical across priors (87.2% on Petrarca),
so the whole difference is in letter mapping, as the diagnosis predicted.

1. **The old prior did not merely underperform on verse; it failed.** Even with the true
   segmentation, EM found nothing at 5,200 letters. Verse has a different letter distribution
   from prose, elisions and archaic forms included.
2. **Weight matters little.** Weights 1 and 3 give the same recovery; 3 is kept because it costs
   fewer bits on verse and nothing on prose.
3. **Petrarca is not Dante.** This shows the prior can be fixed for a verse author it was
   trained on. The fresh evaluation on Dante measures how much transfers across authors.

## Reproducible table

`results.json`, driver settings (threshold 6, pruning at usage 3, four restarts, 60 iterations), 5,200 letters each.

| Text | Prior | Bits/letter on true text | Before pruning | After pruning | Segmentation agreement after pruning | Pieces kept |
|---|---|---:|---:|---:|---:|---:|
| historical | prose | 2.85 | 14.2% | **10.4%** | 89.8% | 646 → 273 |
| historical | prose+verse | 2.84 | 14.4% | **10.9%** | 89.9% | 646 → 271 |
| modern | prose | 2.66 | 12.1% | **9.4%** | 90.4% | 655 → 263 |
| modern | prose+verse | 2.71 | 11.9% | **9.3%** | 90.4% | 655 → 262 |
| verse | prose | 3.50 | 53.4% | **43.5%** | 89.7% | 653 → 262 |
| verse | prose+verse | 2.85 | 13.1% | **9.5%** | 89.7% | 653 → 261 |

With both changes the three development texts land at 9.3–10.9% character error. The frozen round-two method uses the prose+verse prior and one pruning pass.

## Limits

- All development text is 5,200 letters from one contiguous start; no variation over starting points.
- The word segmenter was not retrained on verse; word error on Dante remains limited by it.
- These are development passages that the settings were chosen on. Only the frozen fresh run counts.
- No Voynich text was touched; final test sealed; CPU only; the only download was pinned public-domain verse.
