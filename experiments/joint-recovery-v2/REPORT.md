# Codebook-free Naibbe recovery, round two: fresh evaluation

Frozen method commit: `699ab78e15618ce1a9d8e852627ef867a1dc0aef` ([freeze](freeze.json), [protocol](../joint-development-v2/PROTOCOL.md)).
Four fresh passages of 5,237–5,286 letters, none overlapping any earlier challenge, decoded from ciphertext only; references opened after predictions were saved.

**The gate still fails, but the Dante gap is closed.** Character error is 9.5% on modern Italian and
10.3% on Dante, against 12.5% and 33.5% in round one. Word error is 54% and 60%. The two changes were
usage pruning of the piece lexicon and a prior that includes Petrarca's verse.

**CER / WER / segmentation agreement / gate:** as in [round one](../joint-recovery/REPORT.md).

## What was done

- Froze the round-two decoder (joint EM, usage pruning at 3, joint EM, refinement) with the prose+verse
  prior, committed, prepared two ISDT-test and two Dante passages excluding all earlier source IDs.
- Decoded the four cases in 10 CPU minutes total; graded once; archived all records.

## Why

Round one identified two blockers: spurious candidate pieces capping segmentation at 85–91%, and a
prose-trained prior that could not model Dante. Development fixed both on held-out text; this run
tests whether the fixes transfer to unseen Dante passages, which are by a different author than the
verse used for fitting.

## Results

| Case | Letters | Words | CER | WER | Segmentation agreement | Pieces before → after pruning | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| modern | 5,237 | 961 | 10.7% | 59.5% | 89.4% | 697 → 265 | fail |
| modern | 5,258 | 981 | **8.3%** | 48.8% | 91.2% | 724 → 265 | fail |
| historical | 5,271 | 1,240 | **9.0%** | 56.9% | 90.3% | 754 → 265 | fail |
| historical | 5,286 | 1,330 | 11.5% | 63.3% | 88.4% | 693 → 268 | fail |

| Dataset | Round two CER | Round one CER | Round two WER | Round one WER |
|---|---:|---:|---:|---:|
| modern | **9.5%** | 12.5% | 54.1% | 58.0% |
| historical (Dante) | **10.3%** | 33.5% | 60.2% | 87.0% |

No case hit a cap. Development predicted 9.3–10.9% for the three development texts; the fresh
cases land at 8.3–11.5%.

1. **The verse prior transfers across authors.** Petrarca in training brought Dante from
   30–37% to 9–12%, the same band as modern prose, even though Dante never entered fitting.
2. **Pruning holds on fresh text.** Segmentation agreement is 88–91% on all four cases,
   up from 85–91%, with the candidate lexicon cut by more than 60%.
3. **Letters are still not words.** At 8–12% letter error roughly every ninth letter is wrong,
   which breaks about half of all words after segmentation. Sample, historical, best case:

> incontanenteintesiecerto**muicheeu**uestaeralasettadicattiviadiospiacentieanemicisuiquestiscia**no**ratichem
> incontanenteintesiecerto**fuicheq**uestaeralasettadicattiviadiospiacentieanemicisuiquestiscia**u**ratichemai

## What this establishes and what it does not

- A decoder given only ciphertext, the language, the alphabet and a declared token class recovers
  about 90% of letters of a 5,000-letter Naibbe passage in either modern or 14th-century Italian.
  The declared gate (CER ≤ 1%, WER ≤ 10%) is not met; the method is not validated for the
  benchmark's stated purpose.
- The remaining error is now spread between segmentation (about 10% of tokens still mis-parsed) and
  letter mapping within correctly parsed tokens. Neither dominates, so the next gain needs a
  stronger prior than characters, such as a word-level term in the refinement objective.
- Four passages; no significance claim. Procedural blinding on one machine. These passages are
  now disclosed; exclude their source IDs from future fresh tests.
- The Voynich mechanism test stays closed. Nothing here touches Voynich text or the final test.

## Records and reproduction

- [Frozen settings and hashes](freeze.json); [all per-case metrics](results.json); protocol and
  development record in `experiments/joint-development-v2/`; verse sources in `experiments/verse-prior/`.
- `evaluated-records.tar.gz` holds the public challenge, predictions, challenge hashes and evaluator
  answers with seeds and traces, released only after grading.

```sh
.venv/bin/python -m experiments.verse_sources restore          # unpack the pinned Canzoniere pages
.venv/bin/python -m experiments.joint_recovery_v2 verify
.venv/bin/python -m experiments.joint_recovery_v2 evaluate     # regrades the archived predictions
```

The prior file lives under `artifacts/verse-prior/` (about 50 MB); its hash is in `freeze.json` and it
is refitted deterministically by `experiments.joint_development_v2.fit_priors` from the pinned texts.
CPU only; the only download was pinned public-domain verse; no paid compute.
