# Codebook-free Naibbe recovery, round four: fresh evaluation

Frozen method commit: `b4648f65` ([freeze](freeze.json), [protocol](../joint-development-v4/PROTOCOL.md)).
Four fresh Dante passages of 5,213–5,263 letters, none overlapping any earlier challenge, decoded
from ciphertext only; references opened after predictions were saved.

**Character error on Dante halves, from 10.5% to 5.7%; the stop rule is passed and the gate still
fails.** Word error falls from 56% to 45%. The only change was the lexicon repair between the
second joint EM and the refinement. Parse agreement rises from 90% to 93% on every case, exactly
as on development text, and letters inside correctly parsed tokens stay at 0.3–1.4% error.

**CER / WER / segmentation agreement / gate:** as in [round one](../joint-recovery/REPORT.md).
**Lexicon repair:** dropping whole pieces whose count matches a prefix+suffix bigram under the
decoded key, and admitting the complement of a known half for tokens that have no parse
([protocol](../joint-development-v4/PROTOCOL.md)).
**No modern passages:** the ISDT test split holds fewer than 1,000 fresh letters after three
rounds, and no new corpus was pinned. The modern half of this round rests on development text.

## What was done

- Froze the round-four decoder (round two pipeline, lexicon repair twice, refinement, polish),
  committed, prepared four Dante passages excluding all source IDs used by rounds one to three.
- Decoded the four cases in 22 CPU minutes total; graded the polished and pre-polish letters once;
  measured parse agreement before and after the repair; archived all records.

## Why

Development found the joint EM reaches 97% agreement and 0.5% error when given the true lexicon,
and that the frequency-based candidate lexicon admits frequent bigram strings as one-letter
pieces while missing rare-letter pieces. The repair addresses both from the ciphertext alone.
The protocol declared a stop rule: Dante below 8% or the track closes.

## Results

| Case | Letters | Words | CER polished | CER before polish | WER polished | Agreement after repair | Agreement before repair | Letter error inside correct parses | Letters in mis-parsed tokens | Pieces: candidates → pruned → repaired | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Dante | 5,218 | 1,238 | **4.4%** | 4.5% | 38.0% | 93.6% | 90.3% | 0.3% | 7.0% | 697 → 278 → 319 | fail |
| Dante | 5,223 | 1,241 | 5.2% | 5.8% | 44.7% | 93.6% | 90.4% | 0.7% | 7.0% | 666 → 267 → 314 | fail |
| Dante | 5,263 | 1,287 | 6.3% | 6.9% | 50.1% | 93.5% | 90.2% | 1.4% | 7.2% | 720 → 282 → 363 | fail |
| Dante | 5,213 | 1,253 | 7.0% | 7.6% | 46.0% | 91.0% | 87.6% | 0.5% | 9.4% | 688 → 268 → 291 | fail |

| Dataset | Round four | Round three | Round two | Round one |
|---|---:|---:|---:|---:|
| Dante CER | **5.7%** | 10.5% | 10.3% | 33.5% |
| Dante WER | **44.8%** | 55.8% | 60.2% | 87.0% |

No case hit a cap. Development predicted 4.5–5.7% after polish for the three development texts;
the fresh Dante cases land at 4.4–7.0%. The repair's own record on fresh text matches development:
10–18 concatenation-like whole pieces dropped and 94–136 complements admitted in the first pass,
51–86 low-usage pieces pruned and 5–57 complements admitted in the second.

1. **The repair transfers.** Parse agreement gains 3.3–3.4 points on every case, the same as on
   development text, with no tuning on these passages.
2. **The letter mapping was never the problem.** Inside correctly parsed tokens the error is
   0.3–1.4%, as in rounds three and four development. Mis-parsed tokens now hold 7–9% of the
   letters, down from 10–14%.
3. **Words follow letters.** Word error falls 11 points; at 6% letter error about one word in
   two is still broken by a wrong letter or a wrong split. Sample, best case (true above,
   recovered below):

> contramieiinciascunasualeggeondioaluilostra**zi**oelgrandescempiochefecelar**b**iacoloratainrosso
> contramieiinciascunasualeggeondioaluilostra**fo**elgrandescempiochefecelar**d**iacoloratainrosso

## What this establishes and what it does not

- A decoder given only ciphertext, the language, the alphabet and a declared token class recovers
  about 94% of letters of a 5,200-letter Naibbe passage in 14th-century Italian. The declared
  gate (CER ≤ 1%, WER ≤ 10%) is not met.
- The track stays open under its own rule (Dante below 8%). The remaining gap to the true-lexicon
  oracle is about 4 points of agreement and 5 points of character error. Development attributes
  it to tokens split at the wrong point between two known pieces (about 100 per text) and to true
  one-letter tokens that the concatenation test or the EM still splits.
- Four Dante passages only; no fresh modern passage exists in the pinned corpora. Adding a
  modern half needs a newly pinned corpus and a declared protocol. Four passages; no
  significance claim. Procedural blinding on one machine. These passages are now disclosed;
  exclude their source IDs from future fresh tests.
- The Voynich mechanism test stays closed. Nothing here touches Voynich text or the final test.

## Records and reproduction

- [Frozen settings and hashes](freeze.json); [all per-case metrics](results.json), including
  pre-polish grades, pre-repair agreement and the repair record; protocol and development record
  in `experiments/joint-development-v4/`.
- `evaluated-records.tar.gz` holds the public challenge, predictions (polished and pre-polish
  letters, repaired and unrepaired parses), challenge hashes and evaluator answers with seeds and
  traces, released only after grading.

```sh
.venv/bin/python -m experiments.verse_sources restore          # unpack the pinned Canzoniere pages
.venv/bin/python -m experiments.joint_recovery_v4 verify
.venv/bin/python -m experiments.joint_recovery_v4 evaluate     # regrades the archived predictions
```

The prior file is the round-two prose+verse prior under `artifacts/verse-prior/` (about 50 MB),
hashed in `freeze.json`. The segmenter is the round-one frozen lexicon model, untouched.
CPU only; no downloads in this round; no paid compute.
