# Whole-token admission: development result on released cases

Pattern: **lexicon repair by held-out context**. Admit a rare whole-token piece only when its
letter, chosen from the other occurrences, predicts the held-out one.

Admission passes the declared development criteria: it accepts 3/5 true languages against 2/5, with
no wrong acceptance and all five omitted-language cases still rejected.

**Transfer excess:** decoded bits per letter on the second passage, fixed key, minus the model's
calibration score; the frozen rule accepts at most 0.50.
**B:** the frozen fit. **A:** B plus whole-token admission and two context-reparse rounds.

## What was done

- Refitted all 8 frozen models on all 5 released cases: 40 fits, each giving key B and key A.
- B reproduces every archived key exactly (30 of 30 that exist).
- Keys fixed before transfer; frozen `decide_transfer` applied with unchanged thresholds.
- Protocol and code committed before the run (`9aee7e0`). No cap hit; about 4–5 min per fit.

## Why it was done

The fitted key, not transfer, carries the error: with the true fit key every case passes (excess
−0.14 to 0.26). Missing rare one-letter pieces carry most of the key error. See [protocol](PROTOCOL.md).

```text
frozen stages -> key B
repeat up to 3 times:
    for each split token type seen at least twice with no whole entry:
        saving per occurrence = split bits - whole bits, letter chosen from the other occurrences
    admit types whose total saving > 6 + log2(types tested) bits; refine
2 context-reparse rounds, refine after each -> key A
```

## Results

| True language | B accepted | A accepted | B / A fit CER | B / A transfer CER | B / A transfer excess | A failing gate |
|---|---|---|---:|---:|---:|---|
| catalan | none | none | 6.2 / 2.3% | 9.3 / 3.9% | 0.823 / 0.425 | fit winner Occitan; transfer margin 0.24 |
| german | none | none | 8.8 / 7.4% | 12.5 / 9.9% | 0.745 / 0.611 | transfer excess |
| latin | latin | latin | 4.9 / 1.7% | 6.7 / 4.3% | 0.429 / 0.108 | — |
| czech | czech | czech | 4.8 / 2.8% | 5.5 / 3.6% | 0.339 / 0.189 | — |
| occitan | none | **occitan** | 12.1 / 6.5% | 15.2 / 9.1% | 0.525 / 0.259 | — |

With the true language omitted, A rejects all five; the best wrong transfer excess is 0.67 (Occitan
on Catalan text) and 1.44–1.67 elsewhere. Full scores in [results.json](results.json).

## What the result supports

1. **Admission fixes the Occitan failure and halves most errors.** Transfer CER falls in all five
   cases, by 2.3–6.1 points. Latin and Czech margins widen.
2. **Catalan now fails on a close competitor, not on recovery.** Its transfer excess passes (0.425),
   but Occitan wins the fit passage by 0.15 bits and Catalan wins transfer by 0.24, below 0.25. The
   earlier Catalan case was never scored against Occitan.
3. **German remains a recovery failure.** Its key keeps letter confusions (d→s, a→i, e→m) that
   admission does not touch; even the true lexicon gave only 0.500.
4. **This is development evidence only.** The method was chosen on these cases with their true
   keys. It licenses a fresh, frozen confirmation on new works and keys, including retention of the
   earlier six languages; no accuracy or false-acceptance rate follows.

## Records and reproduction

- `python -m experiments.key_recovery_development run` refits into `artifacts/key-recovery-development/fit/`.
- `python -m experiments.key_recovery_development evaluate` grades and writes `results.json`.
- Module: `voynich/whole_admission.py`; tests: `tests/test_whole_admission.py`.

## Addendum: released rejection-screen controls

Declared in the protocol addendum (`260f752`) before any fit. 12 inputs × 8 models, keys fixed before
transfer, same rule. Full scores in [controls-results.json](controls-results.json).

| Input class | Inputs | B accepted | A accepted |
|---|---:|---:|---:|
| positive (English, Italian, Latin) | 3 | 3 correct | 3 correct |
| shuffle | 3 | 0 | 0 |
| copy-mutate | 3 | 0 (all capped) | 0 (all capped) |
| frequency copy | 3 | 0 | 0 |

- A accepts no negative and no wrong language; all three positives stay rejected with the true
  language omitted.
- Positive transfer CER falls with A: English 7.4 → 3.5%, Italian 6.6 → 4.3%, Latin 9.4 → 3.0%.
- **Copy-mutate cannot conclude under the frozen work limit.** Every refinement of all 24 copy
  fits reaches the 20,000,000-proposal limit, in both arms. These inputs are inconclusive, not
  rejections.
- Mean fit time: 241 s for B, 328 s for A including B.

A meets the addendum's conditions for a fresh confirmation, without the copy-mutate class.
