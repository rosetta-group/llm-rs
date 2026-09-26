# Broadened Latin, German and Catalan priors: development on the released round-one confirmation

Declared 2026-09-26, before any fit below. Development only; the 72 round-one inputs are released.

**Transfer excess:** decoded bits per letter on the transfer passage with the sealed fit key, minus
the model's calibration score; the frozen rule accepts at most 0.50.

## What changes

Three priors are rebuilt; five stay unchanged. Each new prior has 400,000 training letters spread
over many works, and calibration from two held-out works. Roles come from a sha256 order of work
groups, never from a model score ([sources](../key-recovery-confirmation-v2/sources.json)). No
round-one confirmation work is in any new prior.

```text
Fit the 72 round-one inputs under latin_broad2, german_broad2, catalan_broad2 (A and B arms)
Reuse the round-one fits under the five unchanged priors
Apply decide_transfer, unchanged, to all eight candidates
```

Before fitting, true round-one transfer texts scored under the new priors: Latin +0.03 to +0.36,
German −0.02 to +0.13, Catalan −0.21 to +0.17 over calibration (round one: 1.15–1.40,
0.52–0.56, 0.99–1.24).

## What counts as progress

The new priors go to a second fresh confirmation only if, on these 72 inputs, A accepts at least 16
of 24 true languages with no wrong-language, omitted-language or negative acceptance, and no cap.
