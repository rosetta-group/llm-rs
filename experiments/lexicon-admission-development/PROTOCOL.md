# Decoder C with honest transfer scoring: full development on the released second confirmation

Declared 2026-09-28, before the fits below. Development only; all 72 inputs are released.

```text
For each of the 72 released v2 inputs and each of the eight v2 priors:
    fit A (must reproduce the archived key) and C from the same stages; fix both keys
    decode the transfer passage; score it per run (frozen) and with one length code (rejection_v3)
Apply decide_transfer with ceilings 0.45 and 0.50, all eight languages and true-language omitted
```

C = A plus up to two rounds of joint whole and half admission fed through joint EM
(`experiments/lexicon_admission_development.py fit_ac`). Nothing is tuned after the run. A
combination is a candidate for the next frozen rule only if it accepts more true languages than
A with the frozen scoring (16/24) and accepts no wrong language, no omitted language and no negative.
Any candidate still needs a fresh confirmation.
