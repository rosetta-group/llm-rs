# Rules at Voynich-like pairing (RESPACING 9): candidate 19/24 with no false acceptance, but 5 capped negatives

Development only, declared in [PROTOCOL.md](PROTOCOL.md) before any key was drawn. The 48 released
passages of the second confirmation, encrypted again at RESPACING 9 with new keys; 24 blocks × 3
inputs × 8 priors = 576 fits, decoder A. Full scores in [results.json](results.json).

The candidate rule (one length code, Catalan and Occitan grouped, ceiling 0.50) accepts 19 of 24
true languages with no wrong, omitted or negative acceptance. Five negative inputs hit the
20,000,000-proposal work limit, so the declared condition ("no cap") is **not met**. Nothing is frozen.

## Results (decoder A)

| Rule | Correct | Wrong | Omitted | Negatives accepted | Inconclusive |
|---|---:|---:|---:|---:|---:|
| current (confirmed at RESPACING 17) | 13 | 0 | 0 | 0 | 5 |
| honest scoring only | 17 | 0 | 0 | 0 | 5 |
| grouped only | 14 | 0 | 0 | 0 | 5 |
| **candidate** | **19** | 0 | 0 | 0 | 5 |

- Candidate by language: Italian, Catalan–Occitan (Catalan 3), English, German 3/3; Old French,
  Czech, Occitan 2/3; Latin 1/3. Median true-model transfer CER 6.0%.
- The five inconclusive inputs are all negatives (3 shuffles, 2 frequency copies). Each has one
  capped fit: refinement or the third admission round reached the proposal limit.
- Misses: Latin blocks 0 and 16 (fit margin 0.14 and 0.10), Occitan 7 and Czech 22 (true text over
  the ceiling), Old French 19 (fit margin 0.25 fails by rounding, excess 0.662).

## What it shows

1. **The candidate rule holds at Voynich-like pairing.** 19/24 here, against 13 for the confirmed
   rule, with no false acceptance.
2. **The work limit, not the rule, blocks it.** Negatives at heavy pairing need more search than
   the frozen limit allows; a cap is inconclusive by design.
3. **Next step, if wanted:** a resource-only change (a higher proposal limit), checked on these
   released inputs, before any freeze.
