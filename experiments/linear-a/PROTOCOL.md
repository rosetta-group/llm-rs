# Linear A, round one: can a lexicon match pick out a language at Linear A's size?

Status: written before any development result exists, 2026-09-23. Branch `linear-a`.
CPU only. Sources pinned in [sources.json](sources.json). Plan: [PLAN.md](PLAN.md).

## What the solver receives

- Linear A word divisions from the Navarre-AI collation, keeping only words of two or more signs
  whose every sign has an assumed Linear B value (692 distinct words).
- Candidate lexicons from Wiktionary: Greek (Ancient Greek lemmas), Hittite, Akkadian, Ugaritic,
  Hebrew, Etruscan, Egyptian, Sumerian. Each lemma is spelled with Linear B conventions
  (`linear_a/spelling.py`). Logograms written in capitals are excluded.
- It does not receive glosses, word classes of Linear A, or any proposed reading.

Luwian and Hurrian, two frequent proposals, have no Wiktionary extract and are not tested.

## Statistic

```text
for each language L:
  m      = number of sample words whose nearest L form is within theta
  null   = 10 pseudo-samples from a syllable bigram model of the sample itself, same lengths
  z_L    = (m - mean(null m)) / max(sd(null m), 1)
identified = argmax z_L if z_L >= 3 else none
```

The null keeps the sample's syllable patterns and destroys word identity, so lexicon size and
spelling lossiness cancel out of `z`.

## Stages and gates

1. **Development** (`experiments/linear_a_development.py`): synthetic controls only. For each
   language L, `k` lemmas of L are mutated (each syllable's consonant or vowel changed with
   probability 0.2) and mixed with Linear-A-like noise words to 692 words. `k` in 10, 20, 40, 80,
   160; 20 draws each. Negative control: 40 samples of noise only. Threshold `theta` in
   {0, 0.2, 0.25, 0.34} is chosen to minimise the mean log2 of `k*` (smallest `k` with at least
   90% correct identification) among thresholds whose negative control produces a winner in at
   most 5% of draws. Nothing else is tuned.
2. **Freeze.** Code, settings and source hashes go into `freeze.json`, committed.
3. **Linear B control** (the real positive control, not seen before the freeze): `k` Mycenaean
   words of known Greek reading (all 218 usable words when `k` = 218) mixed with noise to 692.
   Gate: Greek identified in at least 90% of 20 draws at some `k` <= 208 (30% of the sample), and
   the frozen negative control false-winner rate <= 5%.
4. **Linear A** runs only if the gate passes. It reports `z` for each language and, for each
   language not identified, the development `k*`: a statement that fewer than about `k*` Linear A
   words can be lexical matches of that lexicon under these spelling rules.

## What would count

A language identified on Linear A with the gate passed is a lead, not a decipherment: it would
need a second, independent test (held-out documents, commodity-sign contexts) before any reading
is proposed. A failed gate means the method cannot tell languages apart at this size, and the
Linear A run is not made.
