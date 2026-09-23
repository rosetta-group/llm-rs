# Rongorongo, round one: the tablets are too short for sign-to-syllable recovery

Status: complete, 2026-09-24, branch `rongorongo`. Protocol, sources and code were committed
before the run ([PROTOCOL.md](PROTOCOL.md), [sources.json](sources.json)); results in
[results.json](results.json). **The gate failed, so the tablets were not decoded.** Nothing here
reads Rongorongo.

## What was done

- Built a Māori syllable model from the 1868 Bible and 1841 New Testament: 2.42 M syllables,
  reduced to Rapa Nui shape (50 syllable types).
- Disguised held-out passages of Grey's narratives and songs as unknown signs with random keys.
  There were 4 passages at each size, from 1,300 to 8,000 syllables, one-to-one and with 20% extra
  homophone signs.
- Recovered the key with a bigram hidden Markov model (emissions by EM, 20 restarts, 200 iterations).

## Why

Rongorongo is the one undeciphered script whose language is almost certainly known. A
statistical decipherment is possible only if a disguised text of the tablets' size can be
recovered. Rochala 2026 reported that it cannot with annealing; this is an independent test
with a different solver.

## Results

Share of syllable tokens recovered, mean (lowest) over 4 passages:

| Syllables | One-to-one | With homophones |
|---:|---:|---:|
| 1,300 | 61% (38%) | 54% (46%) |
| **2,600** (the tablets' frequent-sign stream) | **65% (44%)** | 77% (60%) |
| 5,200 | 77% (67%) | 80% (66%) |
| 8,000 | 84% (71%) | 85% (77%) |

Gate: at least 90% in 4 of 4 passages at 2,600. Observed 84%, 44%, 50% and 83%. **Failed.**

1. **Recovery at the tablets' size is partial and unstable.** At 2,600 syllables two passages
   recover over 80% and two under 50%. A decoder that is right on half the syllables of an
   unknown text cannot be checked, so it cannot produce a reading.
2. **More text helps slowly.** Even at 8,000 syllables, three times the usable tablet stream,
   only 1 of 4 passages passes 90%. Rochala's annealing search reached 100% at 8,000 but 25% at
   2,000. This EM solver does better on short text and worse on long, so each search method has
   its own limits. Neither reaches 90% at 2,600.
3. **This is the easy case.** The control assumes one sign per syllable, a clean transcription
   and a model language close to the target. Rongorongo likely mixes word signs with sound signs,
   its catalogues disagree, and Old Rapa Nui differs from Māori. Each of these lowers recovery.

## What this does and does not establish

With about 2,600 usable sign pairs, even the most favourable assumptions give unverifiable
partial recovery, for two independent solvers. Statistical sign-to-syllable decipherment of
Rongorongo is not possible with the text that exists. This does not test a logo-syllabic model or
other catalogues. It says nothing about what the tablets mean. A development smoke test used Grey's
*Ko nga mahinga*, syllables 20,000–22,600; the protocol passages started at 40% and 70% of each
text, away from it.

## Records and reproduction

```bash
.venv/bin/python -m experiments.rongorongo_round_one     # refuses to overwrite results.json
```

Sources are in `artifacts/rongorongo-sources/` (git-ignored) and are checked against
`sources.json`. The CEIPP transliteration was downloaded but not used, because the gate failed.
