# Linear A, round two: tablet position does not rescue lexical matching

Status: complete, 2026-09-23, branch `linear-a`. Protocol: [PROTOCOL.md](PROTOCOL.md), frozen by
[freeze.json](freeze.json) before the Knossos split was scored. **The gate failed, so the protocol
did not run Linear A.** No reading of Linear A is proposed.

## What was done

- Downloaded all 5,932 DĀMOS Linear B documents at one request per second; none failed
  ([sources.json](sources.json)).
- Labelled each word by what follows it on its line: entry (a number follows), logogram, header,
  other (`linear_a/contexts.py`). Linear A labels come from its Unicode lines: 696 types, of
  which 330 are entry words.
- Scored agreement between a matched lemma's class (proper name or not) and the word's position,
  against a length-stratified permutation null (`linear_a/context_test.py`).
- Development on the mainland archives (2,630 types), then the frozen test on Knossos (2,203 types).

## Why

Round one showed chance matches swamp true ones. A chance match picks a random lemma; a true one
should be a name when the word heads an entry. That was the one extra signal available without
new lexicons.

## Results

| Stage | Greek identified | Other language identified | Greek median z |
|---|---:|---:|---:|
| Development, mainland, best setting (θ = 0, rule "any") | 5% | 0% | 0.64 |
| Knossos test (gate) | 0% | 5% (Sumerian, 1 of 20) | 0.39 |
| Gate needed | ≥ 90% | ≤ 5% | — |

1. **No signal in Linear B itself.** On the mainland, Greek-matched names sit in entry position
   35% of the time (77 of 217), and Greek-matched common words 34% (140 of 319). Most Linear B
   entry words are personal names that Wiktionary does not list, so the true matches that could
   agree are few, and the chance matches dilute them.
2. **The test fails where the answer is known.** On Knossos, the closest archive to Linear A,
   Greek never clears z = 3 in 20 samples of Linear A's size.

## Exploratory, not gated: words shared by Linear A and Linear B

Checked after the test, with no protocol, to decide what to try next.

| Measure | Observed | Chance (bigram model of Linear A, 200 runs) |
|---|---:|---:|
| Linear A types also found in Linear B | 94 | 72.7 (max 88) |
| Of those, three signs or more | 14 | 6.4 (max 13) |
| Same entry / non-entry position in both scripts | 54 of 94 | 49.5 (p = 0.22) |

About 20 Linear A words survive into Linear B beyond chance, among them `pa-i-to`, `su-ki-ri-ta`,
`se-to-i-ja`, `da-i-pi-ta`, `ki-da-ro` and `ta-na-ti`. That is too few to carry a statistical test
of function, and their positions do not agree beyond chance.

## What this does and does not establish

With the public lexicons and Linear A's 696 readable word types, neither spelling matches nor
spelling plus tablet position can pick out Greek from real Linear B. So neither can identify or
rule out a language for Linear A. This does not test Luwian or Hurrian, morphology, or methods
that use a lexicon built for names (for example an onomasticon of Anatolian or Semitic names).

## Records and reproduction

```bash
.venv/bin/python -m experiments.linear_a_context develop
.venv/bin/python -m experiments.linear_a_context verify
.venv/bin/python -m experiments.linear_a_context test
```

The DĀMOS items live in `artifacts/linear-a-sources/damos/` (git-ignored) and are checked
against the manifest hash in `sources.json`. DĀMOS and SigLA content is CC BY-NC-SA 4.0.
