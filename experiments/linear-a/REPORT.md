# Linear A, round one: lexicon matching cannot identify a language at this size

Status: complete, 2026-09-23, branch `linear-a`. Protocol: [PROTOCOL.md](PROTOCOL.md), frozen by
[freeze.json](freeze.json) at commit 11c4a7b before the Linear B control ran. **The gate failed,
so the protocol did not run the Linear A language test.** No reading of Linear A is proposed.

## What was done

- Pinned the Navarre-AI Linear A collation and eight Wiktionary lexicons ([sources.json](sources.json)).
- Spelled every lemma with Linear B conventions (`linear_a/spelling.py`) and matched words by
  syllable edit distance against a phonotactic null (`linear_a/matching.py`).
- Chose the threshold on synthetic controls only ([development-results.json](development-results.json)).
- Froze, then ran the real known-answer control: 218 Mycenaean words of known Greek reading
  mixed into Linear-A-sized samples ([control-results.json](control-results.json)).
- Ran two descriptive checks of the data and the sign values ([descriptive-results.json](descriptive-results.json)).

## Why

Linear A's sign sounds are roughly known and its language is not, so the Naibbe key search
becomes a language test. The Voynich work showed that sample size can decide the outcome, so
the method had to pass a known-answer test at Linear A's size (692 readable words) first.

## Results

| Check | Result | Needed |
|---|---:|---:|
| Toponyms from Linear B found in Linear A | 2 of 14 (`pa-i-to`, `se-to-i-ja`); chance 0.025 | descriptive |
| `ku-ro` totals equal the sum of their entries | 8 of 37 exact, 10 within 1 | descriptive |
| Development `k*` (lexical matches needed for 90% identification, θ = 0) | 80 to 160; Greek not reached at 160 | lowest |
| Linear B control: Greek identified, k = 208 of 692 | 10% of draws | ≥ 90% |
| Negative control: noise-only samples with a winner | 10% (Egyptian 3, Ugaritic 1, of 40) | ≤ 5% |
| All 218 Mycenaean words, no noise: Greek exact matches | 99 vs 81.6 by chance, z = 3.0 | — |

1. **The sign values pass a sanity check.** Two Linear B Cretan place names occur exactly in
   Linear A against 0.025 expected from a syllable model of Linear A. This supports the assumed
   values; it says nothing about the language.
2. **The arithmetic parse is partial.** Only 8 of 37 `ku-ro` totals equal the integer sum of the
   lines above them (HT 13: 130 = 130). The rest mostly close entries on another face or a broken
   part; fractions are ignored. Word and line data are usable but not clean.
3. **Linear B spelling destroys the signal.** CV spelling without final consonants turns so many
   words into short, common patterns that 37% of Linear-A-shaped pseudo-words match some Greek
   lemma exactly. Real Mycenaean words match Greek at 45%, only 8 points above chance.
4. **The matcher does find true cognates.** 83 Mycenaean words match Greek exactly, and at least
   57 of those are the cognate Wiktionary cites (`a-ko-ra` ἀγορά, `a-ki-re-u` Ἀχιλλεύς,
   `a-sa-mi-to` ἀσάμινθος) ([control-cognates.json](control-cognates.json)). Correct matches
   exist; they cannot be told apart from the chance ones.
5. **So the test has no power here.** Even if Linear A were Greek and 30% of its words were in
   the lexicon, Greek would be identified in 1 draw of 10. Consonantal lexicons (Egyptian,
   Ugaritic) also produce false winners, because their unknown vowels match everything.

## What this does and does not establish

It establishes that whole-word lexicon matching under Linear B spelling cannot identify
Greek from Linear B words at Linear A's size, so it cannot be trusted to identify or rule out any
language for Linear A. It does not test Luwian or Hurrian (no lexicon), morphology, or context.
The development `k*` values assume a 20% syllable mutation rate, which is a modelling choice.

Deviation: the frozen `best_form_is_cited_cognate` metric compared Greek script with Latin
script and always reported 0. It is recomputed in `experiments/linear_a_round_one_report.py`.
It is secondary and does not affect the gate.

## What could work next

- **Use context, not only spelling.** Words that come before commodity signs, words that come
  before numbers, and words found on libation vessels form classes. Test whether a candidate
  lexicon's word classes line up with them. This uses the one kind of information that the
  spelling rules do not erase.
- **Use longer words only.** Chance matching falls fast with length. A rerun on words of four or
  more signs needs its own power analysis, and has fewer words.
- **Get a running Linear B corpus** (DĀMOS, if its licence allows) for a control that includes
  realistic personal names.

## Records and reproduction

```bash
.venv/bin/python -m experiments.linear_a_development
.venv/bin/python -m experiments.linear_a_round_one verify
.venv/bin/python -m experiments.linear_a_round_one control
.venv/bin/python -m experiments.linear_a_round_one descriptive
.venv/bin/python -m experiments.linear_a_round_one_report
```

Sources live in `artifacts/linear-a-sources/` (git-ignored) and are checked against
`sources.json`. Results derived from SigLA fields carry CC BY-NC-SA 4.0.
