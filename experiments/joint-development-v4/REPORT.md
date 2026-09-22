# Codebook-free Naibbe recovery, round four: development record

Development results on development text only. The [protocol](PROTOCOL.md) fixes the one change
tested here; the fresh evaluation is reported in the [round-four fresh report](../joint-recovery-v4/REPORT.md).

Round four asked why one token in ten is parsed wrongly. The answer came from an oracle: given
the true piece lexicon, the existing joint EM parses 97% of tokens correctly and recovers letters
at 0.5% error. Nothing was wrong with the model or the search; the candidate lexicon was. A
repair that reads only the ciphertext and the decoder's own output halves the character error.

**Candidate lexicon:** the set of glyph strings the joint EM may use as a whole piece or as the
half of a split; before this round, every string occurring six times or more as a token, prefix or suffix.
**Concatenation ratio:** a whole piece's token count divided by the count its two halves would
produce as a bigram under the decoded key and text.
**Oracle:** the same decoder given the true lexicon from the encoder trace; evaluator-only.

## What was done

- Attributed every mis-parse on the three development texts to its cause, against the encoder trace.
- Ran the joint EM with the true lexicon, with the pruned lexicon minus its spurious pieces, and
  with the pruned lexicon plus its missing pieces, to bound what each defect costs.
- Added `voynich/lexicon_repair.py` (concatenation test, complement admission, iterated EM) with tests
  in `tests/test_repair.py`; ran a four-setting grid; polished the selected setting.
- Recorded two rejected alternatives: a lower candidate threshold and unrestricted complement admission.

## Why

Round three showed that letters inside correctly parsed tokens were already 99% right, so only
segmentation work could pay. Before building a new segmentation model, the cheapest question was
whether the existing one fails on its own or is starved of the right candidates. The oracle
settles it: starved.

## Where the mis-parses come from (round-two pipeline, development text)

| Text | Mis-parsed tokens | Pieces in lexicon | True | Spurious | Missing true pieces | Spurious that are bigram concatenations |
|---|---:|---:|---:|---:|---:|---:|
| historical prose | 344 | 271 | 240 | 31 | 65 | 24 of 24 that occur as tokens |
| modern | 325 | 262 | 240 | 22 | 77 | 18 of 18 |
| Petrarca | 351 | 261 | 241 | 20 | 77 | 16 of 16 |

1. **Every mis-parsed token string is wrong in all its occurrences.** The parse is decided per
   string, not per context; this is a lexicon problem, not a context problem.
2. **Half the mis-parses had both true halves available** (147–178 per text) and were parsed as
   a whole anyway, because the token string itself sat in the lexicon as a spurious whole piece:
   a frequent plaintext bigram had produced the same prefix+suffix string six or more times.
3. **Missing pieces are rare-letter pieces.** Of the 65–77 missing, 58–65 occur fewer than six
   times; their letters are b, f, g, h, m, q, v, z. Usage pruning removed only 7–13.
4. **Oracles bound the two defects.** Removing the spurious pieces alone: 92–93% agreement,
   6.5–7% CER. Adding the missing pieces alone: 94–95%, 3–4%. Both: 97%, 0.5–0.7%.

## Repair results (refined character error; agreement in brackets)

| Text | Round two | min 1, 1 pass | min 1, 2 passes | min 2, 1 pass | **min 2, 2 passes** | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| historical prose | 10.9% (89.9%) | 6.4% (92.8%) | 6.1% (93.0%) | 6.1% (93.2%) | **5.0% (93.8%)** | 0.5% (97.1%) |
| modern | 9.3% (90.4%) | 9.9% (90.8%) | 6.9% (92.7%) | 5.5% (93.7%) | **5.5% (93.8%)** | 0.5% (97.4%) |
| Petrarca | 9.5% (89.7%) | 7.3% (92.0%) | 5.8% (93.1%) | 5.9% (92.9%) | **6.2% (92.9%)** | 0.7% (97.1%) |

"min" is `complement_minimum`. The selection rule fixed in advance picks the lowest mean: min 2,
2 passes (5.5%; the others 5.8%, 6.3%, 7.9%). With the round-three polish: **4.5%, 4.7%, 5.7%**.

1. **The concatenation test is precise.** At ratio 5 it drops 16–21 pieces per text, of which
   13–21 are spurious and 3–5 true one-letter pieces (measured with the true key; the decoded key
   gives the same counts within one).
2. **Complement admission for unparsed tokens is the safe form.** It admits 47–145 strings and
   recovers 20–33 missing pieces; the second pass prunes most of the spurious ones by usage.
   Admitting complements for every token (1,100 strings) collapses the EM to 17% error.
3. **Letters inside correct parses stay at 0.8–0.9% error;** the share of letters in mis-parsed
   tokens falls from 12% to 7–8%.

## What remains

About 100 tokens per text are still split at the wrong point between two known pieces, 60–76
true one-letter tokens are split, and 33–60 true splits are read as one letter. The gap to the
oracle is 3–4 points of agreement and 4–5 points of character error.

## Limits

- Development passages; the settings were chosen on them. Only the frozen fresh run counts.
- The oracle uses the encoder trace and never enters the method.
- No Voynich text; final test sealed; CPU only; no downloads.

## Reproducible table

`results.json`: per text, the round-two baseline, every grid setting with its lexicon counts,
parse-error kinds, attribution and repair record, the true-lexicon oracle, and the polished
selection. Driver: `experiments/joint_development_v4.py`.
