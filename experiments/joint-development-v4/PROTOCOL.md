# Codebook-free Naibbe recovery, round four: lexicon repair

Status: frozen by `experiments/joint-recovery-v4/freeze.json` at the commit recorded there; the
fresh evaluation runs under `experiments/joint_recovery_v4.py`. Rounds one to three are unchanged
([round three protocol](../joint-development-v3/PROTOCOL.md), [round three fresh report](../joint-recovery-v3/REPORT.md)).
CPU only; no cloud, no downloads, no Voynich text, final test sealed.

## What changes and why

Round three ended at 8.8% (modern) and 10.5% (Dante) character error with the letter mapping
99% right inside correctly parsed tokens. Development analysis for this round asked where the
mis-parses come from and ran the same joint EM with the true piece lexicon as an oracle: it
reaches 97% parse agreement and 0.5–0.7% character error on all three development texts. The
model and the search are not the limit; the candidate lexicon is. It has two defects:

**Concatenations admitted as whole pieces.** A frequent plaintext bigram yields the same
prefix+suffix string repeatedly; once it occurs six times it becomes a whole-piece candidate, and
the one-letter derivation then beats the two-letter one. Every such spurious whole piece in
development was a concatenation of two true pieces, and they caused 75–111 mis-parses per text.
**Rare pieces never admitted.** Pieces of rare letters (b, f, g, h, m, q, v, z) occur one to four
times and never reach the threshold; 58–65 true pieces per text are missing, and their tokens
fall back to an unknown whole parse.

Round four adds one stage between the second joint EM and the character refinement and changes
nothing else:

**Lexicon repair** (`voynich/lexicon_repair.py`). Drop a whole piece when its token count is below
`theta` = 5 times the count its two halves would produce as a bigram under the decoder's own key
and recovered text. Admit the complement of a known half, for tokens that have no parse at all,
when it completes at least `complement_minimum` = 2 such tokens. Rerun the joint EM on the new
lexicon. Run this twice (`passes` = 2); the second pass first drops pieces whose expected usage
in the first pass was below 1.0. Both repairs read only the ciphertext and the decoder's output.

## Development evidence and the frozen setting

Three 5,200-letter development texts, prose+verse prior, refined character error before the polish:

| Text | Round two | Repair, selected | Oracle, true lexicon |
|---|---:|---:|---:|
| historical prose | 10.9% | **5.0%** | 0.5% |
| modern | 9.3% | **5.5%** | 0.5% |
| Petrarca | 9.5% | **6.2%** | 0.7% |

The grid was `complement_minimum` ∈ {1, 2} × `passes` ∈ {1, 2}; the rule fixed before running was
to freeze the setting with the lowest mean refined error over the three texts. That is
`complement_minimum` = 2, `passes` = 2 (mean 5.5%; the others 5.8%, 6.3%, 7.9%). Parse agreement rises
from 90% to 93–94%. The round-three polish then gives 4.5%, 4.7% and 5.7%. Two alternatives were
tested and rejected in the same analysis: lowering the candidate threshold to 3 or 4 (worse:
12.0% and 11.3% on modern) and admitting complements for every token, not only unparsed ones
(worse: 17.2%).

## Method

```text
round two pipeline: joint EM -> usage pruning -> joint EM
repair, twice: drop concatenation-like whole pieces (ratio < 5)
               [second pass: first drop pieces with expected usage < 1]
               admit complements of known halves for unparsed tokens (>= 2 tokens)
               joint EM on the repaired lexicon
round three tail: character refinement -> lexical polish (weight 1)
report the polished letters; also record the pre-polish letters and the pre-repair parses
```

## Fresh evaluation

- Four Dante passages of 5,200–6,000 letters, excluding every source ID used by any earlier
  challenge including rounds one to three. No modern passages: the ISDT test split holds fewer
  than 1,000 fresh letters after three rounds, and no new corpus was pinned for this round. The
  modern comparison rests on development text only and is declared as such.
- Encoding, blinding, grading, gate and the four-passage caveat as in round three.
- Evaluator-side measurements: parse agreement before and after repair, letter error within
  correctly parsed tokens, share of letters in mis-parsed tokens, and the polished versus
  pre-polish grades.

## Limits declared in advance

- The oracle ceiling is 97% agreement and 0.5% character error; the repair recovers roughly half
  the distance to it. About 100 tokens per text are still split at the wrong point between two
  known pieces, and the concatenation test drops a few true one-letter pieces.
- Word error remains limited by the untouched segmenter and by the remaining mis-parses; the gate
  (CER ≤ 1%, WER ≤ 10%) is not expected to pass.
- The stop rule stated before this round: if Dante does not fall below 8% character error on the
  fresh passages, the codebook-free Naibbe track closes. Nothing here concerns Voynich text.
