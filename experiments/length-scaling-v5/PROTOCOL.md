# Context admission of rare cipher pieces (development only)

Declared 2026-09-24, before any run. CPU only. No download, no sealed passage, no Voynich text.
The same three development texts, key and seed (7000), at 20,800 letters.
[Length-scaling-v4](../length-scaling-v4/REPORT.md) found that after the reparse, spurious pieces are
nearly harmless, and the 33–38 missing true pieces (52–56 at RESPACING 9) are the letter bottleneck.

## Candidate (`voynich/piece_admission.py`)

Run the reparse pipeline R: square-root-scaled round four, refinement, then two reparse and key-refit
rounds. Then, once:

1. **Candidates.** For each token type seen at most 5 times, take every reading that uses exactly one
   new piece. That is the whole token as a new one-letter piece, or a prefix+suffix split where one
   half is a known piece in its role and the other half is new.
2. **Scoring.** For each (role, piece) and each of the 23 letters, sum over occurrences the saving
   in character bits: order-5 prior, 4 letters of context either side, plus homophone-choice bits,
   counting only occurrences that improve.
3. **Admission.** Admit a candidate with its best letter if the total saving is at least 10 bits.

Then two more reparse and key-refit rounds, and polish. Nothing reads the oracle trace; the trace is
used only to report how many admitted units are true.

## Runs and selection, fixed now

Three texts at RESPACING 17 and at RESPACING 9. Baselines are the recorded R rows: mean 1.79% at 17
and 3.30% at 9 ([v3](../length-scaling-v3/REPORT.md), [v4](../length-scaling-v4/REPORT.md)).

**Adopted** if the mean polished CER is at least 0.3 points below the baseline at one or both
settings, and no text at either setting is more than 0.2 points worse. Also reported: admitted units
and how many are true, missing true units after admission, WER and cap hits.
