# Codebook-free Naibbe recovery, round three: word-level polish

Status: frozen by `experiments/joint-recovery-v3/freeze.json` at the commit recorded there; the
fresh evaluation runs under `experiments/joint_recovery_v3.py`. Rounds one and two are unchanged
([round two protocol](../joint-development-v2/PROTOCOL.md), [round two fresh report](../joint-recovery-v2/REPORT.md)).
CPU only; no cloud, no downloads, no Voynich text, final test sealed.

## What changes and why

Round two ended at 9.5% (modern) and 10.3% (Dante) character error. Development attribution
shows where the error sits: within correctly parsed tokens, about 88% of all letters, the
letter error is 0.5–1.0%; every other error comes from the roughly 10% of tokens whose parse is
wrong. The character prior tolerates a wrong letter when it leaves a pronounceable string; a
lexicon does not. Round three adds one stage and changes nothing else:

**Lexical polish.** After the character-level refinement, sweep the role units in order of
frequency. For each unit the character prior shortlists the five best alternative letters; each
alternative is re-scored as character bits plus `polish_weight` times the change in the
lexicon segmenter's minimum segmentation cost, computed in merged windows of ±30 letters around
the unit's occurrences. A change is accepted only when the combined cost falls. Up to three
sweeps. The segmenter is the round-one frozen lexicon model; it is not retrained.

Development (three 5,200-letter texts, prose+verse prior): weight 1 lowers error on all three
(historical prose 10.9% → 9.6%, modern 9.3% → 8.6%, Petrarca 9.5% → 9.2%); higher weights help
prose and hurt verse; weight 1 is frozen. Re-parsing tokens under the same local cost was tested
and rejected: it helped one text and hurt another. Warm-starting the joint EM from the polished
key was tested separately and is not part of this round.

## Method

```text
round two pipeline: joint EM -> usage pruning -> joint EM -> character refinement
polish: for each unit by frequency, shortlist 5 letters by character bits,
        accept the letter that most lowers char bits + weight * lexical bits over local windows
report the polished letters; also record the pre-polish letters for a paired comparison
```

## Fresh evaluation

- Two modern ISDT test passages and two Dante passages, 5,200–6,000 letters, excluding every
  source ID used by any earlier challenge including rounds one and two.
- Encoding, blinding, grading, gate and the four-passage caveat as in round two.
- Additional evaluator-side measurements: letter error within correctly parsed tokens and the
  share of letters in mis-parsed tokens, to check that the development attribution holds.
- Both the polished and the pre-polish letters are graded, so the polish's effect is paired.

## Limits declared in advance

- The polish cannot repair a mis-parsed token; it can only fix letters inside correctly parsed
  ones, which development shows are already near 1% error. The expected gain is about one point.
- Word error remains limited by the untouched segmenter and by mis-parsed tokens.
- The gate (CER ≤ 1%, WER ≤ 10%) is not expected to pass. Nothing here concerns Voynich text.
