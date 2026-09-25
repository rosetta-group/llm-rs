# Whole-token admission after the frozen fit: development on released cases

Declared 2026-09-25, before the eight-model comparison below. This is development on released
records, not a fresh confirmation. No thresholds of the decision rule change. No Voynich text,
new sources or paid computation.

**Transfer excess:** decoded bits per letter on the second passage, fixed key, minus the model's
calibration score. The frozen rule accepts at most 0.50.

## What was already looked at

The steps below were exploratory and used the true key of each released case. They chose the
method; they do not count as evidence for it.

1. The frozen fitter reproduces all five archived true-model keys exactly.
2. With the true fit-passage key, transfer excess is 0.26, 0.20, −0.14, 0.05 and −0.02 for Catalan,
   German, Latin, Czech and Occitan. Transfer is not the bottleneck; the fitted key is.
3. With the true piece lexicon, EM and refinement give 0.25, 0.50, −0.13, 0.11 and 0.11. Adding
   the missing true pieces to the found lexicon gives 0.31, 0.57, −0.11, 0.11 and 0.13; removing
   the spurious ones gives 0.85, 0.71, 0.39, 0.33 and 0.49. The missing pieces carry the error.
4. The missing pieces are mostly one-letter whole tokens seen 1–6 times, below the candidate
   minimum of 6, and every one also splits into two known pieces.
5. The leave-one-out test in `voynich/whole_admission.py` separates them from true splits with
   AUC 0.87–0.99 on these cases, true model only. Context reparse alone or a warm-started EM alone
   changed transfer excess by less than 0.05 in four cases.

## What will be done

```text
For each of the five released cases and each of the eight frozen candidate models:
  run the frozen fit stages once (joint EM, prune, repair, refine) -> key B
  from that state: up to 3 rounds of whole-token admission, refine after each
  then 2 rounds of context reparse (width 128), refine after each -> key A
  fix both keys, then decode the transfer passage with each
Apply the frozen decide_transfer to B and A: all eight languages, and the true language omitted
```

Admission threshold, fixed before this run: key-entry bits $1 + \lceil \log_2 21 \rceil = 6$ plus
$\log_2(\text{types tested})$. Types seen once are never admitted. Caps: the frozen 3,600 s fit
cap for B and 7,200 s for A; any cap makes that case inconclusive.

## What counts as progress

A is worth a fresh confirmation only if, on these five cases, all of the following hold:

- it accepts more true languages than B;
- it accepts no wrong language;
- it still rejects all five cases when the true language is omitted;
- no cap is hit.

Anything else is recorded as a failure of A. Five released cases cannot estimate accuracy or a
false-acceptance rate; a pass here only licenses a fresh, frozen confirmation on new works and keys.
