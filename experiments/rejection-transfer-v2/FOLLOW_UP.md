# Follow-up candidates, not changes to this frozen screen

These are development proposals from already graded cases and a code audit. The
screen's thresholds, stopping rule and decoder remain unchanged. No candidate here
has earned a manuscript run.

## Decision calibration

The English positive had fit excess 0.584 and transfer excess 0.374. Its key transferred
with 6.91% CER, versus 6.30% on fit text; the sole failed condition was the 0.50 fit
ceiling. Italian passed with 0.405 / 0.404 excess and 5.56% / 5.85% CER. Latin also passed with 0.111 / 0.419 excess and 6.94% / 9.37% CER. These three cases
suggest testing a decision centered on the held-out score, while retaining agreement
of language winners, winning margins and coverage checks. This is a proposal inferred
after grading, not a retrospectively successful version of the present screen.

```text
Use released cases as development data
Compare the present rule with a transfer-centered rule
Calibrate across multiple sources and negative generators
Freeze both rules and the sample/budget plan
Grade new passages and new keys once
```

The English plaintexts themselves differ by 0.370 bits per letter in prior excess.
Absolute excess therefore combines ordinary passage variation with decoding errors;
the ceiling is not a calibrated probability of being natural language. See
`english-postgrade-diagnostic.json`.

## A stronger copying control

The first copy/mutate input's best-ranked key covered 54.24% of transfer tokens, far
below the 95% gate. Its excess also failed. A subsequent control should preserve the
usable cipher-piece inventory more closely, so rejection cannot rest mainly on unseen
pieces. Include a copying-only or frequency-matched copying control in addition to
the present mutation stress test. Neither this generator nor a pass against it stands
for all possible meaningless generators; no published Timm generator was run here.

## Make repeated controls affordable without changing the objective

The first two complete blocks required 5.862 fit-worker hours for 30 fits. A linear
projection is about 29.3 fit-worker hours for all ten blocks, above this round's
12-hour limit. The driver also reserves the worst-case five hours for the next
five-prior batch (five fits at up to 3,600 seconds), so it can stop before consuming
all 12 hours. Any resulting budget stop is inconclusive, not a failed statistical
gate. This is a runtime forecast from the completed blocks, not a recovery estimate.

The existing refiner in `voynich/variable_units.py:240` recomputes full-text scores for
all pairs of cipher units on each swap sweep. For U units and N decoded positions,
the pair batch has O(U² N) scoring work at fixed n-gram order. It also materializes two
int16 arrays with U columns and U(U-1)/2 rows: O(U³) memory, about 2 GB for those two
arrays alone at U = 1,000. The function accepts `sweeps=50`, but never reads that
argument; stopping instead depends on kicks, improvements and wall time.

1. First consider chunking the pair batch. It can bound memory while preserving the
   same candidate set and global minimum; test ties and uncapped output equivalence.
2. Consider exact incremental scoring of only the n-grams affected by a key change,
   including the homophone cost. The same file's annealing kernel already uses a
   position index for incremental updates. A new greedy implementation would still
   need equivalence tests; do not claim a measured speedup before benchmarking it.
3. Declare a work-based search budget and record wall time separately. Estimate the
   full control-study cost before choosing its sample count. The current screen's
   wall-clock budget and reserve rule remain unchanged.

Do not edit the frozen refiner in place. Any implementation belongs in a new module,
with released data used only for development and fresh controls for confirmation.

## Final screen outcome

The Latin copy/mutate control hit English and German refinement caps. The screen
ended inconclusively after 45 fits and 8.884 fit-worker hours: positives 2/3 accepted,
shuffle 3/3 rejected, absent language 3/3 rejected, copy 2/3 rejected and one
inconclusive. The reserve rule would also forbid another batch (8.884 + 5 > 12),
but the recorded stopping cause is the refinement cap. The next three actions were
subsequently authorized and are declared in [their development protocol](../rejection-followups/PROTOCOL.md).
