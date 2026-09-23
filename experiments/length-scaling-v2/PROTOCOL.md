# Length-aware lexicon for Naibbe recovery (development only)

Declared 2026-09-23, before any run. CPU only. No download, no sealed passage, no Voynich text.
Follows the [length-scaling negative](../length-scaling/REPORT.md). There, absolute count
thresholds let spurious concatenated pieces grow from 32–75 to 295–408 at 20,800 letters,
and refinement hit a fixed cap.

## Change

Let $k = n / 5200$. Every absolute count threshold and time cap is multiplied by $f(k)$,
so the relative-frequency cutoff that works at 5,200 letters is kept.

| Setting | Round four | Scaled |
|---|---:|---|
| candidate minimum count | 6 | round$(6 f)$ |
| prune usage | 3 | $3 f$ |
| repair complement minimum | 2 | max(2, round$(2 f)$) |
| repair usage floor | 1 | $1 f$ |
| refine cap (s) | 300 | $300 k$ |
| polish cap (s) | 1,200 | $1200 k$ |

The concatenation ratio θ = 5 is already relative and stays unchanged, as does everything else.

- **Primary, linear:** $f(k) = k$.
- **Secondary, square root:** $f(k) = \sqrt{k}$. It is reported, never selected in place of the primary.

## Runs

Same texts, key, seed and nested prefixes as the length-scaling test, at 10,400 and 20,800
letters. At 5,200 letters $f = 1$, so the settings equal round four's, and the recorded 5,200
rows are reused.

## Decision rule, fixed now (as before, for the primary)

If the mean polished CER at 20,800 letters is at most half the 5,200-letter mean of 5.62%,
that is at most 2.81%, declare a sealed long-passage round with the linear rule. Otherwise
record the result and move to parse fixes. Lexicon true/spurious/missing counts and cap
hits are reported for both variants.
