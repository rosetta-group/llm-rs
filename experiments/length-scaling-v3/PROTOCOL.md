# Two lexicon and parse candidates against the square-root length baseline (development only)

Declared 2026-09-23, before any run. CPU only. No download, no sealed passage, no Voynich text.
Follows the [length-aware lexicon test](../length-scaling-v2/REPORT.md). There, square-root
threshold scaling gave mean CER 3.19% at 10,400 and 3.25% at 20,800 letters, with 178–200
spurious and about 35 missing pieces at 20,800. Count thresholds alone had reached their limit.

## Baseline

Round four's decoder with square-root-scaled count thresholds and linearly scaled caps, exactly as
in length-scaling-v2. Its recorded rows are the baseline. Texts, key, seed and prefixes are
unchanged: 10,400 and 20,800 letters of the three development texts.

## Candidates, each changing one thing

- **glue: self-inclusive concatenation test** (`voynich/lexicon_repair_v2.py`). Round four builds
  the bigram count that judges a whole piece from split parses only. So a spurious concatenation
  already parsed as one whole piece removes its own occurrences from the expectation it is tested
  against. Here the occurrences parsed as that whole piece are added to the bigram and half-piece
  counts. θ = 5, the passes and the complements are unchanged.
- **reparse: context reparse with key refit.** After repair and refinement, run two rounds of:
  re-split every token with the existing full-context beam (`voynich/context_reparse.py`, width
  128) under the current key, then refit the key by refinement on the new parse. The earlier
  single fixed-key reparse was rejected. This candidate differs by refitting the key between rounds.
  Polish follows as usual.

## Selection, fixed now

A candidate is eligible if its mean polished CER is at least 0.5 points below the baseline at
20,800 letters, and at most 0.2 points above it at 10,400. Among eligible candidates, pick the lowest
20,800 mean. If a selected candidate's 20,800 mean is also at or below 2.81%, the original
length-scaling threshold, declare a sealed long-passage round. Otherwise record the result. The
two candidates are not combined in this test. Lexicon counts, mis-parse types, reparse changes and
cap hits are reported.
