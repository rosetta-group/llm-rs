# Word segmentation v4: a historical spelling model for unknown words (development only)

Declared 2026-09-24, before any development scoring of a v4 candidate. CPU only. No download.
Baseline: the frozen v3 segmenter, with order-5 spelling on lexicon types, $p_{unk}$ = 2.34% and the
elision join. Its development WER is 8.65% on historical prose, 20.68% on Petrarca and 5.61% on modern
ISDT.

## Hypothesis

v3's spelling model learns mostly from 377k modern Morph-it forms, so archaic spellings look like
unlikely words. On Compagni, 344 of 391 remaining extra spaces fell inside missing forms such as
`erono`, `fusse` and `-orono` verbs.

## Candidates, order 5, everything else as v3

- **hist:** spelling fitted on word types of historical training text only: historical prose train
  and Petrarca train poems.
- **mix:** $P(w) = \tfrac12 P_{\text{lexicon}}(w) + \tfrac12 P_{\text{hist}}(w)$.

No other setting is tried, and no development word is added anywhere.

## Selection, fixed now

A candidate is eligible if its mean WER over historical prose and verse is at least 2 points below
v3's, and no stream worsens by more than 1 point. Pick the lowest historical mean; ties go to modern
WER, then `mix`. If none is eligible, record the negative. An eligible candidate is then frozen for a
fresh test on a new historical author. Compagni is excluded, because it motivated this candidate.
