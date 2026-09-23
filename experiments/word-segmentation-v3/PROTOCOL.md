# Word segmentation v3: a letter-level unknown-word model

Declared 2026-09-23, before any development scoring of a v3 candidate. CPU only.
No cipher decoding, Voynich text, image modelling, paid compute or download.
The frozen v2 files and the base segmenter model are unchanged.

## Hypothesis

The [diagnosis](REPORT.md) found that 83–90% of extra spaces fall inside words missing
from the lexicon. The flat unknown cost of $15 + 3\ell$ nats always loses to a split into
known pieces. Replace it with the cost of an explicit unknown-word model:

$$\text{cost}(w) = -\log p_{\text{unk}} - \log P_{\text{spell}}(w)$$

- **$P_{\text{spell}}$**: an interpolated Witten–Bell letter n-gram with word-start
  padding and an end-of-word symbol, fitted on the base model's lexicon types
  (Morph-it forms plus training word forms). Types, not tokens, since unknown words are rare.
- **$p_{\text{unk}}$**: set from training text only, with no development input. It is
  the Good–Turing share of unweighted training tokens (ISDT train plus historical
  prose train) whose form occurs once there and is not a Morph-it form.
- **Known words**: the known-word cost, transitions, beam, maximum length and
  parameters (`alpha=1, bigram=0.5`) stay as frozen.

**Elision join (optional component).** `normalize` merges an elided word with the next
one when it drops the apostrophe (`l'innocenti` → `linnocenti`). The optional rule joins
a predicted token from the fixed list `l d s c n m t v dell all nell sull dall coll quell
quest bell sant` to the next token when that token starts with a vowel or `h`. The
list comes from Italian orthography, not from development errors.

## Development comparison

There are four candidates: n-gram order 3 or 5, each with elision off or on. There is no
other setting and no weight sweep, and no development word is added to any lexicon. The
comparison uses the three existing 5,200-letter streams with perfect letters: historical
prose, Petrarca verse and modern ISDT. The baseline is the saved v2 weight-0 prediction.

**Selection** reuses the v2 rule. A candidate is eligible if it improves mean
historical WER (prose and verse) by at least 3 points, and no stream worsens by more
than 1 point. Among eligible candidates, pick the lowest historical mean. Break ties by
lower modern WER, then lower order, then elision off. If none is eligible, record the
negative and consume no fresh text. Report boundary counts, and extra and missing spaces
attributed as in the diagnosis.

## Freeze and fresh evaluation, conditional on selection

Commit the code, the protocol, the development record and the fitted-model digests
before any fresh source is fetched. Fresh text is one historical prose author and one
modern corpus, both unused so far. Villani, VIT, ISDT, Novellino, Decameron, Petrarca and
Dante are excluded. The sources are named and pinned before download, and the choice
never depends on decoder output.

Passage construction repeats v2: four disjoint passages per source, each 5,200–6,000
letters. Chapter rubrics and editorial material are removed with a source-specific
check like the v3 extractor, recorded before any passage is built. Passages sharing an
exact 20-word sequence with the fitting or development text are rejected. References
stay evaluator-only until the baseline and the selected predictions are both saved.

**Success** has the same thresholds as v2. Pooled historical WER must improve by at
least 3 points and modern WER may worsen by at most 1 point. The per-passage 10% WER
gate is reported separately. Passing segmentation alone does not open a Naibbe or
Voynich test. The only follow-up it justifies is a separately declared paired cipher
comparison.
