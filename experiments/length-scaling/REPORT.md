# Length scaling of Naibbe recovery: better to 10,400 letters, worse at 20,800; the lexicon does not scale

Development only, 2026-09-23. Round four's decoder with settings unchanged. Three development texts,
nested prefixes, one key and seed per text. CPU: three parallel processes, 80 minutes. The protocol
was committed before the run ([PROTOCOL.md](PROTOCOL.md)).

## Results ([summary.json](summary.json); per-length records in `historical.json`, `modern.json`, `verse.json`)

Polished CER / parse agreement / WER with the v3 segmenter:

| Text | 5,200 | 10,400 | 20,800 |
|---|---|---|---|
| Historical prose | 5.13% / 93.9% / 34.8% | 3.93% / 94.7% / 29.0% | **3.00%** / 95.6% / 24.3% |
| Modern ISDT | 5.60% / 93.8% / 38.6% | **2.75%** / 95.9% / 23.3% | 3.84% / 94.9% / 24.1% |
| Petrarca verse | 6.12% / 93.8% / 45.4% | **3.49%** / 95.1% / 33.5% | 5.69% / 93.3% / 39.0% |
| **Mean CER** | 5.62% | **3.39%** | 4.18% |

**Decision rule: not met.** The 20,800-letter mean (4.18%) is above half the 5,200-letter mean (2.81%).
Under the protocol, no sealed long-passage round follows. But doubling the length to 10,400 already
cuts letter error by 40%.

## Why 20,800 letters goes wrong

Candidate pieces, true / spurious / missing against the oracle-only trace:

| Text | 5,200 | 10,400 | 20,800 |
|---|---|---|---|
| Historical | 273 / 40 / 39 | 292 / 135 / 33 | 318 / 295 / 16 |
| Modern | 270 / 32 / 48 | 308 / 151 / 24 | 316 / 408 / 23 |
| Verse | 276 / 75 / 49 | 300 / 140 / 30 | 314 / 400 / 20 |

1. **The lexicon's thresholds are absolute, not relative to length.** With more text, more
   frequent two-piece concatenations pass the candidate count threshold. Spurious pieces grow
   7–13×, while missing true pieces fall from about 45 to about 20.
2. **The main error changes type.** At 20,800 letters the commonest mis-parse is a split token read
   as one spurious whole piece: 311–504 per text, against 49–59 at 5,200.
3. **Refinement was cut short.** The case cap scaled with length, but refinement's own 300 s cap
   did not, and it hit that cap on all three 20,800-letter texts. This confound is declared, not tuned.
4. **More text does help where the lexicon holds.** Missing pieces fall steadily, and prose, whose
   lexicon grows least, still improves at 20,800 (3.00%).

## Implication

Length is a real lever, blocked by settings that are fixed in absolute counts. The next
development candidate is a length-aware lexicon:
- scale the candidate and repair count thresholds with ciphertext length, as a declared rule;
- scale refinement's cap with length.

Then test it at 5,200, 10,400 and 20,800 letters with the same decision rule. Parse fixes such as
context reparse can be layered on afterwards. The Voynich text is long, so a method that improves
with length matters most for it. This is development evidence; no sealed result changes.
