# Confirmed and candidate rules at Voynich-like pairing (RESPACING 9): development

Declared 2026-09-29, before any key is drawn. Development only: the 48 plaintext passages are the
released passages of the second confirmation, encrypted again with new keys.

**RESPACING 9:** about 75% of letters are encrypted in pairs; the only Naibbe setting that matches
the manuscript's near-duplicate rate. Every confirmation so far used the published setting (17).

```text
24 blocks, same pairing as the second confirmation; new key and seeds; three inputs per block
Decoder A fits all eight v2 priors; keys fixed before transfer
Score two rules on the same fits:
    current    per-run transfer score, eight languages, ceiling 0.50 (confirmed at RESPACING 17)
    candidate  one length code per passage, Catalan and Occitan as one group, ceiling 0.50
```

The candidate rule goes to one fresh confirmation (both pairing settings, independent keys) only if,
here, it accepts at least 16 of 24 true languages with no wrong-language, omitted-language or
negative acceptance and no cap. Otherwise the limit is recorded and nothing is frozen.
