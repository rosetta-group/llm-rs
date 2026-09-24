# Can Naibbe produce the Voynich near-duplicate rate? (descriptive test of the mechanism)

Declared 2026-09-24, before any run. CPU only, no download. No Voynich decoding: the Voynich
figures are the recorded training-page statistics in
[voynich-suspects](../voynich-suspects/REPORT.md).

## Target

Near-hapax share of tokens in 10,000-token windows, meaning forms seen once that sit one edit from a
form seen at least 5 times. The Voynich text gives **8.33% (EVA)** and **12.20% (v101)**. The
published Naibbe sample gives 5.86%; natural languages give 1.2–2.2%.

## Grid

The published encryptor (`artifacts/decipherment/vendor/naibbe.py`) is run with its own tables and
glyph map. Only its module-level settings are varied:

| Setting | Values |
|---|---|
| plaintext language | Latin (ITTB), Italian (ISDT + historical prose), Old French, German, English: the prior texts of the language-ID control, first 40,000 letters |
| RESPACING (letter-alone chance = R/36) | 9, 17 (published), 27 |
| deck | 56 cards (published), 78 cards |
| space removal rate (adjacent cipher words merged) | 0, 0.03 (published), 0.10 |

That is 90 configurations, each with 2 encryption seeds. For each, compute the near-hapax share on the
first two 10,000-token windows and average the 4 values. Tokens are the cipher words after space
removal, as in the published sample.

## Criterion, fixed now

- **Naibbe can match** if any configuration reaches at least 8.33%, the lower Voynich figure.
- **Naibbe is a poor fit** if every configuration stays below it.

The configurations nearest the target are reported with their hapax share of types. This does not
test other tables or other verbose ciphers, and it does not test whether Voynich *is* Naibbe. It
tests only whether this statistic rules the published mechanism out.
