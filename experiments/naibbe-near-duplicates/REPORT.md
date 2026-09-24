# Naibbe reaches the Voynich near-duplicate rate only with heavy letter pairing, and never the v101 figure

Run 2026-09-24. Protocol committed first (`6568d26`). CPU, 6 minutes. No Voynich decoding.
[results.json](results.json) holds all 90 configurations.

## Result against the declared criterion

- **Criterion met: Naibbe can match.** 16 of 90 configurations reach at least 8.33%, the EVA figure.
  The best is 9.36% (English, RESPACING 9, 78-card deck, no space removal).
- **Every one of the 16 uses RESPACING 9.** That enciphers only 25% of letters alone and 75% in pairs.
  The published setting, 17, pairs 53%.
- **No configuration reaches the v101 figure of 12.20%.**

Mean near-hapax share by setting, averaged over the others:

| Setting | Values → near-hapax share |
|---|---|
| RESPACING | 27: 3.3% · 17 (published): 6.2% · **9: 8.4%** |
| deck | 56: 6.0% · 78: 6.0% |
| space removal | 0: 6.2% · 0.03: 6.1% · 0.10: 5.7% |
| language | Italian 5.5% · Latin 5.8% · German 6.1% · Old French 6.2% · English 6.3% |

At published settings the five languages give 5.8–6.7%, which reproduces the published sample's
5.86%.

## What else does not match

**Hapax share of types.** The 16 matching configurations give 55–62%. The Voynich text gives 71%
(v101 71.2%, EVA 70.6%). So at settings that match the near-duplicate share, Naibbe makes fewer
distinct one-off forms than the manuscript.

## Implication

1. **Near-duplication does not rule the Naibbe mechanism out.** It does constrain it: a Naibbe
   reading of the Voynich text needs heavy letter pairing, about RESPACING 9, far from the
   published default.
2. **The mismatch in one-off forms is a second, independent check.** Near-duplicate share and hapax
   share of types can't both be matched in this grid. A mechanism claim should report both.
3. **The decoder has only been tested at the published RESPACING of 17.** Heavier pairing means more
   two-piece tokens and a larger space of piece combinations. Before any Voynich use, the round-six
   decoder needs a development test on RESPACING-9 ciphertext.

This tests one statistic against one set of published tables. Other tables, other verbose ciphers
and a joint match of several statistics remain untested.
