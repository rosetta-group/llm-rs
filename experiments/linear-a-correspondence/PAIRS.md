# The 12 historical pairs, checked against source records

Date: 2026-09-24. These are spelling candidates, not identified names or translations.
The table separates legible matching signs from complete, unambiguous word attestations.

**What was done**

- Traced every historical pair to the pinned Linear A records and DĀMOS transcription lines.
- Kept q/k and numbered signs distinct; checked uncertainty, broken edges and source conflicts.
- Recorded full source evidence in `pair-evidence.json` and strict accepted tokens in
  `accepted-attestations.json`. Neither occurrence counts nor tablet faces add pair votes.

**Why:** the old parser removed distinctions and damage marks before matching. For example,
`a3-du-ṛọ[` became `a-du-ro`, and `a-qa-ro` became `a-ka-ro`.

| Historical normalised pair | Linear A evidence | Linear B evidence | Strict decision |
|---|---|---|---|
| a-du-re → a-du-ro | [KH 4](https://sigla.phis.me/document/KH%204/index-word.html), a-du-re immediately before a gap | [KN Bk(-) 9695, .2](https://damos.hf.uio.no/3930), `a3-du-ṛọ[` | Exclude: different first sign, uncertain final sign, broken boundaries |
| a-ka-ru → a-ka-ro | [HT 2](https://sigla.phis.me/document/HT%202/index-word.html), HT 86a/b, a-ka-ru | [KN Sc(-) 233](https://damos.hf.uio.no/220), `]a-qa-ro` | Exclude: ka ≠ qa, and broken left edge |
| a-ta-re → a-ta-ro | [ZA 8](https://sigla.phis.me/document/ZA%208/index-word.html), a-ta-re | [PY An(6) 35, .5](https://damos.hf.uio.no/4327), a-ta-ro; PY Jn(1) 415 also has a3-ta-ro, not needed | Retain exact a-ta-re → a-ta-ro |
| a-ti-ru → a-ti-ro | [ZA 4a](https://sigla.phis.me/document/ZA%204a/index-word.html), a-ti-ru | [KN Dc(-) 1272, .B](https://damos.hf.uio.no/1171), a-ti-ro | Retain exact a-ti-ru → a-ti-ro |
| di-de-ru → di-de-ro | [HT 86a](https://sigla.phis.me/document/HT%2086a/index-word.html), HT 95b; HT 95a has a boundary disagreement: source word di-de-ru-qe-ra2-u | [KN Dv(-) 1504, .B](https://damos.hf.uio.no/1343), di-de-ro; KN Dl(-) 9016 has a broken variant | Retain from intact attestations; omit HT 95a |
| ka-ka-ru → ka-ka-ro | [HT 118](https://sigla.phis.me/document/HT%20118/index-word.html), HT 122b, HT 93a actually read qa-qa-ru | [KN As(1) 604, .3](https://damos.hf.uio.no/552), qa-qa-ro | Retain, correct labels to qa-qa-ru → qa-qa-ro |
| ka-sa-ru → ka-sa-ro | [HT 10b](https://sigla.phis.me/document/HT%2010b/index-word.html), ka-sa-ru | [KN Dv(-) 1450, .B](https://damos.hf.uio.no/1313), ka-sa-ro; [KN C-(2) 912, .8](https://damos.hf.uio.no/843) | Retain exact ka-sa-ru → ka-sa-ro |
| ka-ta-re → ka-ta-ro | [KH 41](https://sigla.phis.me/document/KH%2041/index-word.html), ka-ta-re bounded by gaps on both sides | [KN Xg(-) 8101](https://damos.hf.uio.no/2832), ka-ta-ro; [MY Z-(-) 202](https://damos.hf.uio.no/5735); KN As(1) 604 has uncertain ka-ta2-ro | Exclude conservatively: A word completeness unresolved; not a demonstrated wrong reading |
| ke-ku-re → ke-ku-ro | [HT 20](https://sigla.phis.me/document/HT%2020/index-word.html), actually qe-ku-re | [KN Vc(5) 7656](https://damos.hf.uio.no/2558), `ke-ku-ṛọ[`; [PY Mn(1) 162](https://damos.hf.uio.no/4770), `ke-ku-ṛọ` | Exclude: qe ≠ ke, uncertain B ending |
| pa-ja-re → pa-ja-ro | [HT 29](https://sigla.phis.me/document/HT%2029/index-word.html), HT 88, HT 8b, ZA 10b, pa-ja-re | [KN As(2) 1519, .6](https://damos.hf.uio.no/1355), pa-ja-ro | Retain exact pa-ja-re → pa-ja-ro |
| ta-ta-re → ta-ta-ro | PK 1 glyph reading is ta2-ta-re; no SigLA sign layer in pinned record | [PY Eo 224](https://damos.hf.uio.no/4564), ta-ta-ro; PY Eb 874, PY Ep 301, KN As 607 also attested | Exclude: ta2 ≠ ta, and lacks required A layer |
| te-ja-re → te-ja-ro | [HT 117a](https://sigla.phis.me/document/HT%20117a/index-word.html) word/sign layers give te-*56-re; glyph layer gives te-ja-re. Explicit record conflict: AB056 ≠ AB057 | [KN X 5525](https://damos.hf.uio.no/1860), `]te-ja-ro`; [KN X 8661](https://damos.hf.uio.no/3190), `ṭẹ-j̣ạ-ṛọ`; [KN V(3) 479](https://damos.hf.uio.no/443), `te-ja-ṛọ` | Exclude: conflicting A sign, no unmarked B attestation |

Six historical candidates survive as exact source-supported pairs. The full strict corpus is
extracted mechanically; the six were not selected as the only input to the null tests.
Rejected candidates remain in the evidence archive. Exclusion is not proof that a damaged
reading is false or that two forms cannot be related through additional sound changes.

## Source scope and attribution

The evidence is from the pinned Navarre-AI collation (revision
`3a83a327505bd5e01c41f05185fdcd74a40fb062`, corpus SHA-256
`34b2d7d65c1377367e9a393cfa4004f6f9101f4ee5eea9f9b1a2ba830466ff59`) and the existing
5,932-item DĀMOS snapshot. Links point readers to primary entries; the archived transcription,
not a later live update, determines inclusion. Full hashes are in `freeze.json`. A live check
of DĀMOS 3930 also showed `a3-du-ṛọ[` on the audit date; the SigLA HT 117a word-view page and
drawing were inspected for provenance, without claiming a new palaeographic reading.

[SigLA's help](https://sigla.phis.me/help.html) documents uncertainty marks and word annotation.
[DĀMOS's search guide](https://damos.hf.uio.no/howto) explains that normalised searches remove
brackets and subscript dots: finding a normalised word is not evidence for an intact reading.
The SigLA-derived layers and lineara.xyz glyphs may share underlying editions, so agreement
between them is a quality check, not independent confirmation. Referenced GORILA pages,
source notes and museum IDs remain available in the pinned records/evidence where supplied.

Transcription-derived data: **CC BY-NC-SA 4.0**, attribution to SigLA (E. Salgarella and
S. Castellan), DĀMOS (F. Aurora), and the Navarre-AI collation/lineara.xyz sources. Source
licences and upstream URLs are retained in `experiments/linear-a/sources.json` and
`experiments/linear-a-context/sources.json` and under `artifacts/linear-a-sources/`.
