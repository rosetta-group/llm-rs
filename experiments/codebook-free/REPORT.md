# Codebook-free recovery: failed broader control

This tests recovery after removing supplied cipher-family labels and the Naibbe codebook.
The decoder is not validated for codebook-free Naibbe: it also fails its broader synthetic control.

**Character error rate (CER):** character edit distance divided by reference length; insertions can make it exceed 100%.
**Positive control:** verified ciphertext with a known original, used to check that the method can recover content.
**Oracle diagnostic:** the best candidate chosen after reading the reference; it cannot count as decoder success.

```text
Freeze the unpaired Italian prior, candidate mappings, score, and CPU budget
Generate four fresh passages under three encodings; keep originals and keys private
Run one decoder on opaque IDs and ciphertext only
Freeze every candidate and the selected prediction
Open originals to grade; report failed controls and oracle diagnostics separately
```

![Codebook-free recovery](../method-benchmark/figures/codebook-free.png)

| Encoding | Corpus | Selected CER | Selected word error | Recovery gate passes |
|---|---|---:|---:|---:|
| substitution | modern | 0.00% | 5.72% | 2/2 |
| substitution | historical | 146.29% | 208.73% | 0/2 |
| variable-homophonic | modern | 446.91% | 773.09% | 0/2 |
| variable-homophonic | historical | 459.30% | 652.72% | 0/2 |
| naibbe | modern | 657.06% | 826.48% | 0/2 |
| naibbe | historical | 662.46% | 926.69% | 0/2 |

1. **Assistance removed.** The public cases have only opaque IDs and ciphertext. No family,
   structural codebook, true key, encoder trace, original length, or word boundaries are
   supplied. Italian and its normalized alphabet remain known. The decoder tries observed
   characters and space-delimited units, with bijections and many-to-one one/two-letter
   expansions. These are explicit representation assumptions, not arbitrary cipher discovery.
2. **Controls and separation.** Two fresh modern and two fresh historical passages,
   1,200–1,800 letters each, exclude all earlier benchmark source sentences. Each is encoded
   by substitution, an artificial variable-length homophonic code, and published Naibbe
   with a hidden global letter permutation. The artificial code has two cipher spellings
   per plaintext chunk. All encoder roundtrips passed before solving. There are four
   independent passages, not twelve independent originals. Source/preparation and grading
   can access gold; the solver path cannot. This is auditable software separation.
3. **Selection failure.** The bijection candidate recovered all letters in all four simple
   substitution cases. The selected output preserved those correct candidates only for
   modern Italian. On both historical cases the fixed score preferred an incorrect
   expansion: e.g. score −2.197 for the wrong candidate versus −2.402 for the exact letters.
   The language prior therefore does not reliably choose the correct decoding. It is
   a mean conditional four-gram score with a unigram-distribution penalty, not a calibrated
   model of ciphertext generation or output length. Historical outputs have 146.3% pooled
   CER because the wrong mappings add many characters; this is not a clipped accuracy score.
4. **Broader failure.** The variable-length control and Naibbe fail even under the
   post-hoc best-candidate diagnostic (roughly 79–87% character error per case). Their
   selected outputs expand even further. Thus the weakness is not selection alone: search
   has not recovered the broader control. This result invalidates transfer claims for
   this decoder under this budget; it does not establish that every codebook-free method
   must fail, nor that Voynich lacks meaning. The earlier 99.44% Naibbe letter accuracy
   depended on the supplied structural codebook.

The gate required CER <=1% and word error <=10% per passage. Only the two modern
substitution cases pass. No full passage is exact. All candidate searches completed
72,000 proposals without hitting the 120-second cap; total solving and segmentation
took 83.5 CPU wall-clock seconds. Grading time is additional. The old and
new benchmarks differ in passages and assumptions; their difference is not a causal
estimate of the codebook's effect. No tuning or reranking followed grading.

The next mechanism to test, if pursued, is **decoding selection under language-prior
mismatch**: can a generative encoding/complexity criterion prefer the correct historical
candidate without access to its reference? First validate selection and broader search
on separate development controls, then freeze and use new held-out passages. This is
not another BPC sweep and has not been launched automatically.

Source: Michael A. Greshko (2025),
[The Naibbe cipher](https://doi.org/10.1080/01611194.2025.2566408),
[published code and license](https://github.com/greshko/naibbe-cipher).
[Pinned source/attribution](../decipherment-sources.json). The published encoder is used
only in challenge preparation; no structural tables are opened by the solver.

Audit: [protocol](PROTOCOL.md), [frozen code/prior hashes](freeze.json),
[all per-case and candidate scores](results.json). Candidate mappings, ciphertext,
predictions, and hidden answers remain in `artifacts/codebook-free/`. No Voynich final-test
text was used. Do not replace this negative run with a retuned score on the same passages.
