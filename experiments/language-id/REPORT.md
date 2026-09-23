# Language-identification control: the true language wins in 5 of 5 Naibbe ciphertexts

Evaluated 2026-09-23. Five sealed ciphertexts, 25 decodes, CPU, about 100 minutes of compute.
Protocol and priors were committed before any passage was encrypted ([PROTOCOL.md](PROTOCOL.md),
[freeze.json](freeze.json)). No download and no Voynich text.

## Question

The pipeline must not assume its source language. So the test takes Naibbe ciphertext from
a known language, decodes it under each candidate prior, and checks whether the true language
fits best.

## Setup

- **Languages:** medieval Latin (ITTB), Old French (PROFITEROLE), German (GSD), English (EWT) and
  Italian. Italian's prior is historical prose plus ISDT; its passage is an unused Compagni passage.
- **Priors:** order-5 character priors, each fitted on exactly 606,976 letters.
- **Decoding:** each 5,221–5,379-letter passage got a new key and Naibbe seed. It was decoded
  under all five priors with round four's shared stages. No cap was hit.
- **Score:** $\text{excess}_L$ = bits per letter of the decode under $L$, minus $L$'s own held-out
  entropy. The lowest excess is the prediction.

## Results ([results.json](results.json))

Excess in bits per letter. Rows are the true language, columns the prior. The lowest in
each row is in bold.

| True \ prior | Latin | Old French | German | English | Italian | Margin |
|---|---:|---:|---:|---:|---:|---:|
| Latin | **0.453** | 1.492 | 1.768 | 2.081 | 1.769 | 1.04 |
| Old French | 2.414 | **0.463** | 1.795 | 2.023 | 1.873 | 1.33 |
| German | 2.493 | 1.549 | **0.272** | 2.111 | 1.891 | 1.28 |
| English | 2.543 | 1.565 | 1.948 | **0.233** | 1.928 | 1.33 |
| Italian | 2.076 | 1.230 | 1.603 | 1.731 | **0.478** | 0.75 |

- **Primary: 5 of 5 correct.** The median margin to the runner-up is 1.28 bits per letter; the
  success rule needed at least 4 correct.
- **Secondary:** raw bits per letter also picks the true language 5 of 5 times. So does the oracle,
  which scores the true plaintext with no decoding.
- **Under the true prior, the decode is close to natural text:** 0.23–0.48 bits per letter above
  held-out entropy. Under a wrong prior it is 1.2–2.5 bits worse. With a wrong prior the decoder
  still returns text, but that text fits its prior much worse.
- **Italian has the smallest margin (0.75 bits).** Its passage has the genre shift the protocol
  expected, historical chronicle against a mostly modern prior. Old French is its runner-up.

## What this licenses, and what it does not

- **It licenses** using the lowest excess to compare candidate languages, for Naibbe-class
  ciphertext of about 5,200 letters, when the true language is among the candidates.
- **It does not show** that the Voynich text is a Naibbe-class cipher, or that its language is
  among these five. It also doesn't show the method separates close relatives, such as two
  Italian dialects or Latin against Italian at very small margins. Five cases, one passage
  per language, is a pilot.
- **An Italian prior on the Voynich text is not justified by default.** The same scoring must
  rank all candidates. And if every candidate gives an excess far above 0.5 bits per letter,
  the right conclusion is "none of these", not the lowest one.

A Voynich run is still gated by the research plan: the Naibbe recovery gate of 1% CER and 10%
WER is not met. Scoring Voynich text under these priors would be a mechanism test that needs an
explicit decision. Candidates worth adding first include Occitan/Catalan, Czech, Hebrew in
transliteration and Latin of other genres, each needing a pinned corpus.
