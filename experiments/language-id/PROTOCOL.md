# Language-identification control for ciphertext-only Naibbe decoding

Declared 2026-09-23, before any control passage is encrypted or decoded. CPU only.
There is no download; every corpus is already pinned in `experiments/language-sources.json`.
No Voynich text is used.

## Question

The Voynich work must not assume a source language. Before any prior is used on the
manuscript, this checks the pipeline itself. Given Naibbe ciphertext from a known language,
does decoding under each candidate prior pick out the true language?

## Languages and text

| Language | Prior and held-out text | Encrypted passage |
|---|---|---|
| Latin (medieval, ITTB) | `UD_Latin-ITTB` train | same file, final sentences |
| Old French | `UD_Old_French-PROFITEROLE` train | same file, final sentences |
| German (modern) | `UD_German-GSD` train | same file, final sentences |
| English (modern) | `UD_English-EWT` train | same file, final sentences |
| Italian | historical prose train, then ISDT train | one unused Compagni passage |

All text passes through `normalize`, the 23-letter Naibbe alphabet, so German `ß` is
dropped and diacritics are stripped. Each file is used in sentence order. The first
sentences, up to $N$ letters, fit the prior. The last sentences give one encrypted
passage of 5,200–6,000 letters. The 20,000 letters just before the passage give the
held-out text. $N$ is the smallest prior budget any language allows after its reservations,
so every prior is fitted on the same amount of text. Every prior is an order-5
`CharacterPrior` fitted the same way. A passage is rejected if it shares a 20-word
sequence with its prior text.

The Italian passage is Compagni because the pinned Italian test text is used up. So Italian
has a genre shift between prior and passage, which makes its identification harder.

## Decoding and score

Each passage gets a fresh random key and a fresh Naibbe seed. Each of the 5 ciphertexts is
decoded under each of the 5 priors, 25 decodes in all. The decode is round four's shared
stages with frozen settings: joint EM, pruning, joint EM, lexicon repair, refinement.
Polish is skipped because it needs a per-language word model. Recovered text $r_L$ under prior $L$.

$$\text{excess}_L = \frac{\text{bits}_L(r_L)}{|r_L|} - h_L$$

$h_L$ is prior $L$'s bits per letter on its own held-out text. The excess is **primary**.
It measures how much worse the best decode fits $L$ than natural $L$ text does.
The predicted language is the one with the lowest excess. Raw bits per letter is **secondary**.
There is also an oracle row: each prior's excess on the true plaintext, with no decoding.
It shows whether the priors themselves separate the languages.

**Success:** the true language ranks first for at least 4 of 5 ciphertexts, and the median margin
to the runner-up is reported. This is a five-case pilot. One passage per language is not
a population estimate, and the decoder settings were tuned on Italian, which may favour
Italian. Failure means this pipeline cannot support a language claim for the Voynich text.
