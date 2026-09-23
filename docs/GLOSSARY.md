# Glossary

Terms as used in this repository. One line each; details in the linked reports.

## Data

- **Voynich manuscript** — 15th-century codex in an undeciphered script; Yale Beinecke MS 408.
- **Transcription (v101, EVA)** — two conventions for writing Voynich glyphs as Latin letters. GC2a uses v101; ZL3b uses EVA. They differ in what counts as one symbol.
- **Folio / quire** — a leaf of the manuscript / a physical gathering of leaves. Page splits keep both sides of a folio together.
- **Currier A/B** — two statistical varieties of Voynich text; not proven to be distinct languages.
- **Validation / final test** — pages used to compare methods / pages reserved and never scored.
- **Naibbe cipher** — Greshko's published cipher that encrypts Italian into Voynich-like glyph strings; the main recovery test bed.
- **Timm sample** — text from a published self-copying generator; a control for prediction experiments.
- **ISDT, Italian-Old** — Universal Dependencies treebanks used for modern and Dante Italian.
- **Novellino, Decameron, Canzoniere** — non-Dante historical prose (two works) and Petrarch's verse, pinned from Wikisource for the historical prior.

## Cipher structure

- **Codebook** — the table mapping letters to glyph strings. Never given to the solver in codebook-free rounds.
- **Piece** — a glyph string standing for one letter. A Naibbe token is one piece or two glued together.
- **Role** — whether a piece is a whole token, the first part or the second part; the same glyph string can mean different letters in different roles.
- **Unit** — a role-tagged piece, the thing a key maps to a letter.
- **Parse / segmentation** — how a token is split into pieces. "Segmentation agreement" is the share of tokens parsed as the encoder did.
- **Homophonic** — several cipher symbols for one letter. **Variable-length** — one symbol for one or two letters.

## Methods

- **Character prior** — an order-5 interpolated character model of Italian; the solver's only knowledge of the language.
- **Description length (MDL)** — total bits to encode plaintext plus key plus residual choices; the criterion for choosing among candidate decodings.
- **Key beam** — Nuhn et al.'s search that extends a partial key one cipher symbol at a time, keeping the best partial keys.
- **HMM EM** — Berg-Kirkpatrick and Klein's model: letters follow a trigram model, each letter emits a cipher symbol, emissions learned by expectation-maximisation.
- **Joint EM** — this project's extension where each token's parse is a hidden variable learned with the emissions.
- **Usage pruning** — dropping candidate pieces the joint model barely uses, then rerunning EM.
- **Refinement** — iterated local search over a complete key under the description length.
- **Lexical polish** — re-scoring letter changes with a dictionary segmenter's local cost.
- **Lexicon repair** — dropping whole pieces whose count matches a prefix+suffix bigram under the decoded key, and admitting the complement of a known half for tokens with no parse.
- **Concatenation ratio** — a whole piece's token count over the count its two halves would produce as a bigram; low means the piece is really two letters.
- **Oracle segmentation** — the true parse from the encoder trace; a diagnostic only.
- **Segmenter** — a lexicon Viterbi model that inserts word spaces into a letter string.

## Evaluation

- **BPC** — bits per character; prediction loss, lower is better.
- **CER / WER** — character / word error rate: edit distance divided by reference length. Can exceed 100% when output is too long.
- **Gate** — the pre-declared pass mark for a recovery round: CER ≤ 1% and WER ≤ 10% on every case.
- **Sealed / fresh** — passages whose originals are hidden until predictions are saved and which no earlier challenge used.
- **Freeze** — a committed record of code hashes, inputs and settings, made before sealed passages exist.
- **Unicity distance** — roughly, the amount of ciphertext below which several keys give equally plausible plaintext.
- **Folio bootstrap interval** — resampling manuscript leaves to express page-sampling uncertainty; not seed or design uncertainty.

## Linear A track

- **Linear A** — Bronze Age Cretan script, mostly administrative tablets; sign sounds roughly known from Linear B, language unknown. See [LINEAR_A.md](LINEAR_A.md).
- **Linear B** — the later Mycenaean script, deciphered as Greek; used as the known-answer control for every Linear A method.
- **Sign values** — the sounds assumed for Linear A signs from Linear B signs of the same shape (about 60).
- **Spelling rules** — Linear B conventions applied to candidate words: open syllables, final and most cluster consonants dropped, r/l merged, voicing not written.
- **Known-answer (positive) control** — running a method on Linear B cut to Linear A's size, where it must find Greek; the gate was 90% of samples.
- **Negative control** — input with no language signal (noise words, shuffled syllables) that must not produce a winner.
- **Shuffled-syllable control** — Linear A syllables shuffled across words; keeps syllable frequencies, destroys words.
- **Entry word** — a word followed by a number on its line, the position of a person or place on a list.
- **Lexical match** — a candidate-language word whose spelled form is within a small syllable edit distance of a Linear A word.
- **Grammar profile** — a vector of word statistics (vowel alternations, final vowels, entropy, length) compared between lists.
- **`ku-ro`** — the Linear A word read as "total", because its number equals the sum of the entries above it.
- **Navarre-AI corpus** — the pinned collation of SigLA and lineara.xyz Linear A records (1,884 records).
- **DĀMOS** — the Oslo database of Mycenaean Linear B; 5,932 documents crawled; Knossos was the test split.
- **TLHdig** — digital Hittite tablet corpus (Beta 0.3) with Hittite, Luwian, Palaic, Hurrian, Hattic and Akkadian words tagged by language.
- **LAMAN, Oracc** — open lists of Anatolian and of Levant and Babylonian personal names.
- **Keftiu** — the Egyptian name for Crete; Keftiu names are Egyptian renderings of Cretan names, known only as consonants.
