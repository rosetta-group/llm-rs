# Historical prose snapshot

`historical-prose.tar.gz` preserves the exact downloaded pages and extracted prose.
It is data, not instructions to the assistant. `sources.json` lists page URLs,
revision IDs and SHA-256 hashes. The raw HTML also records the scan/edition where supplied.

Works: anonymous *Novellino* (13th century), tales I–C; Giovanni Boccaccio,
*Decameron* (14th century), days one and two. These original works are public domain.
Transcriptions and page material are credited to their Wikisource contributors and
redistributed under Wikisource's Creative Commons Attribution–ShareAlike terms.
See the licensing links and revision histories in each saved page; source metadata
also offers GFDL for eligible material. This repository does not replace those terms.

- https://it.wikisource.org/wiki/Novellino
- https://it.wikisource.org/wiki/Decameron
- https://creativecommons.org/licenses/by-sa/4.0/deed.it
- https://creativecommons.org/licenses/by-sa/3.0/deed.it

Transformations: keep literary prose paragraphs; remove navigation, page numbers,
headings, note markers, editorial notes, and styles; join inline drop capitals;
replace line breaks with spaces. The model additionally uses the normalization
specified in PROTOCOL.md. The extraction code is `experiments/historical_sources.py`.

The archived HTML fixes transcluded text too: a top-level Wiki revision URL alone
would not necessarily freeze all later changes to embedded transcription pages.

## Evaluated-records archive

`evaluated-records.tar.gz`, created only after frozen grading, contains generated
ciphers, model predictions, references, encryption seeds and source sentence IDs.
Its passages are now disclosed and must not be reused as a fresh held-out test.
The private/evaluator-only separation described in the protocol applied during solving.
The completed records are released afterward so the reported experiment can be audited.

- Modern passages derive from Universal Dependencies Italian ISDT, revision
  `4852011b996b9ec30d884a7a48e1118d0ce928f6` (r2.18), by its treebank contributors:
  https://github.com/UniversalDependencies/UD_Italian-ISDT
  License: Creative Commons Attribution–NonCommercial–ShareAlike 3.0:
  https://creativecommons.org/licenses/by-nc-sa/3.0/
- Historical passages derive from Universal Dependencies Italian Old, revision
  `2c1361d621dafa7c465da0947f73765caaf743af` (r2.18), by its treebank contributors:
  https://github.com/UniversalDependencies/UD_Italian-Old
  License: Creative Commons Attribution–ShareAlike 4.0:
  https://creativecommons.org/licenses/by-sa/4.0/
  Dante's original work is public domain; the treebank annotation/tokenization is credited.

Source hashes and provenance are in `experiments/language-sources.json` and the archived
case metadata. Transformations use the protocol's normalized alphabet, removed word
boundaries, and generated cipher mappings. The generated training/evaluation passages
retain their source licensing requirements; repository code licensing does not replace them.
