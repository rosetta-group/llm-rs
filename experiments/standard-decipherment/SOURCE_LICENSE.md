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
