# Fresh sources for the v3 segmenter test (recorded before passage construction)

- **Historical:** Dino Compagni, *Cronica delle cose occorrenti ne' tempi suoi*, Books I–III,
  Wikisource (page revisions 3798126 and later in [sources.json](sources.json); the archived
  HTML hash pins the transcluded pages). A new author, never used for fitting or testing.
  This edition has no chapter rubrics. Book titles fall below the 8-word minimum,
  there is no line-end hyphenation, and errata spans render the corrected reading
  ([extraction.json](extraction.json)).
- **Modern:** UD Italian ParTUT, CC BY-NC-SA 4.0, pinned commit in `sources.json`.
  **Deviation from the name approved in chat ("test split"):** the test split alone has about
  3,640 words (~17k letters), fewer than four 5,200-letter passages need. The dev split is
  appended after test, in file order. No model has seen any ParTUT split. Since UD v2.1,
  ParTUT's splits avoid sentences overlapping UD_Italian. The 20-word overlap
  check still runs against ISDT and all fitting text.
- The overlap check also covers every previously released fresh passage (Villani, VIT).
- The source choice was made in chat by name, before download, without any decoder output.
  Passages are then taken in page and file order by the unchanged v2 packer.
