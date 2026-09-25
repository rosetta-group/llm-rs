# Theban image audit: source-access stopping point

This bounded search sought the original images needed to test the proposed attachment of *65.
No full-context image of the four target objects was obtained, so no spacing verdict is possible.

**Full context:** the disputed glyph plus neighbouring signs, quantities and surviving tablet edges.
**Sign function:** syllabic writing versus a commodity marker; neither choice alone establishes parentage.

## What was done

- Checked the four pinned DĀMOS records, LiBER coverage, publisher/catalogue routes,
  Google Books records, Internet Archive title search and open scholarly reproductions.
- Downloaded Palaima2006, Judson2016 and the Thebes museum volume; inspected relevant
  text and rendered pages. Judson reproduces a Gp124 .1 glyph drawing, but not its line context.
- Recorded source hashes and access outcomes in [sources.json](sources.json), and prepared
  an [unsent scan-request packet](ACQUISITION.md). No contact, purchase or access request was sent.
- Preserved all 11 earlier freezes. This is a source-access record, not a new computational
  experiment: no model, spacing measurements, test suite changes or new experimental freeze.

## Why

The transcription's spacing already reflects an editor's decision. An isolated sign crop cannot
independently test the distance to its neighbours, even when its tablet identity is secure.

```text
Look for the named tablet's photograph and facsimile
Check that neighbours, broken edges and line identity are visible
If only a glyph crop or transcription is available, reject spacing measurement
Record the missing source and park the image-dependent test
```

## Access result

| Target | Material obtained | What remains missing |
|---|---|---|
| Fq236 .5 | Palaima, Judson and Pierini discussions; DĀMOS text | TOP I pp94–95 photograph/drawing/transcription spread |
| Gp124 .1 | Isolated glyph drawing in Judson p225, attributed there to TFC III | Preceding `ko`, following divider/VIN, original spatial context and photograph |
| Gp227 .2 | DĀMOS text and discussions of `-u-jo` | Photograph and full-line drawing |
| Fq254[+]255 | DĀMOS joined-object text and published transcription | Photograph/drawing with .6, .7 and .13 in context |

DĀMOS items 5240, 5375, 5410 and 5252 have no direct tablet photographs: their `image` field
contains a Trismegistos catalogue link. LiBER explicitly excludes Theban archival documents
from current coverage. This is an access limitation, not evidence that images do not exist.
[LiBER project description](https://liber.cnr.it/),
[DĀMOS Fq236](https://damos.hf.uio.no/5240).

Google Books supplied catalogue/snippet records for TOP I, not the requested pages; page
access failed and the public Books API returned HTTP429. The Internet Archive title query
returned zero records. The publisher's AGS response endpoint returned HTTP202 with an empty
body. The museum volume's printed pp92–93 contain other tablet photographs with descriptive
captions, not an identified match to our four targets. None is substituted for a target image.
[TOP I catalogue](https://books.google.com/books?id=kcwLAQAAMAAJ),
[museum volume](https://www.latsis-foundation.org/content/elib/book_17/thiba_en.pdf).

## What the additional sources change

1. **The hand discrepancy has an edition history.** Palaima2006 p147 calls Fq236 hand310.
   Pierini2018 p115 n9 reports reassignment of Fq224/228/236/238/244 to hand304 in TFC IV;
   this agrees with pinned DĀMOS. Keep both historical labels with their sources. This is
   a reported reassignment, not our own handwriting identification.
   [Palaima2006](https://sites.utexas.edu/scripts/wp-content/uploads/sites/3428/2020/06/2006-TGP-65FARorJuAndOtherInterpretativeConundra.pdf),
   [Pierini2018](https://www.academia.edu/71674859/SYLLABOGRAM_65_OR_LOGOGRAM_129_far_THE_SIGN_%F0%90%80%8E_ON_THEBES_TABLETS_AB_65_IN_LINEAR_A_AND_SOME_REMARKS_ON_THE_O_STEM_GENITIVE_SINGULAR_IN_XO).
2. **Fq236 is a disputed exception.** Judson2016 pp151–152 leaves Fq132 and Fq236 uncertain:
   their breaks prevent recovery of the relevant glyph-to-measure spacing and other commodity
   placement. Pierini2018 pp115–117 instead favours syllabic use from internal comparisons
   and scribal practice, while acknowledging that the broken word may have continued.
   The image could check surviving distances; it cannot restore lost neighbours.
   [Judson thesis](https://www.repository.cam.ac.uk/handle/1810/265630), [Pierini2018](https://www.academia.edu/71674859/SYLLABOGRAM_65_OR_LOGOGRAM_129_far_THE_SIGN_%F0%90%80%8E_ON_THEBES_TABLETS_AB_65_IN_LINEAR_A_AND_SOME_REMARKS_ON_THE_O_STEM_GENITIVE_SINGULAR_IN_XO).
3. **Gp124 is the more promising comparison if images arrive.** Judson pp147–148 favours
   syllabic use at Gp110 .2 and Gp124 .1 from divider placement. Her p225 crop confirms a
   published glyph drawing is available, but omits those dividers and neighbours. These are
   two evidence levels, not two independent confirmations. Palaima2006 p147 also identifies
   same-hand divider comparators Gp110/112/122/127/168. These remain acquisition targets.
   [Judson thesis](https://www.repository.cam.ac.uk/handle/1810/265630), [Palaima2006](https://sites.utexas.edu/scripts/wp-content/uploads/sites/3428/2020/06/2006-TGP-65FARorJuAndOtherInterpretativeConundra.pdf).

## Decision

**Park the image-dependent lead.** The full TOP/TFC surfaces and the direct AGS response remain
unread. Do not repeat the same web searches or fit a spacing classifier to editorial text.
Reopen only when full-context scans or equivalent museum images become available; the packet
specifies what is needed. A future image study must declare controls and freeze its measurement
rules before measuring. It cannot be described as blind validation of a newly discovered lead:
the investigator already knows the proposed readings.

No new son/daughter labels, family edges, Linear A scores or changes to earlier negative results.
Pierini's proposed Linear A analogies are not adopted: they do not supply independent name,
gender or kinship labels for our target corpus. The report narrows the stopping condition;
it does not claim to have completed the unavailable surface adjudication.
