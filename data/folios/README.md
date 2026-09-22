# Voynich folio image archive

The new [Object–Relation pilot](object-pilot/REPORT.md) adds reviewed objects,
evidence boxes and compound relations on 24 panels. Independent review is pending.

**Pixel Layout v1** is our first reproducible image-description method.
It records visible colour and geometry; it does not infer what a passage means.

Open the [searchable gallery](analyses/pixel_layout_v1_16d1ceb438d2/index.html),
[CSV table](analyses/pixel_layout_v1_16d1ceb438d2/descriptions.csv), or
[analysis registry](analyses/pixel_layout_v1_16d1ceb438d2/analysis.json).

**Panel:** one named entry in the Voynich.nu index, such as `f67r1`.
**Scan:** one Yale photograph, sometimes showing several panels.
**analysis_id:** the method name plus a hash identifying its code, rules, inputs and dependency versions.

```text
Download the Yale scans and catalogue panel links
Match shared-scan panels to catalogue thumbnails; review the proposed crops
Verify image hashes; crop each panel
Measure colour, spatial distribution, dark-mark bands and supported circles
Write one row per (analysis_id, folio_id)
Preserve each analysis; changed methods get a new analysis_id
```

1. **Coverage.** `images/` contains all 213 Yale manifest images at 1,800 pixels on
   the longest side: 121.2 MB of JPEGs, including covers, binding and flyleaves.
   `sources/catalogue-index.json` lists all 228 named catalogue panels and their
   links to 204 manuscript scans. Covers and binding do not get folio descriptions.
2. **Foldouts.** `crops.json` records 39 reviewed panel rectangles. Other panels
   use their complete linked scan. The two `f101v` catalogue links share an HTML
   anchor; thumbnail filenames distinguish `f101v1` from `f101v2`. Rectangles can
   include narrow neighbouring margins. They are not precise object masks.
3. **Provenance.** `images.json` stores URLs, dimensions, byte counts and SHA-256
   checksums. Native-resolution URLs remain available for detailed follow-up.
   `sources/yale-manifest.json` preserves Yale's labels, attribution and rights
   metadata. The 39 reference thumbnails preserve how shared panels were located.

## Run

Run from the repository root. CPU only; no model download or paid service.

```sh
.venv/bin/pip install -r experiments/folio_requirements.txt
.venv/bin/python -m experiments.download_folios
.venv/bin/python -m experiments.describe_folios
.venv/bin/python -m unittest discover -s tests -p 'test_folio_description.py'
```

The downloader resumes existing downloads and checks archived hashes. It never
silently replaces an archived image. The description script verifies every image,
then writes `analyses/<analysis_id>/` with:

- `index.html`: searchable image gallery and description table; opens offline.
- `descriptions.csv`: one row per panel, including the required `analysis_id` column.
- `descriptions.jsonl`: the same records with typed nested values.
- `analysis.json`: exact rules, input/code hashes, dependency versions and output checksums.
- `previews/`: cropped panel thumbnails for review.

A repeat run must produce identical files for the same ID. A changed method, crop,
image manifest or dependency creates a new ID and preserves earlier results.
To compare methods, join their tables on `folio_id`; retain `analysis_id` in the key.

Verified on 2026-09-21: all 92 repository tests pass; a complete second analysis run
reproduces every output byte; all 231 generated output checksums match the Git index.
The gallery was opened and its `f81r` filter checked visually. `.gitattributes`
preserves archive bytes, including the CSV's standard CRLF record endings.

To reconstruct crop proposals, run `python -m experiments.folio_crops` in the same
environment. It writes `tmp/folio-crop-proposals.json` and does **not** replace the
reviewed crop file. The archived rectangles are the reproducible analysis inputs.

## Heuristics

Rules live in `voynich/folio_description.py`; every threshold is recoverable from
the versioned code and configuration. The first run uses these steps:

1. **Colour.** Resize a panel to at most 600 pixels, remove a 3.5% border and estimate
   parchment colour from the brighter half. Differences in CIELAB colour channels
   identify green-like, blue-like and warm red/brown-like pixels. These are colour
   proxies, not pigment chemistry. Brown ink can enter the warm category.
2. **Composition.** Measure substantial green/blue patches, their bounding boxes
   and centres, a 3×3 occupancy grid, total extent and coarse left/right symmetry.
   A patch must occupy at least 0.08% of the analysed interior. Patch counts are
   not counts of plants, people, leaves or containers. Symmetry is undefined when
   too little cool colour is present.
3. **Marks and circles.** Count bands of locally dark marks without OCR. Propose
   circles with a Hough transform, then require radial edges over at least 50% of
   the circumference and support in 14 of 24 angular sectors. Merge nearby centres.
   This rejects many accidental circles formed by handwriting, but misses faint,
   distorted or incomplete rings.
4. **Description.** Apply ordered rules: supported circles; little cool colour
   with many horizontal bands; distributed colour patches; localised colour;
   otherwise unresolved. Record the fired rule alongside the measurements.

## Table fields

| Columns | Meaning |
|---|---|
| `analysis_id`, `folio_id` | Composite row key; heuristic version and named panel. |
| `image_id`, `scan_label`, `image_path`, `image_sha256`, `catalogue_url` | Trace the panel back to its source. |
| `crop_bbox_normalized` | `[left, top, right, bottom]` in the complete scan, coordinates 0–1. |
| `panel_width`, `panel_height` | Pixels before analysis resizing/border removal. |
| `layout`, `description`, `rules_fired` | Rule-selected layout, readable description and decision trace. |
| `domain` | `unassigned`; v1 does not classify subject matter. |
| `green_fraction`, `blue_fraction`, `warm_fraction`, `dark_mark_fraction` | Fractions of the analysed interior, not the complete photographed rectangle. |
| `horizontal_band_count`, `circle_candidate_count`, `cool_component_count` | Geometric proxies, not textual lines or semantic objects. |
| `occupied_grid_cells`, `cool_grid_3x3` | Cool-colour occupancy in row-major order; occupied means at least 2%. |
| `cool_span_x`, `cool_span_y`, `coarse_colour_symmetry` | Normalised colour extent and reflection similarity; null symmetry when sparse. |
| `palette`, `cool_regions`, `circle_candidates` | Colour flags and the individual spatial measurements; JSON within CSV cells. |
| `quality_flags` | Unvalidated proxies, possible reverse-side show-through, unresolved domain, small panel resolution. |

## What this permits

This is an exploratory image-description dataset. It can support comparisons of
description methods across botanical, figurative, diagrammatic and text-heavy
pages. The [visual review](VISUAL_REVIEW.md) records direct inspection and known
limitations. Thresholds were developed on visible manuscript images; no held-out
semantic accuracy is claimed. The images were not text-masked. This is **not** the
independent two-annotator dataset required for a new confirmatory text/image study.

The user's 2026-09-21 request reopens image acquisition and description. The closed
prediction track, failed decipherment gate and parked association tests remain as
recorded. No transcription, OCR, language model or text/image significance test is
used here. Recto/verso and panels of the same physical foldout must not be treated
as independent observations in later experiments.

## Sources and reuse

- [Voynich.nu folio index](https://www.voynich.nu/folios.html): panel names and scan
  links, accessed 2026-09-21. The machine download used the equivalent non-`www`
  host. Catalogue prose was not copied into descriptions.
- [Yale IIIF manifest](https://collections.library.yale.edu/manifests/2002046) and
  [catalogue](https://collections.library.yale.edu/catalog/2002046): manuscript scans.
- [Yale image reuse guidance](https://www.library.yale.edu/policies/reuse) and
  [public-domain image policy](https://web.library.yale.edu/about/copyright/permission.html).
  These are reproductions of the public-domain manuscript. Yale's manifest also
  carries general copyright boilerplate; it does not supply an explicit CC0 license.
  We retain its metadata and do not invent a new license for the scans.

Required attribution: **Cipher manuscript (Voynich manuscript). General Collection,
Beinecke Rare Book and Manuscript Library, Yale University.** Reference thumbnails
are served by René Zandbergen's Voynich.nu, with manuscript imagery courtesy of
the Beinecke Library. Downloaded Yale JPEGs are retained unchanged; previews are cropped/resized.
