# Scope: Luwian, Hurrian and Keftiu data for a fifth round

Status: scoping only, 2026-09-23. Nothing below is downloaded yet.

## Finding: hand-typing is mostly unnecessary

| Need | Source | Size | Licence | Effort |
|---|---|---:|---|---|
| Hurrian running text | [TLHdig Beta 0.3](https://zenodo.org/records/20328284), language tag `Hur` | ~140,000 signs (~45,000 words) | CC BY 4.0 (Zenodo record) | download 74 MB zip, parse XML |
| Hattic running text | same, tag `Hat` | ~46,000 signs (~15,000 words) | same | same |
| Cuneiform Luwian | same, tag `Luw` | ~37,000 signs (~12,000 words) | same | same |
| Palaic | same, tag `Pal` | ~5,700 signs | same | same |
| Hittite, Akkadian (same scribes, same format) | same, tags `Hit`, `Akk` | 2.9 M and 0.15 M signs | same | same |
| Hieroglyphic Luwian (Iron Age) | [ACLT](https://luwian.web-corpora.net/) | ~2 M tokens (all Luwian) | not stated | search interface only; no export |
| Keftiu names, BM EA 5647 | Peet 1927, *Essays in Aegean Archaeology*, pp. 90–99, [archive.org DLI scan](https://archive.org/details/in.ernet.dli.2015.281391) | 8 names | public domain (published 1927; Peet died 1934) | read 10 page images, type 8 names |

Language counts are from the [TLHdig-TF sign-language report](https://github.com/alexsosn/TLHdig-TF/blob/main/reports/sign-language.md).
TLHdig transliterations are syllabic cuneiform (`ḫa-at-ti-li`), so they convert to Linear B-style
spelling with the round-one rules and one extra step (join signs, drop logograms in capitals).

## What each source could test, and its power

1. **Grammar profile (strongest case).** Round four's name-profile control passed (Linear B → Greek
   names 20 of 20), and length matching kept it. TLHdig gives running text for Hittite, Luwian,
   Palaic, Hurrian, Hattic and Akkadian in one format, so the profile test can compare Linear A
   with Anatolian and Hurrian-area languages directly. Hattic uses prefixes, and Linear A shows
   initial `a-` / `ja-` / `i-` alternations. The Greek control needs a running-text reference;
   round four's (Wiktionary forms) passed only 15 of 20. The fix is length matching, declared in
   advance.
2. **Frequent-word matching.** TLHdig gives word frequencies, which Wiktionary did not. Linear A
   transaction words (`ku-ro`, `ki-ro`, `po-to-ku-ro`, `a-du`) are frequent. Matching only against
   each language's top few hundred forms lowers the chance rate. Round one's failure suggests low
   power, so this needs its Linear B control first.
3. **Keftiu names.** Verifying 8 names is worthwhile for the record, but 8 items cannot reach
   significance: in round four, chance alone gave 5.9 of 7 names a look-alike.

## Proposed fifth round

```text
download TLHdig 0.3 (74 MB) and the Peet scan (9.5 MB PDF); pin hashes
convert TLHdig words by language tag to Linear B-style spelling
write protocol: length-matched profiles; frequent-word matching; gate as before
controls first: Linear B entry words -> Greek names; Linear B all words -> Greek running words
only if a control passes: run Linear A against Hittite, Luwian, Palaic, Hurrian, Hattic, Akkadian
type the 8 Keftiu names from Peet's plates into lists.json, marked verified
```

Estimated work: half a day of code, under an hour of CPU. No paid compute.
