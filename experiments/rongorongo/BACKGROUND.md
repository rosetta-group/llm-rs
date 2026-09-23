> AI-compiled research brief (web search, 2026-09-23). Items tagged [U] were not re-verified. Numbers attributed to Rochala 2026 are as that repository reports them.

# Rongorongo: background brief for a codebook-free decipherment test

Compiled 2026-09-23 from web search and fetch. Confidence tags: [E] established,
[D] debated, [S] speculative, [U] unverified or recalled from memory or estimated.
Counts marked "measured here" were computed on 2026-09-23 by streaming the public
kohaumotu.org transliteration pages through a regex. No files were stored.

---

## 0. Headline for the project

A published matched-size known-answer control already exists, and it fails.
Rochala (2026, github.com/rochal/rongorongo, section 23) ran simulated annealing
with a syllable-bigram model, the Ugaritic/Hebrew style of attack. He assigned
syllables one-to-one to the 40 most frequent Barthel head signs, with parallel
copies dropped. [E, for what the repo reports]

- The tablets give about **2,600 adjacent pairs** among those 40 signs.
- A real Rapa Nui text (Apai, about 1,100 syllables) enciphered as unknown signs was
  recovered at **5%** under a 6,000-syllable Rapa Nui model. With 5.4M syllables of
  Māori and Tahitian scripture as a proxy model, recovery was **20%**.
- On a held-out Māori text, recovery was 25% at 2,000 syllables, 62% at 4,000,
  and **100% at 8,000**.
- Conclusion given there: the tablets have about a third of the text needed. The
  one-sign-one-syllable hypothesis can be neither confirmed nor refuted with this
  method.

This matters because the project's solver was validated at 2,600–5,200 letters.
The frequent-sign stream of rongorongo sits at or below the bottom of that range,
before any allowance for logograms or ligatures. The planned control is still the
right first step, and the literature predicts it will fail at matched size.

---

## 1. The corpus

### 1.1 Objects
- **Twenty-six objects, letters A–Z** in Barthel's nomenclature. Most are tablets. I is the Santiago Staff, J and L are reimiro breast ornaments, X is the Tangata Manu figure, and Y is a snuff box made from a cut tablet. E (Keiti) burned at Louvain in 1914 and survives only in photos and rubbings. [E] Source: kohaumotu.org corpus list, adapted from Dederen 1993, http://kohaumotu.org/rongorongo_org/corpus/1.html
- Ferrara et al. 2024 say "twenty-seven wooden objects". Wikipedia says 26 commonly accepted texts, and about half are in good condition. [E] Other claimed pieces, such as the Topaze stick fragment (Cryptologia 49(3), 2025), are outside the canonical set. [D]
- **None of the originals is on Rapa Nui.** They are in Rome, Santiago, St Petersburg, Washington, Vienna, London, Berlin, Honolulu, Paris and New York. [E] Sources: kohaumotu corpus list; Australian Museum, https://australian.museum/learn/cultures/pasifika-collections/rapa-nui-collections/kohau-rongorongo-talking-tablet/

### 1.2 Glyph counts
| Measure | Value | Source |
|---|---|---|
| Dederen/CEIPP per-object sign counts, summed | 14,021 | kohaumotu 1.html table, summed here [E] |
| "Over 15,000 glyphs" | about 15,000 | Wikipedia, Rongorongo [E] |
| CEIPP transliteration tokens, ligature components split | 15,079 tokens on 301 lines; 14,016 carry a 3-digit code; 373 are `000` (illegible) | measured here from kohaumotu.org/rongorongo_org/translit/*.html [E] |
| Whole units, a compound counted once | 11,216 | Rochala 2026 README [E] |
| Stream with one witness per parallel family | 8,688 tokens | Rochala 2026 [E] |
| Largest texts | I Staff about 2,900 (CEIPP: 2,471 codes); A Tahua about 1,825 (1,694); H about 1,580 (1,606); P about 1,163 (1,571); B about 1,135 (1,306); C Mamari about 1,000 (1,004) | Dederen, then measured here [E] |

Counts differ because authors disagree on what one sign is. The kohaumotu table
itself warns that its counts are "very approximate". [E]

### 1.3 Inventory size under each catalogue
| Catalogue | Size | Notes |
|---|---|---|
| Barthel 1958 | about 600 numbers (001–799; "599" or "632" shapes depending on how they are counted) | Numbers group allographs. Some ligatures get their own numbers, but not systematically. [E] Lastilla et al. 2021; Valério/Davletshin |
| CEIPP extended Barthel, as transliterated | 627 distinct 3-digit bases; about 1,993 distinct full codes with affixes (y mirror, x inverted, f, s, V, a/b, etc.) | measured here [E]. Väätäinen 2026 counts L = 633 (numeric) and L = 1,897 (with variants) [E] |
| Rochala head signs / units | 649 head types; 2,067 whole-unit types | [E] |
| Barthel 1971 | about 120 core glyphs | Wikipedia [E] |
| Pozdniakov & Pozdniakov 2007 | **52 basic glyphs said to cover 99.7% of the corpus**; the top 26 cover 86% | Forum for Anthropology and Culture 3:3–36. They argue this matches the roughly 54–55 syllables of Rapa Nui. [D] |
| Horley 2005, 2021 | about 125–130 basic glyphs | Rapa Nui Journal 19(2):107–116; the 2021 book (Rapanui Press, 637 pp.). [D] Väätäinen gives L = 125 |
| Davletshin 2022 | logosyllabic; no full list published | JPS 131(2) [D] |
| Rochala 2026 test | The top 55 head signs cover about 62% of tokens; the other about 600 carry the rest. Shape-based merging removes only about a quarter. Template decomposition does not bring the count near 52. | [E for the measurement; D for what it means] |

For a solver: the plain-syllabary reading needs about 50 phonetic signs. Barthel
has about 600, and whole-unit types number about 2,000. Pozdniakov's 52 works only
if about 600 Barthel numbers are allographs or ligatures of those 52, which is
contested.

### 1.4 Layout
- **Reverse boustrophedon.** Reading starts bottom left, and the tablet is turned
  180° at each line end, so alternate lines are upside down relative to each
  other. [E] Wikipedia.
- The digital transliterations already store each line in reading order, so no
  flipping is needed. [E]
- Ligatures are coded in Barthel notation: `-` juxtaposed, `.` linked, `:` stacked,
  `'` fused, `;` stacked and linked. [E]
  http://kohaumotu.org/rongorongo_org/corpus/codes.html

### 1.5 Parallel passages
- **"Grand Tradition"**: tablets H, P and Q carry essentially the same text
  (Kudrjavtsev, 1940s). [E]
- Many shorter parallels run across A, B, C, E, G, K, N, R and S (Pozdniakov 1996;
  Sproat 2003, "Approximate string matches in the RR corpus"; Spaelti's parallel
  pages, http://kohaumotu.org/blog/rongorongo/parallel-texts/). [E]
- Rochala 2026 sorts the corpus into three families of copied texts, the 380.1
  lists, the two triadic texts (I and one other), three refrain texts, and about a
  dozen unique sides. [E, as reported]
- Why it matters: copies shrink the independent text from about 11.2k to about
  8.7k units. Copies are also free allograph evidence, since scribes swap variants
  in equivalent slots. [E]

### 1.6 The Mamari lunar calendar (C, lines Ca6–Ca9)
- Barthel identified it. Guy (1990, JSO 91:135–149, https://www.persee.fr/doc/jso_0300-953x_1990_num_91_2_2882) reads crescent and circle glyphs as a lunar month with intercalary-night rules. [E that the passage is calendrical; D on the details]
- Horley (2011, JSO 132:17–38) and Davletshin (the OUP chapter "The Rongorongo 'Lunar Calendar' ... and the Type of Script") build on it. [D]
- Rochala 2026 can rebuild a structure that counts to 30. Markers on named boundaries do no better than chance. [D]
- Butinov & Knorozov (1957) proposed a genealogy on the Small Santiago tablet (Gv5–6), with a repeated "son of" pattern. [D]
- Fischer (1997) proposed that the Santiago Staff triads (X-76-Y) are procreation chants. [D, widely doubted]

### 1.7 Dating
- **Ferrara, Tassoni, Kromer et al. 2024**, *Sci. Rep.* 14:2794, https://www.nature.com/articles/s41598-024-53063-7 (CC BY 4.0). [E]
  - Tablet **D (Échancrée)**, *Podocarpus latifolius*: 1493–1509 cal AD (68.3%).
  - Tablet A, *Fraxinus excelsior*, European ash: 19th century.
  - Tablets B and C, *Thespesia populnea*: 18th–19th century. Mamari's ranges span 1694–1840.
- Earlier direct dates: Tablet Q 1812–1836; Berlin tablet 1811–1838 (Journal of Island and Coastal Archaeology 2021, https://www.tandfonline.com/doi/full/10.1080/15564894.2021.1950874). [E]
- Caveat: radiocarbon dates when the tree died, not when the text was carved. Old or drifted wood can be carved late. So the evidence for pre-contact invention is suggestive, not proof. [E caveat; D inference]
- Glyph 67 is read as the extinct Rapa Nui palm, which would put the script before about 1650. [S]

### 1.8 Is it full writing?
- The mainstream view summarised by Wikipedia is that it may be proto-writing or a mnemonic device. [D]
- Routledge (1914–15) called it an idiosyncratic mnemonic. [D]
- **Full-writing side:**
  - Ferrara et al. 2024 cite complex ligatures, long linear sequences and corrections. [D]
  - Pozdniakov 2007 argues for a syllabary. [D]
  - Davletshin 2022 argues for a logosyllabary: 11 of 20 proposed readings confirmed by cross-readings, 7 logographic and 4 syllabic. East Polynesian language. *Waka Kuaka / JPS* 131(2), https://thepolynesiansociety.org/index.php/JPS/article/view/579. [D]
- **Against a pure syllabary:** Rochala 2026 finds that the statistics at matched size look like a word or content-word vocabulary, not a syllable stream. [D]
- Metoro's 1870s "readings" for Bishop Jaussen are held not to be real readings. Rochala finds that Metoro named sign shapes consistently, and that a second reciter, Ure Vaeiko, contradicts him. [E that neither is a decipherment]

---

## 2. Machine-readable transcriptions

| # | Resource | URL | Format | Encoding | Coverage | Licence (exact) | Bulk download |
|---|---|---|---|---|---|---|---|
| 1 | **CEIPP numerical transliteration** (Barthel's *Corpus Inscriptionum Paschalis Insulae*, bequeathed to CEIPP), HTML pages on the archived rongorongo.org, now at kohaumotu.org | http://kohaumotu.org/rongorongo_org/translit/{a,b,mamari,d,...,x}.html; format spec at http://kohaumotu.org/rongorongo_org/corpus/digit.html | One line per text line, e.g. `Da03 600-200:042.041-610-...*`. The original CEIPP files (IBM disks) are fixed 21-column, one sign per row, with an uncertainty flag (`?`/`!`) and a join code | CEIPP extended Barthel: 3-digit code plus affix letters plus join punctuation; `000` = illegible; `(010-020)!` = an estimated lacuna | All 26 objects, 301 lines, about 15.1k tokens (measured here). No separate page for W, Y or Z (Y: 2 lines appear elsewhere) | Copyright page: "There are no restrictions on copying and distributing the contents of this site as long as the sources are acknowledged and the distribution is non-profit." Sources: CEIPP for drawings and transliteration. http://kohaumotu.org/rongorongo_org/copy.html | Yes: about 25 static pages, easy to scrape. The TLS certificate had expired (plain HTTP works). [E] |
| 2 | **Spaelti XML corpus** (Philip Spaelti) | http://kohaumotu.org/Rongorongo/xml/ (A–W, X, Y .xml, plus `corpus.html` codes-only view and `corpus_count.html`); partial GitHub mirror https://github.com/phspaelti/RR-corpus (A–F only, last push 2014) | XML with a schema (`tablet_schema.xsd`): tablet/side/line/glyph, with `<code>`, `<link>`, and SVG path tracings per glyph | Barthel/CEIPP codes, with tracings from Barthel 1958 and Fischer 1997 | Full on the site (A–Y; Z is missing) | Site: "All of these works are as far as I know under copyright… fair use… observe the usual rules of attribution"; commercial use needs the copyright holders' permission. GitHub repo: **no licence file** | Yes, per-tablet XML (A.xml is 13 MB with SVG). [E] |
| 3 | **rongopy** (J. G. de Souza) | https://github.com/jgregoriods/rongopy | `ga_lstm/tablets/raw/*.csv` (line id, CEIPP string); `tablets.json`, `tablets_clean.json`, `tablets_simple.json`; `horley_encoding.py` (Barthel to Horley 2021 map); `horley_parallels.csv` | CEIPP (13 tablets: A B C D E G H K N P Q R S), plus a Barthel-to-Horley reduction to about 130 glyphs | 13 main tablets; no Staff (I) and no small objects | **GPL-3.0** (repo licence). The data derives from CEIPP | Yes, via git clone. [E] Väätäinen checked that the rongopy tablets are the CEIPP transliteration reformatted, 15 of 15 checks. |
| 4 | **rochal/rongorongo** (P. Rochala, 2026) | https://github.com/rochal/rongorongo | Scripts plus a derived corpus; reproduces 29 analyses | CEIPP from kohaumotu; Barthel tracings via Wikimedia Commons | All 26 objects, 11,216 units | **MIT** (code). Data keeps its source terms | Yes. [E] |
| 5 | **rongorongo-catalogue-audit** (I. Väätäinen, 2026) | https://github.com/ipezygj/rongorongo-catalogue-audit ; Zenodo doi:10.5281/zenodo.21964265 | `fetch_data.py` pulls the CEIPP XML from kohaumotu plus the rongopy Horley map. **Nothing is redistributed** | Three catalogues side by side (L = 1,897 / 633 / 125) | 25 inscriptions | **MIT** (code); the data keeps its own licences | Fetch script. [E] |
| 6 | lxgf/rongorongo (research platform) | https://github.com/lxgf/rongorongo | Laravel/Docker app with a DB dump; SVG per glyph instance, scraped from kohaumotu | Barthel, 632 codes | about 11k occurrences, 24 tablets | **MIT** (repo) | Yes. [E] |
| 7 | skolachi/rongorongo; john-agentic-ai-tools/rongorongo-lab; others | GitHub | Derived from kohaumotu or rongopy | CEIPP | Partial | MIT / none / GPL-3.0 inherited | Yes. [E] |
| 8 | Internet Archive "Rongorongo (All Legible Texts)" | https://archive.org/details/rongorongotexts | **Images only** (Barthel tracings, photos) | — | Most texts | Public Domain Mark 1.0 (as asserted by the uploader) | Yes, but it is not a text transcription. [E] |
| 9 | Horley 2021 tracings and Horley encoding | Book, Rapanui Press 2021 | Print only | Horley basic glyphs (about 130) | Full | © publisher | No digital release found. Only the map in rongopy exists. [U] |
| 10 | Pozdniakov 2007 encoding (52 glyphs) | Paper, *Forum for Anthropology and Culture* 3 | Tables in the paper | Pozdniakov | — | © | No digital dataset found. [U] |
| 11 | INSCRIBE 3D models (Lastilla, Ravanelli, Valério, Ferrara 2021; DSH 37(2):497) | https://site.unibo.it/inscribe/en/about-1 | 3D viewer; new transcription of D (212 graphic elements) | Barthel-based with variant marks | Tablet D only (plus others in the viewer) | Not stated | Viewer only. [E] |

**Recommendation:** use #1 or #2, the CEIPP transliteration from kohaumotu.org. It
is the only complete machine-readable corpus, and every other dataset derives from
it. Its terms allow non-profit redistribution with attribution. For a reduced
inventory, the Barthel-to-Horley map in rongopy (GPL-3.0) is the only public one.

---

## 3. Rapa Nui language data

### 3.1 Rapa Nui itself
| Resource | Period | Size | Licence | URL / format |
|---|---|---|---|---|
| **Thomson 1891, *Te Pito te Henua*** (with Ure Vaeiko's 1886 recitations, incl. Apai) | 1886 | about 2,900 syllables of Rapa Nui (Rochala) | Public domain (1891) | archive.org `cu31924105726222`, `tepitotehenuaor00thomgoog` (scan + OCR). [E] |
| **Métraux 1940, *Ethnology of Easter Island*** (Bishop Mus. Bull. 160): legends, chants, lists in Rapa Nui | 1934–35 fieldwork | about 3,000 syllables of Rapa Nui text; with Thomson about 6,000 syllables in 47 types (Rochala) | HathiTrust **Full View** (public domain in the US); no explicit licence | https://babel.hathitrust.org/cgi/pt?id=mdp.39015012116151. [E] |
| **rongopy `rapa_nui_texts/`** (barthel.txt, blixen.txt, fischer.txt) and `ga_lstm/language/corpus.txt` | chants and recitations from Barthel 1960, Englert 1948, Campbell 1971, Métraux 1971, Fedorova 1978, Blixen | corpus.txt about **4,950 words**; the three chant files about 12 KB | Repo GPL-3.0; the underlying texts are copyrighted editions (Barthel, Blixen, Englert) | https://github.com/jgregoriods/rongopy (plain text). [E] |
| Churchill 1912, *Easter Island: the Rapanui speech…* | about 1911 | vocabulary, little running text | Public domain | archive.org `easterislandrapa00churuoft`. [E] |
| Jaussen 1893/94 (Metoro lists, vocabulary) | 1870s | handwritten notes; Rochala reports the OCR is noise | Public domain | archive.org. [U] |
| Englert, *Leyendas* (1939 texts, publ. 1980; Eng. 2001); *La Tierra de Hotu Matu'a* 1948 (grammar + dictionary) | 1930s | Largest body of pre-1940 narrative; part of Kieviet's 124.5k-word "older" subcorpus | **© Editorial Universitaria / Rapanui Press**; not open | print. [E] |
| Manuscript E (Barthel 1978; Frontier 2008) | before 1914, copied after | Largest indigenous-written text | © editions | print. [E] |
| **Kieviet 2017, *A Grammar of Rapa Nui*** (Language Science Press, SDL 12) | Examples drawn from 1910–2010 texts | 630 pp.; **2,000+ numbered glossed examples** (my rough count) plus interlinear texts in Appendix A | **CC BY 4.0** (Open Textbook Library listing) | PDF and LaTeX source at https://github.com/langsci/124 (the GitHub repo has no licence file; the book is CC BY). [E licence; U count] |
| Kieviet's full research corpus | older texts about 1910–40: **124,500 words**; newer texts 1977–2010: **399,000 words**; early 1970s: 14,500 words | about 538k words | **Not released** (Toolbox database, private; includes Programa Lengua Rapa Nui texts) | Kieviet 2016 dissertation §1.6.2, https://research.vu.nl/ws/files/42164853/complete%20dissertation.pdf. [E] |
| Rapa Nui New Testament, *He Vānaŋa o te ꞌAtua* (2018) | modern | largest single Rapa Nui text (NT + OT portions) | **© Wycliffe / Sociedad Bíblica Chilena**; read-only on YouVersion | https://www.bible.com/versions/2164-rap-rapa-nui ; not on eBible. [E] |
| FineWeb-2 `rap_Latn` | modern web text (mostly jw.org, some issuu) | a single 84 KB parquet shard, roughly 30–60k words (estimate) | **ODC-By 1.0** for the dataset; the underlying pages are © their owners (Watch Tower) | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/tree/main/data/rap_Latn. [E exists; U size] |
| Glot500 `rap_Latn` | — | 442 rows, from TeDDi; the sample is **English interlinear glosses**, not Rapa Nui | mixed | https://huggingface.co/datasets/cis-lmu/Glot500 — useless here. [E] |
| Wiktionary (English) | modern | **366 Rapa Nui lemmas** (category count) | CC BY-SA 4.0 | https://en.wiktionary.org/wiki/Category:Rapa_Nui_lemmas ; kaikki.org has no Rapa Nui page (404). [E] |
| **POLLEX-Online** | lexicon | **1,657 Easter Island reflexes** (of 64,413 across 67 languages) | **None stated**; the site asks users to cite Greenhill & Clark 2011, *Oceanic Linguistics* 50(2) | https://pollex.eva.mpg.de/language/easter-island/ (HTML; no bulk download or CLDF release found). [E] |
| Tatoeba | modern | **32 sentences** (API count) | CC BY 2.0 FR | https://tatoeba.org. [E] |
| Wikipedia | — | No rap.wikipedia.org. The Incubator test wiki `Wp/rap` has about 130 pages | CC BY-SA | https://incubator.wikimedia.org. [E] |
| Universal Dependencies / OPUS | — | No Rapa Nui treebank found; OPUS has none beyond Tatoeba | — | [U, a negative search result] |

**Bottom line for Rapa Nui:** open text of the old genres amounts to about **6,000
syllables** (Thomson 1891 plus Métraux 1940, both public domain, OCR needed). Add
the rongopy chant files (about 5k words, from copyrighted editions). The one
large, open, *modern* source is the Kieviet grammar's examples (CC BY), plus
FineWeb-2 jw.org text (ODC-By wrapper over copyrighted pages). Modern Rapa Nui is
heavily Tahitianised and has Spanish code-mixing. Kieviet notes many Tahitian
loans post-date 1940. So modern text is a poor model for "Old Rapa Nui". [E]

**Syllable inventory:** 10 consonants (p t k ʔ m n ŋ v r h) × 5 vowels plus 5
bare vowels gives about 55 short syllable types. Old orthographies often omit ʔ
and vowel length. Rochala sees 47 types in 6,000 syllables. [E]

### 3.2 Proxy Polynesian corpora
| Language | Resource | Size | Licence | URL |
|---|---|---|---|---|
| Māori | 1868 Bible; 1841 NT; Grey 1853/1854 song and tradition collections | about **2.8M syllables** after Rapa Nui-shaping (Rochala) | Public domain (19th c.) | archive.org OCR. [E] |
| Māori | FineWeb-2 `mri_Latn` | 168,520 docs, 214 MB parquet | ODC-By 1.0 wrapper | HF. [E] |
| Māori | eBible `mri` (1952/2008) | full Bible | © Bible Society NZ | not open. [E] |
| Tahitian | 1878 Bible; 1853 NT | about **2.6M syllables** (Rochala). No *k* or *ŋ*, so about a fifth of the Rapa Nui syllabary is missing | Public domain | archive.org. [E] |
| Tahitian | FineWeb-2 `tah_Latn` | 5,023 docs, 9 MB | ODC-By 1.0 wrapper | HF. [E] |
| Hawaiian | eBible `haw1868` Bible | full Bible | **Public Domain** | https://ebible.org/find/details.php?id=haw1868. [E] |
| Hawaiian | FineWeb-2 `haw_Latn` | 96,394 docs, 129 MB | ODC-By 1.0 wrapper | HF. [E] |
| Marquesan | FineWeb-2 `mrq_Latn` | 64 docs, 58 KB | ODC-By 1.0 wrapper | HF — too small. [E] |
| All | POLLEX cognate sets for sound-correspondence mapping | 64k reflexes | none stated | pollex.eva.mpg.de. [E] |

Māori is the best proxy. It keeps *k* and *ŋ*, and its regular correspondences
map onto Rapa Nui: wh→h, w→v. Rochala found the true Rapa Nui text scores better
under the Māori model than under the 6k-syllable Rapa Nui model. [E, as reported]

---

## 4. Prior computational and statistical work

| Work | Method | Result | What it means for a solver |
|---|---|---|---|
| Barthel 1958 | Catalogue and parallels | about 600 numbers; 120 core (1971) | Base encoding. [E] |
| Pozdniakov 1996; Pozdniakov & Pozdniakov 2007 | Frequency analysis; allograph merging using parallels; compare with Rapa Nui syllable frequencies | 52 glyphs cover 99.7%; positional profiles resemble syllables; the repetitive phrasing suggests restricted genres | Upper bound of about 52 phonetic signs **if** a syllabary. [D] |
| Horley 2005, 2011, 2021 | Allograph statistics; lunar calendar | about 125–130 basic glyphs | Horley map available through rongopy. [D] |
| Sproat 2003 | Approximate string matching | Found parallel passages | Copies are near-duplicates; deduplicate them. [E] |
| Melka 2008, 2009 (*Cryptologia* 32(2), 33(1)) | Structure of Keiti; Staff statistics | Staff has a triadic structure around 76 | Staff (about 2,500–2,900 signs) is a distinct genre; handle it separately. [E] |
| Davletshin 2012–2022 | Combinatorics, cross-readings | Logosyllabic: 7 logograms and 4 syllabic signs confirmed out of 20 | A pure one-sign-one-syllable model is misspecified. [D] |
| Valério & Davletshin (about 2018), "Allographs, graphic variants and iconic formulae…" | Allograph classes | Many Barthel pairs are variants | Allograph merging changes results a lot. [D] |
| de Souza, rongopy (2021–23) | GA + LSTM discriminator; later seq2seq GRU trained on frequency-ranked syllables | "Results were not consistent"; glyph frequencies not Zipfian like a syllabary; no assignment claimed | Earlier failed attempt, no matched control. [E] |
| Rochala 2026 | 29 scripted analyses with nulls; SA decipherment with positive control | Positive control fails (5–20% recovery); about 8,000 syllables needed; tablets offer about 2,600 frequent-sign pairs; head-sign inventory is too large for a syllabary under every merge tried; whole units behave like a word vocabulary; forward and reversed text score the same | **The most direct precedent; replicate it first.** [E as reported; not peer-reviewed] |
| Väätäinen 2026 (Zenodo) | Rao et al. (2010) normalised conditional entropy across 3 catalogues | Switching catalogue moves the statistic by 0.318, four times the corpus-vs-shuffle signal (0.077). Real syllabaries (Rapa Nui, Māori, Linear B) at matched size have 2.5–10× more sequential structure than rongorongo | Entropy findings depend on the catalogue; report every result under several inventories. [E as reported; not peer-reviewed] |
| Ferrara group (INSCRIBE) | 3D scanning, new transcriptions | D re-transcribed (212 elements); corrections found | Barthel's transcription has errors. [E] |

Constraints on the solver:
1. **Phonetic sign count:** at most about 50–55 if syllabic. The observed head-sign
   count is about 630–650, and whole units about 2,000.
2. **Ligatures:** about 15.1k tokens split into components versus 11.2k whole
   units, so about a quarter to a third of tokens sit inside compounds. It is
   unclear whether fused components are read in sequence.
3. **Allographs:** unresolved. Barthel, Horley and Pozdniakov give inventories of
   633, 125 and 52.
4. **Word division:** none marked, so word-level methods (the Ugaritic word-match
   constraint) do not apply.
5. **Reading order:** the Staff and the triads may not be linear prose.

---

## 5. Main risks

1. **The corpus is too small for the method.** [E] Rochala's matched control needs
   about 8,000 syllables for 100% recovery. Recovery is 25% at 1,000–2,000 and 62%
   at 4,000. The tablets give about 2,600 frequent-sign pairs after dropping
   copies. This is below the project's own validated range once logograms and rare
   signs are removed.
2. **It may not be (only) phonetic writing.** [D] Options range from a mnemonic
   device or proto-writing (Routledge; the mainstream view) to a logosyllabary
   (Davletshin 2022). Matched-size statistics do not look like a syllable stream
   (Rochala; Väätäinen). A substitution solver assumes one sign equals one
   syllable. If most signs are logograms, a clean control says nothing about the
   real target.
3. **Inventory and allograph ambiguity decides the result.** [E] Choosing Barthel
   (633), Horley (125) or Pozdniakov (52) moves conditional entropy four times more
   than the real signal does (Väätäinen). A solver result under one catalogue is
   not robust unless it repeats under the others.
4. **Transcription disagreements and errors.** [E] Barthel's tracings match the
   wood at only about 0.65 on hand-boxed lines (Rochala). INSCRIBE found
   undocumented signs on D. There are 373 illegible `000` tokens, and several
   texts are badly worn (M, O, T, U, V).
5. **The language model is weak and anachronistic.** [E] Open Old Rapa Nui is
   about 6k syllables. Larger text is either modern and Tahitianised, or locked
   behind copyright (Englert, Manuscript E, Kieviet's corpus, the 2018 Bible).
   The genre also mismatches: the tablets appear to hold lists, chants and
   calendars, while the proxies are scripture.
6. **Duplication inflates apparent size and apparent structure.** [E] The H/P/Q
   family and other copies must be collapsed. Otherwise n-gram statistics and
   solver scores are inflated.
7. **Cultural sensitivity.** [E]
   - The kohau rongorongo are central to Rapa Nui identity. Rapa Nui knowledge
     holders say oral tradition keeps part of their meaning (Pakomio and Tuki,
     Australian Museum page).
   - None of the originals is on the island. Rapa Nui bodies (Ma'u Henua, the
     Council of Elders, CODEIPA) are pursuing repatriation of heritage. Their best
     known claim is for the moai Hoa Hakananai'a (2018; https://hyperallergic.com/easter-islanders-are-visiting-british-museum-to-request-repatriation-of-ancestral-heritage/).
   - Implications: frame outputs as method tests, not "decipherments". Publicising
     a claimed reading could cause harm. Consider CARE principles for Indigenous
     data, and notify or consult Rapa Nui institutions (Museo Antropológico P.
     Sebastián Englert; Ma'u Henua) before publishing any reading. [S on what is
     best practice]
8. **Licences.** [E]
   - CEIPP data: non-profit use with attribution.
   - rongopy is GPL-3.0, which is viral if its files are vendored.
   - The Englert, Barthel and Blixen texts in rongopy are copyrighted editions.
   - The Rapa Nui Bible and jw.org text are ©.
   - Keep derived datasets fetch-on-demand rather than redistributed, as Väätäinen does.

---

## Key URLs
- CEIPP transliteration: http://kohaumotu.org/rongorongo_org/translit/ ; format: http://kohaumotu.org/rongorongo_org/corpus/digit.html ; terms: http://kohaumotu.org/rongorongo_org/copy.html
- Spaelti XML: http://kohaumotu.org/Rongorongo/xml/ ; terms: http://kohaumotu.org/Rongorongo/copyright.html
- rongopy: https://github.com/jgregoriods/rongopy
- Rochala 2026: https://github.com/rochal/rongorongo
- Väätäinen 2026: https://github.com/ipezygj/rongorongo-catalogue-audit
- Ferrara et al. 2024: https://www.nature.com/articles/s41598-024-53063-7
- Davletshin 2022: https://thepolynesiansociety.org/index.php/JPS/article/view/579
- Lastilla et al. 2021: https://doi.org/10.1093/llc/fqab045
- Kieviet grammar: https://langsci-press.org/catalog/book/124 ; https://github.com/langsci/124
- Métraux 1940: https://babel.hathitrust.org/cgi/pt?id=mdp.39015012116151
- Thomson 1891: https://archive.org/details/cu31924105726222
- POLLEX: https://pollex.eva.mpg.de/language/easter-island/
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
