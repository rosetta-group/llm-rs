> AI-compiled research brief (web search, 2026-09-24). Items tagged [U] were not re-verified; counts from the CDLI API are as the agent measured them.

# Proto-Elamite: background brief for a structure-and-arithmetic study

Compiled 2026-09-24. Tags: [E] established, [D] debated, [S] speculative,
[U] unverified (from memory or a single unreviewed source).
"Checked" means I looked at the data or API myself during this research.

---

## 0. Summary in five lines

- The corpus is small and mostly numbers. CDLI has 1,467 transliterated texts, 1,399 after cleaning, with 33,778 tokens. Of these, 11,364 are numerals and 14,906 are non-numerical signs [E; Born et al. 2021].
- The number systems are already worked out: sexagesimal, decimal, bisexagesimal and capacity. The only open question is which system a given ambiguous numeral uses. 5,389 of 7,984 readable notations fit all four systems [E; Born et al. 2023/2025].
- Arithmetic closure between obverse entries and reverse totals is already published: a subset-sum search plus expert checking disambiguated 24 texts [E; Born et al. 2023]. An independent 2026 re-implementation added permutation nulls [E, but not peer-reviewed; nlarch/proto-elamite].
- Proto-cuneiform is the natural known-answer control. It is in the same CDLI database, uses the same N-sign notation, and has about 5,000–7,000 texts. Its number systems were deciphered by arithmetic (Friberg 1978–79; Damerow & Englund 1987).
- There is probably no lexical "total" word to find. Totals are marked by position (on the reverse) plus a collective object sign [E for position; U for absence of a word]. So the Linear A "KU-RO" pattern maps to "which line is the total", not "which sign is the total word".

---

## 1. Corpus facts

### 1.1 Size and sites

| Quantity | Value | Source |
|---|---|---|
| CDLI catalogue records, period "Proto-Elamite" | 1,755 (1,616 with genre Administrative) | checked, cdli.earth/search?period=Proto-Elamite, 2026-09-23 [E] |
| Transliterated texts in CDLI snapshot | 1,467 (2018-06-02 and 2023-02-09 snapshots both) | checked (sfu-natlang/pe-decipher-toolkit and pe-headers data); Born et al. 2022 [E] |
| Cleaned texts (tablets with only unreadable or numeric signs removed) | 1,399 | Born et al. 2019, 2021 [E] |
| Lines / tokens | 11,013 entries; 33,778 tokens; 7,508 broken; 11,364 numerals; 14,906 non-numerical | Born et al. 2021 (Findings ACL) [E] |
| Mean text length | 27 readable signs, 10 non-numerical; max 724 (198 non-numerical) | Born et al. 2019 [E] |
| Older estimates | "just over 1600 pieces", ca. 10,000 lines | Englund 2001/2004 preprint p.5 [E] |
| Recent overview | ~1,700 tablets | Kelley 2026 (Cambridge Element); Monroe et al. 2025 ("over 1,700") [E] |

Sites [D on exact counts; Wikipedia summary of Dahl's work]: Susa (>1,500–1,600 tablets and fragments), Tall-i Malyan/Anshan (33), Tepe Yahya (27, plus blank tablets), Tepe Sofalin (12; Dahl, Hessari & Yousefi Zoshk 2012), Tepe Sialk (5), and one each from Ozbaki, Shahr-i Sokhta and Tepe Hissar.
- https://en.wikipedia.org/wiki/Proto-Elamite_script
- Englund 2001 preprint, MPIWG Preprint 183: https://d-nb.info/1139676466/34

Susa therefore dominates at about 95%. Site-level analyses outside Susa have n ≤ 33 and no power.

### 1.2 Document types [E]

All texts are administrative. At most two texts might be school exercises (Englund 2001 n.8). Englund's three-part structure (2001, Fig. 3, p.7) is:
1. **Heading**: a sign or sign combination with no number that "qualifies all transactions". M157 is the classic header.
2. **Entries**: person or institution signs, then object signs, then a numerical notation. The order within an entry is `signs , number`.
3. **Totals**: usually on the reverse. They combine a collective ideographic notation with the summed number.

Known genres include livestock husbandry and slaughter (Dahl 2005), labour and ration administration (Dahl, Hawkins & Kelley 2018), grain distributions for sowing (plow/yoke texts), and vessels and beer. Hierarchical texts can have up to three levels (Englund 2001 p.8).

Checked (2018 snapshot, 1,467 texts): 1,329 have an `@reverse`. Counting numbered reverse lines gives 330 texts with 1 line, 123 with 2 and 48 with 3; 762 reverses have no numbered line (blank or broken). This is consistent with Born et al.'s filter of about 425 "one or two reverse entries" candidates.

### 1.3 Sign inventory

- **CDLI working list**: signs M001–M521 (after Meriggi), with `~x` variants and `|A+B|` complex graphemes [E; Born et al. 2022]. Snapshot counts: 49 numerical sign names and 1,623 non-numerical sign types. The non-numerical types split into 287 basic signs, 1,087 variants and 249 complex graphemes. 745 of the 1,623 types (46%) are hapax [E; Born et al. 2019].
- **Dahl 2002** (CDLB 2002:1, https://cdli.earth/articles/cdlb/2002-1): about 1,900 non-numerical signs, variants counted separately. About 1,050 occur once and about 1,700 occur ≤9 times. Only a small core is frequent: M288 709, M388 528, M218 453, M9 213, M387 206 [E].
- **Dahl 2019** (TCL 32) estimates 774 signs and variants. This is quoted in Born et al. 2021 [E].
- **Meriggi 1974** list: fewer than 400 entries after merging variants. Mecquenem 1949 listed more than 5,500 "signs", mostly variants. Englund judged Meriggi's list of limited value, with mirror-imaged and mis-grouped signs [E; Englund 2001 p.6].
- **Numeral sign names used in ATF** (checked, 2018 snapshot, 28 distinct N-names). Top counts: N01 5,805; N14 1,979; N39B 1,198; N24 592; N30C 586; N34 334; N30D 320; N45 195; N23 90; N08A 73; N39C 54; N51 41; N48 24; N02 22.

**Implication**: at most a few dozen non-numerical signs have n ≥ 100. Distributional profiles are only viable for the top ~50–100 signs. Profile statistics need shuffled-sign controls, which Born et al. did not run.

### 1.4 Number systems and what they count

The relative values below are from Englund 2001 Fig. 4 and Born et al. 2023/2025 Fig. 1. Commodity attributions are [E] where arithmetic supports them and [D] otherwise.

| System | Structure (from N01 up) | Counts | Status |
|---|---|---|---|
| Sexagesimal S | N01 ×10 N14 ×6 N34 ×10 N45 ×6 N48 (1, 10, 60, 600, 3600) | discrete inanimate objects (vessels, tools); possibly high-status humans | [E] structure; [D] "high-status humans" |
| Decimal D | N01 ×10 N14 ×10 N23 ×10 N51(GAL-qualified) ×10 N54 | animals and low-status humans/labourers (e.g., M388 591 = 5N23 9N14 1N01 in Scheil 1923:45; small cattle M346 in Scheil 1905:212) | [E] no proto-cuneiform counterpart; attributions [E/D] |
| Bisexagesimal B | N01 ×10 N14 ×6 N34 ×2 N51 ×10 N54 (1, 10, 60, 120, 1200) | discrete grain products/rations | [E] |
| B# (framed B) | as B, notation framed by strokes | processed grain products | [D] |
| Capacity C (ŠE) | N01 ×6 N14 ×10 N45 ×3 N34 ... ; fractions N39B (= 1/5 N01?), N24, N30C/D | grain capacity, especially barley; small units also used as ideograms for grain products | [E]. Friberg's 1978–79 discovery that N14 = 6 N01 (not 10) came from arithmetic |
| C#, C" | derived capacity systems | processed grain; emmer (C") | [D] |
| Area A | single text (Scheil 1935:5224) | surface | [D], possibly an import |

Numeral ambiguity [E; Born et al. 2023/2025, Table 1]: of 8,011 intact notations, 7,984 have at least one reading and only 1,899 are unambiguous. 5,389 fit all four of B, C, D and S. The ambiguity is mostly in N01 and N14, whose value ratio differs by system: 10 in S, D and B versus 6 in C.

Mixed tablets [E]: 12 tablets are C+S, 15 are C+D, 4 are C+B and 1 is S+D. No tablet uses two integer systems except one probable error.

### 1.5 What is known about sign functions

| Sign | Proposed function | Evidence | Tag |
|---|---|---|---|
| M157 (and M157~a) | "household/institution" header | position 1, no numeral; HMM state 7 | [E] as header; [D] meaning |
| M136, M218 (as first sign), M059, M305, M327 | headers / account owners | position; Dahl 2019 | [D] |
| M288 | grain container ("gur"-like), the most frequent sign; counted with the C system | arithmetic in C; summary lines | [E] as C-counted object; [D] exact meaning |
| M388 | man / male worker category; possible "Personenkeil" before names; decimal-counted | Scheil 1923:45 total 591; position before "syllabic" strings | [E] human category; [D] exact meaning |
| M124 | parallel worker/overseer category (alternates with M388) | Dahl et al. 2018:25 | [D] |
| M072, M370-complexes | female, child | Kelley 2018 thesis | [D] |
| M346, M367, M006, M362 | ewe, billy-goat, ram, nanny-goat; decimal-counted | Dahl 2005; graphic link to PC UDU | [E/D] |
| M056 (~f) | plow ("PLOW" = 2 N39B grain) | ratio texts Scheil 1935:117 | [E] ratio; [D] meaning |
| M054 | yoke ("YOKE" = 2½ N39B) | Scheil 1935:156 | [E] ratio; [D] meaning |
| M036 | grain ration container, a functional equivalent of PC GAR; about 30 variants | Englund 2001 n.31; Dahl 2005 | [D] |
| M297, M297~b | bread/keg/ale | Friberg 1978; Meriggi | [S/D] |
| M376 | high-status human (or livestock); sexagesimal-predictive | Born et al. 2023; Dahl 2005 | [D] |
| M001, M096, M218, M387, M371, M057, M066 | "syllabic" signs in personal names | stable clusters (Born 2019); Dahl 2019 syllabary | [D] |
| M009 M003~b/~c | administrative postscript | small text group | [S] |

"Total" in PE: there is no dedicated total word in the literature I found. Totals are identified by position (reverse), by larger magnitude and by a collective ideogram such as M288 or M297 [U for "no total word"]. Born et al. 2023/2025 note that some CDLI transliterations label a summary entry explicitly. The 2018 snapshot has no "summary" comment tags and 12 occurrences of "total", so the annotation is sparse (checked).

Worked instance (checked, P008014 = MDP 06, 214, https://cdli.earth/artifacts/8014). The obverse has 13 N01, 2 N14 and 3 N39B. The reverse reads `M288 , 4(N14) 1(N01) 3(N39B)`.
```
C system (N14 = 6 N01):  obverse 13 + 2*6 = 25 N01 + 3 N39B ; reverse 4*6 + 1 = 25 N01 + 3 N39B  -> equal
S/D system (N14 = 10):   obverse 13 + 20 = 33 ; reverse 41                                      -> not equal
```
So one closed tablet fixes the ratio N14:N01 = 6 and the system for 11 ambiguous entries. The object M288 appears only in the first entry and is implicit afterwards (Born et al. 2023/2025 §4.2).

### 1.6 Relations to other scripts

- **Proto-cuneiform** [E]: PE shares numerical sign forms and systems S, B and C. PE adds the decimal system D, which proto-cuneiform lacks (Englund 2001 p.8, 12). Ideograms are mostly not shared, but some are (UDU~M346, APIN~M056, KUR2~M388, SAL~M072). PE headers resemble proto-cuneiform colophons [D] (Damerow & Englund 1989:15).
- **Linear Elamite** [D]: Desset, Tabibzadeh, Kervran, Basello & Marchesi 2022, ZA 112(1):11–60, https://doi.org/10.1515/za-2022-0003. They decipher Linear Elamite as a phonetic script (V, C, CV) from bilingual royal names; the corpus is about 40 inscriptions. They propose that PE and LE are one system at two stages and give a PE–LE sign concordance.
- Kelley, Born, Monroe & Sarkar 2022 (Iranica Antiqua 57; https://anoopsarkar.github.io/papers/pdf/IA57001.pdf) applied Desset's values to the PE corpus. Only 22 signs overlap with Dahl's syllabary. They warn that random value assignments also produce short readable "words". The mapping stays unproven [D].
- Englund (2001 p.6) had argued that PE and LE show "little graphic" connection.

---

## 2. Machine-readable data

### 2.1 CDLI: where and how

**Identification rule** (checked): PE artifacts have `period.id = 5`, "Proto-Elamite (ca. 3100-2900 BC)". Their language is "undetermined" with inline code `qpc`, the same as proto-cuneiform. The ATF line is `#atf: lang qpc` in 1,453 of 1,467 texts (one `qpe`). So PE must be selected by period, not by language. Proto-cuneiform periods are Uruk V (id 2), Uruk IV (id 3) and Uruk III (id 4).

**Current API** (checked 2026-09-23, framework at cdli.earth):
- Search as JSON with embedded ATF: `curl -H 'Accept: application/json' 'https://cdli.earth/search?period=Proto-Elamite&limit=100&page=N'`. Each record has `inscription.atf`, `period`, `provenience`, `genres` and `languages`. It returns 1,755 records (1,616 with `&genre=Administrative`).
- Single artifact: `https://cdli.earth/artifacts/8014` with `Accept: application/json`.
- Paginated listings carry `Link:` headers for rel first, next and last. `/artifacts?limit=2` gives last page 212,778, so the whole catalogue is about 425k records.
- Docs: https://cdli.earth/docs/api. Inscriptions are also served as C-ATF (`text/x-c-atf`), CDLI-CoNLL and CoNLL-U via `/inscriptions/*`. My plain `Accept: text/x-c-atf` request on `/artifacts/8014/inscription` returned empty, so use the JSON route.
- CLI: `cdli-api-client` (npm, MIT, https://github.com/cdli-gh/framework-api-client) runs `cdli export --entities inscriptions --format atf` and `cdli search`.

**Legacy bulk dump** (checked with `gh api`): https://github.com/cdli-gh/data holds `cdliatf_unblocked.atf` (Git LFS, 86,897,831 bytes) and `cdli_cat.csv` (Git LFS, 154,768,722 bytes). It was last updated August 2022, has no LICENSE file, and its README says "Last update was August 2022." The Zenodo mirror of release 2022.08 (https://zenodo.org/records/6975724, DOI 10.5281/zenodo.6975724) says licence "Other (Open)".

**ATF conventions** (checked):
- PE entry: `N. SIGNS , NUMERAL`, for example `2. |M175+M131~d| M305 M263 x M288 , 2(N01)`.
- Proto-cuneiform entry: the order is reversed, `N. NUMERAL , SIGNS`, for example `1. 2(N01) , 1(N24)`. Proto-cuneiform has cases and subcases (`1.a.`, `1.b.`) and columns.
- Numerals are written `n(Nxx)`, meaning digit Nxx repeated n times, for example `4(N14) 1(N01) 3(N39B)`.
- Signs are `M###`, variants `~a`, complex graphemes `|A+B|` or `|AxB|`.
- Damage marks are `#`, `?`, `[...]`, `x` and `X`. Corrections are `<...>` and `!`.
- Structure is marked with `@obverse`, `@reverse`, `@column n` and `$ ...`. Header comments are `# header` (652 in the 2018 snapshot).
- Marginal systems are B#, C# and C". Born et al. drop them.

**Licence, exactly as stated** at https://cdli.earth/terms-of-use (checked 2026-09-23). No Creative Commons licence is named for transliterations or catalogue data.
- Transliterations: "may be freely copied, aggregated and re-used according to common and fair academic practice". CDLI asks that reuse of considerable textual data cite CDLI and its web address.
- Images: non-commercial use only. Copyright stays with the owning institutions; line art belongs to the named authors. Commercial use needs written permission.
- In practice: the ATF can be reused with attribution. Do not redistribute photographs. Label derived data as "CDLI terms of use", not CC0 or CC-BY.

### 2.2 Other datasets and repos (checked with `gh api`)

| Repo | Content | Licence | Notes |
|---|---|---|---|
| sfu-natlang/pe-headers (https://github.com/sfu-natlang/pe-headers) | 1,467 per-tablet ATF files (snapshot 2023-02-09) + `data.csv` header labels (Expert/HMM/LR) | none stated | Corpus behind Born et al. 2022 and the nlarch replication; best "comparable" snapshot |
| sfu-natlang/pe-decipher-toolkit | `data/cdli_atf_20180602.txt.bz2` (60 KB; 1,467 texts), notebook, sign PNGs | LGPL-3.0 (code); data is CDLI | Born et al. 2019 companion |
| sfu-natlang/pe-compositionality | same 2018 ATF + sign images per numeral/sign | none | Born et al. 2021 |
| sfu-natlang/pe-sign-value-data | PE corpus with Desset/Dahl sound values inserted | none | Kelley et al. 2022 |
| sfu-natlang/pe-pc-datasets | n-gram JSON for PE (1 MB) and PC (26 MB, Uruk III/IV admin, ordered/unordered) | CC-BY-SA-4.0 | Counts only, no texts |
| cdli-gh/proto-elamite_data | 1,329 EPS sign drawings | CC-BY-4.0 | Sign images, not texts |
| MrLogarithm/clee | Logan Born's command-line PE environment with DB | none | tooling |
| nlarch/proto-elamite (README links github.com/NicolasMasselot/proto-elamite) | Independent reimplementation of Born et al. numeral conversion + subset-sum/full-sum closure + permutation nulls; 34 tests; MIT code | MIT (code), CDLI data | **Directly overlaps idea (a)**; exploratory, AI-assisted, not peer-reviewed |
| oracc/pcsl | Proto-Cuneiform Sign List | CC0-1.0 | PC sign metadata |
| cdli-gh/proto-cuneiform_signs | PC sign list | CC-BY-4.0 | |
| Gwrhkhsh/4ky (Zadworny) + 4ky web app | PC research scripts; account-type labels | CC0-1.0 (scripts) | PC account-type tags (Zadworny & Gordin 2025) |
| yhynoo/alp | SVM account-type tagger; `4ky_clean.json` (3.6 MB) | none | PC labels |
| MahmoodKhalil57/ProtoElamite, souldriver007/mdp-ancient-scripts, JLPARTIN/PROTO-ELAMITE, others | amateur/AI-generated claims ("N45 = 100", "100% accuracy", "Meluhha") | none | [U]; treat as unvetted, do not cite as prior results |

---

## 3. Prior computational work (do not duplicate)

1. **Born, Kelley, Kambhatla, Chen & Sarkar 2019**, LaTeCH-CLfL (https://aclanthology.org/W19-2516/). Neighbour, HMM and Brown clusterings plus 10-topic LDA over 1,399 texts.
   - They recover a stable "syllabic/PN" cluster (M057, M066, M096, M218, M371) and a livestock cluster (M367, M346, M006, M309).
   - LDA topics reproduce the livestock and labour genres. One topic puts 37.3% of its mass on M288.
   - Repetition is low: no 7-gram repeats, and the most common n-grams occur in ≤3.2% of texts.
   - They ran no shuffled-sign null [E].
2. **Born, Kelley, Monroe & Sarkar 2021**, Findings ACL (https://aclanthology.org/2021.findings-acl.362/). An image-aware BiLSTM language model. Complex graphemes are "at least partly compositional", and they find construction rules. A Transformer underperformed because the corpus is too small. Image models beat text models [E].
3. **Born, Monroe, Kelley & Sarkar 2022**, EMNLP (https://aclanthology.org/2022.emnlp-main.620/). Headers.
   - A 15-state HMM state 7 matches expert header labels with precision 0.93, recall 0.67 and accuracy 0.70.
   - Logistic regression on Transformer attention over the pre-numeral span reaches 92% accuracy, 95% after fixing 25 labels. The majority baseline is 77%.
   - Cohen's κ between expert and regression is 0.766 (0.849 corrected). They propose 18 novel headers.
   - Signs 2–3 carry header information, and sign 1 predicts genre (Cramér V = 0.39) [E].
4. **Born, Monroe, Kelley & Sarkar 2023**, CAWL; corrected as arXiv:2502.00090v2 (2025) because Englund 1996 had misprinted the decimal values (https://arxiv.org/abs/2502.00090).
   - Rule-based readings of 8,011 intact numerals; 1,899 are unambiguous.
   - A subset-sum search over tablets with 1–2 reverse lines disambiguates **24 texts** after expert checking. The P-numbers are not published.
   - A bootstrap (DL-2-ML) classifier with "cautious" rule selection reaches 4-way F1 0.94 (baseline 0.88) and 2-way F1 0.96, on a test set of only 48 items (B 3, C 18, D 14, S 13).
   - Findings:
     - Only 11 of 244 signs occur next to notations from more than one system (M001, M056~f, M059, M096, M124, M218, M305, M327, M371, M387, M388).
     - The M056~f : M288 ratio is 2.5 N01(S) per N01(C).
     - In P009343, M288 amounts are always 4× the adjacent M376 entry and 2× the M367~i entry, across 44 entries.
     - M388 entries carry larger capacities than M124 entries. M288 entries are the largest, M263 among the smallest.
     - The livestock signs M346, M362, M367 and M417 predict decimal; M376 predicts sexagesimal.
   - No null model is reported [E].
5. **Born, Monroe, Kelley & Sarkar 2023**, CAWL (https://aclanthology.org/2023.cawl-1.11/). Deep clustering of sign images to test the PE character inventory [E].
6. **Kelley, Born, Monroe & Sarkar 2022**, Iranica Antiqua 57. Tests Desset et al.'s LE-derived sound values on PE; inconclusive to negative [D].
7. **Monroe, Kelley, Born & Sarkar 2025**, Near Eastern Archaeology 88(4):314–323 (https://doi.org/10.1086/738240). A review of the above (BiLSTM, HMM, topic models). I could not read it (paywalled) [U on details].
8. **Kelley 2026**, *Proto-Elamite: Writing and Society in Early Iran*, Cambridge Elements; **Kelley 2018** DPhil (Oxford), *Gender, Age, and Labour Organization in the Earliest Texts from Mesopotamia and Iran*, https://ora.ox.ac.uk/objects/uuid:afa3362e-1182-43aa-a2b9-d675bd8c585a. Covers the sign sets for gender and age (M388, M124, M072, M370 complexes) [E/D].
9. **Dahl**: sign frequencies (CDLB 2002:1); complex graphemes (CDLJ 2005:3; three complex-grapheme types, and complex capacity signs like M036 with about 30 variants); animal husbandry at Susa (SMEA 47, 2005); "Early writing in Iran" (Iran 47, 2009); labour administration (Dahl, Hawkins & Kelley 2018, AOAT 440); TCL 32 (2019), the full Louvre edition and bibliography [E].
10. **Classic manual arithmetic**: Friberg 1978–79; Damerow & Englund 1987 (proto-cuneiform) and 1989 (Tepe Yahya PE texts); Englund 2001/2004. They established every number-system structure by checking sums [E].
11. **nlarch/proto-elamite, 2026 (not peer-reviewed)**. On the 2023-02-09 snapshot:
    - Subset-sum closures: 151 of 425 candidates, 52 with an unambiguous witness.
    - Full-sum closures: 63, 21 with a witness.
    - Null (obverse bundles permuted among tablets with equal slot counts, 5,000 draws): means 63.4 (subset raw) and 6.3 (full-sum raw).
    - The pool-wide shuffle is a bad null (p = 0.37) because reverse totals are large by construction.
    - The author's caveat: permuting obverse bundles also breaks genre, magnitude and scribe, so the null is too easy [E as a computation; D as inference].

**What stays open** (none of the above does these):
- Blind recovery of the ratios themselves. All prior work *assumes* the Figure-1 ratios.
- A validated arithmetic method with a proto-cuneiform positive control.
- Calibrated nulls conditioned on magnitude and system.
- Automatic detection of implicit-object inheritance and of sub-totals in hierarchical texts.
- Corpus-wide commodity exchange ratios (like 2.5 : 1 or 4 : 1).
- Role classification (object / person / header / qualifier) checked against proto-cuneiform ground truth.
- Testing whether `~variants` behave identically in arithmetic contexts.

---

## 4. Known-answer control: proto-cuneiform

**Why it fits** [E]: both corpora are administrative accounts from 3300–3000 BCE. They share the N-sign ATF, the same S, B and ŠE (capacity) systems, and the same total-on-reverse layout (Englund 2011, "Accounting in proto-cuneiform", https://cdli.earth/files-up/publications/englund2011a.pdf; Englund 2004 "Proto-cuneiform account-books and journals", https://cdli.earth/files-up/publications/englund2004a.pdf). Many ideograms are read through later cuneiform: ŠE barley, UDU sheep and goats, SAL female and KUR2 male workers, GAR ration, DUG vessel, GAN2 field. So sign functions have ground truth.

**Size and identifiers**:
- CDLI catalogue (checked 2026-09-23):
  - Uruk III (period id 4): 5,929 records, 5,185 Administrative, 710 Lexical.
  - Uruk IV (id 3): 1,892 records, 1,876 Administrative.
  - Uruk V (id 2): 200 records.
- Language "undetermined", code `qpc`, the same as PE. Filter by period, then genre=Administrative, to exclude lexical lists. Lexical lists are 1(N01)-per-line and would contaminate the numbers.
- Literature: Englund 2001 gives about 6,000 tablets and >38,000 lines. Zadworny & Gordin 2025 (ALP workshop, https://aclanthology.org/2025.alp-1.3.pdf) give about 7,000 texts, about 5,500 economic, and >800 signs [E].
- Licence: the same CDLI terms of use as above.

**Known structure usable as ground truth**:
- Numerical systems [E]: Damerow & Englund 1987 identify about 15, five of them common: S, B, ŠE, GAN2 (area) and EN. There are also S′, B*, ŠE′/ŠE*/ŠE″, DUGb/DUGc and a U4 calendar system.
- Key ratio: ŠE has N14 = 6 N01, while S has N14 = 10 N01. Friberg found this from sums in the Jemdet Nasr texts, the exact analogue of the PE capacity/sexagesimal ambiguity [E].
- Totals and sub-totals sit on the reverse. In grain texts, entries are converted to grain equivalents and summed into grand totals (Englund 2011) [E].
- There is probably no lexical "total" sign in proto-cuneiform either. ŠU.NIGIN2 "total" appears later, in the Early Dynastic period [U].
- Account-type labels for about 600 texts across 7 classes (cereal 323, animal 125, human 58, field 42, textile 24, dairy 23, fish 22; Zadworny & Gordin 2025; 4ky app) give a genre ground truth [E].
- Colophon/header analogues (Damerow & Englund 1989:15) [D].

**Matched-size control**: subsample proto-cuneiform to 1,400 tablets. Match the distributions of entries per tablet, reverse lines per tablet and damage rate to PE. Run the method blind: hide the ratios and ideogram glosses. Score it against known values. Repeat over ≥20 subsamples to get a power estimate.

**Caveats for the control**:
- Proto-cuneiform entries put the number before the signs, and cases nest (`1.a.`), so the parser must be format-aware.
- Proto-cuneiform sign order inside a case is the transliterator's guess (pe-pc-datasets doc).
- Proto-cuneiform has more systems, which makes the task harder than PE. That is conservative for a positive control.

---

## 5. Testable ideas, ranked by feasibility

Common protocol for all three:
```
freeze PE snapshot (cdli.earth JSON, period id 5) and PC snapshot (period ids 3,4, genre Administrative); record hashes
build parser for both ATF orders; keep damage states
for method M:
    run M blind on PC subsamples matched to PE size   -> positive control score
    run M on PC and PE with shuffled-sign / permuted-bundle nulls conditioned on slot count and magnitude -> negative control
    if PC score passes pre-registered gate and PE beats its null: report PE result
    else: report failure
```

### (a) Blind recovery of number-system ratios, total lines and summed entries (highest feasibility)

- **Task**: treat the ratios between N-signs within each system as unknowns (for example N14:N01 in {5, 6, 10, 12}, N34:N14, N39B:N01). Also treat the system label of each notation as latent.
  - Search ratio vectors and system assignments to maximise the number of tablets where a reverse line equals the sum of all or part of the obverse.
  - Output: recovered ratios, which reverse line is the total, which entries it sums, and the implicit-object inheritance (P008014 pattern).
- **Data**:
  - PE: about 450 tablets with 1–2 reverse lines (checked: 330 + 123). There are 8,011 intact notations.
  - PC: about 5,000 administrative texts.
- **Positive control**: recover S (N14 = 10), ŠE (N14 = 6) and B (N51 = 2 N34) blind on proto-cuneiform subsamples at PE size. The gate: the correct ratio must be the arg-max and beat every alternative by a pre-set margin.
- **Negative control**: permute obverse bundles within strata of equal slot count, system set and magnitude decile. This is sharper than nlarch's null, which the author says is too easy.
- **Expected power**: high. nlarch's full-sum count is 63 against a null mean of about 6. One closed tablet already separates N14 = 6 from 10 (P008014).
- **Novelty vs prior work**:
  - Born et al. and nlarch assume the ratios. Recovering them blind and validating on proto-cuneiform is new.
  - The ratios are known, so the PE result is itself a known-answer check. New information would be the full-sum closure set with P-numbers and sub-total hierarchies.
  - For the one weakly known part, the fractions (N39B, N24, N30C/D), it would be a genuine estimate.
- **Risk**: low novelty on the integer systems. The value is in the validated pipeline, the calibrated false-positive rate and the P-number list, which Born et al. never published.

### (b) Commodity exchange ratios between object signs (medium-high feasibility)

- **Task**: after disambiguation, find sign pairs (A, B) whose quantities keep a constant ratio across entries or tablets, for example M056~f : M288 = 2.5 : 1 and M376 : M288 = 4 : 1 (Born et al. 2023). Equivalence rates identify rations per worker, seed per plow and grain per ration unit. They constrain sign function: a sign with a fixed grain rate is a recipient or a unit, not a commodity.
- **Data**: about 1,900 unambiguous notations, plus those that (a) disambiguates.
- **Positive control**: proto-cuneiform grain-equivalence texts, where ration units like GAR and grain products have known per-unit grain values (Englund 2011; Friberg). Test whether the miner recovers them at matched size.
- **Negative control**: shuffle numbers across entries within tablet and within system, keeping magnitudes. Correct for multiple testing over pairs.
- **Expected power**: medium. Constant ratios need ≥3–5 co-occurring entries per pair. Only frequent pairs qualify, likely 10–40 pairs.
- **Prior overlap**: three ratios are known (plow, yoke, and M376/M367~i in one text). A systematic corpus-wide search with a null is not done.

### (c) Role classification of non-numerical signs by position relative to numbers (medium feasibility)

- **Task**: label frequent signs as header, owner/person, counted object or qualifier. Use these features:
  - position (first sign of the text, immediately before a numeral, entry-initial)
  - the system of the adjacent numeral (Born et al.'s 11 multi-system signs are qualifiers)
  - whether the sign reappears in the total line (collective object)
  - magnitude profile
- **Data**: signs with n ≥ 30, about 60–100 signs.
- **Positive control**: proto-cuneiform with known glosses. Examples of counted objects: ŠE~a, UDU~a, DUG~a, GAR. Examples of human categories: SAL, KUR2, SAG. Examples of officials/institutions: EN~a, SANGA~a, NUN~a. Score macro-F1 against these at matched size. Note the reversed number/sign order in proto-cuneiform.
- **Negative control**: shuffle signs within entry (destroys position) and across tablets within genre (destroys co-occurrence). Report the lift over both.
- **Expected power**: medium for 3 classes, low for fine classes. Headers are already done at 92–95% (Born et al. 2022). The new part is the object vs person split validated on proto-cuneiform.

### (d) Sign-variant identity (low-medium)

- **Test**: do `~a` and `~b` variants of the same base sign have the same arithmetic behaviour: system, magnitude and ratio? Examples: M288 variants, M036's ~30 variants, M297 vs M297~b (Born et al. found different magnitudes).
- **Control**: proto-cuneiform variants whose identity is known (for example ŠE~a vs ŠE~b, where the literature treats them as graphic variants) [U on specific pairs].
- **Power**: low. Most variants have n < 10 (Dahl 2002: about 1,700 of 1,900 signs have ≤9 tokens).

### Not recommended

- Lexical or phonetic matching, for example to Linear Elamite. The corpus has 745 hapax types and no repeated 7-gram, and Kelley et al. 2022 showed random values make "words".
- Deep sequence models: Born et al. 2021 found that Transformers fail at this size.

---

## Key URLs

- CDLI API docs: https://cdli.earth/docs/api ; search JSON: https://cdli.earth/search?period=Proto-Elamite&limit=100 (Accept: application/json)
- CDLI terms: https://cdli.earth/terms-of-use
- Legacy dump: https://github.com/cdli-gh/data ; Zenodo https://zenodo.org/records/6975724
- pe-headers corpus: https://github.com/sfu-natlang/pe-headers
- Born et al. numerals: https://arxiv.org/abs/2502.00090
- Born et al. headers: https://aclanthology.org/2022.emnlp-main.620/
- Born et al. clustering: https://aclanthology.org/W19-2516/
- Born et al. compositionality: https://aclanthology.org/2021.findings-acl.362/
- Deep clustering: https://aclanthology.org/2023.cawl-1.11/
- nlarch replication: https://github.com/nlarch/proto-elamite
- Englund 2001 preprint: https://d-nb.info/1139676466/34
- Englund 2011 PC accounting: https://cdli.earth/files-up/publications/englund2011a.pdf
- Dahl 2002 frequencies: https://cdli.earth/articles/cdlb/2002-1 ; Dahl 2005 complex graphemes: https://cdli.earth/articles/cdlj/2005-3
- Desset et al. 2022: https://doi.org/10.1515/za-2022-0003 ; Kelley et al. 2022: https://anoopsarkar.github.io/papers/pdf/IA57001.pdf
- Zadworny & Gordin 2025: https://aclanthology.org/2025.alp-1.3.pdf
- Kelley 2018 thesis: https://ora.ox.ac.uk/objects/uuid:afa3362e-1182-43aa-a2b9-d675bd8c585a
