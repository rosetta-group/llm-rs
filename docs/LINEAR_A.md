# Linear A: language tests, accounting, kinship and source audits

**Stopped at the user's request, 2026-09-24.** The [final closeout](LINEAR_A_CLOSEOUT.md)
records conclusions, verification and conditional reopening requirements. No further work is
queued. Earlier “next” suggestions below and in experiment reports are historical, superseded
by that decision. They do not authorize another search or experiment.

This page preserves the chronology of the original `linear-a` rounds (2026-09-23) and
`codex/linear-a-audit` follow-ups (2026-09-24). It contains no translation. Failed controls
retire the tested methods; they do not establish a limit on all possible methods.

Historical round records: [one](../experiments/linear-a/REPORT.md),
[two](../experiments/linear-a-context/REPORT.md), [three](../experiments/linear-a-names/REPORT.md),
[four](../experiments/linear-a-probes/REPORT.md), [five](../experiments/linear-a-tlhdig/REPORT.md).
Rounds one to three were frozen by a `freeze.json`; rounds four and five committed their protocol
and lists before any run.

## The question, and how it differs from Voynich

Linear A is the reverse of the Voynich problem. Most sign sounds are roughly known, because
about 60 signs share a shape with Linear B signs of known value; the language is unknown. So
the Naibbe task (recover a key, language given) becomes a language test: do Linear A words
look like the words of some known language more than chance allows?

The corpus is small and administrative: 1,884 records, 696 readable word types of two or more
signs, and 330 of those in *entry* position (a number follows, as for a person or place on a
list). Every method had to pass first on Linear B, which is Greek, cut to Linear A's size.

## What the original data checks support

| Check | Result | Chance |
|---|---:|---:|
| Linear B Cretan place names found in Linear A (`pa-i-to` Phaistos, `se-to-i-ja`) | 2 of 14 | 0.025 |
| Linear A word types also found in Linear B | 94 | 72.7 (max 88) |
| `ku-ro` "total" equals the sum of its entries (HT 13: 130 = 130) | 8 of 37 | — |

The assumed sign values reproduce known place names, and about 20 Linear A words survive into
the Linear B archives beyond chance. These support the values; they say nothing about grammar.

## Why the original language tests did not identify a language

| Round | Method | Greek found in Linear B control | Needed |
|---|---|---:|---:|
| One | whole-language lexicons, exact spelling match | 10% | 90% |
| Two | as one, plus: is the match a name where the tablet puts a name? | 0% | 90% |
| Three | entry words only, against proper-name lists | 0% | 90% |

1. **Spelling erases the words.** Linear B spelling drops final consonants, merges r and l, and
   does not write voicing. 37% of Linear-A-shaped pseudo-words match some Greek lemma exactly;
   real Mycenaean words match at 45%. True matches exist (`a-ko-ra` ἀγορά) but are only
   8 points above chance.
2. **Position adds nothing detectable.** Greek-matched names sit in entry position 35% of the
   time and Greek-matched common words 34%. Most Linear B names are absent from dictionaries.
3. **Name lists miss Bronze Age names.** Only 9 of 32 Mycenaean names appear in the classical
   Greek name list; Linear B entry words match Greek names *less* often than chance words do
   (6.5% against 8.1%).

A fourth round tried seven targeted probes (Egyptian and Keftiu names, gods, trade words, name
profiles, spelling rules, role transfer); none passed its threshold
([report](../experiments/linear-a-probes/REPORT.md)). Greek scribes turning Minoan `-re`/`-ru` names
into `-ro` gave 12 normalised pairs against 3.1 by chance. The source audit below retains
six and archives the specific adaptation lead after its stricter control fails.

A fifth round compared grammar profiles with Anatolian and Hurrian running text (TLHdig). Linear A
came out nearest Hittite in 20 of 20 samples, but so did shuffled Linear A syllables; the match
comes from syllable and vowel frequencies (cuneiform, like Linear A, rarely shows *o*), not from
words ([report](../experiments/linear-a-tlhdig/REPORT.md)).

The tested lexical methods do not reliably find Greek where the answer is known. Their failure
does not establish that the corpus lacks information available to other representations or methods.

## Repair audit, 2026-09-24

The original reports and frozen code are preserved. New implementations, protocols, source
hashes and results are in [the audit](../experiments/linear-a-audit/REPORT.md).

1. **Duplicate words were mistaken for changing endings.** The old profile gives two copies
   of `ka-ta` 100% prefix and suffix alternation. The repaired profile deduplicates types and
   samples references without replacement, with exact common length quotas. Luwian now recognises
   itself 19/20 times and Palaic 20/20 (Palaic uses 164 types). Their original control failures
   therefore did not establish an information limit.
2. **The profile still cannot test word structure.** Both real and globally shuffled Linear B
   classify as Greek 20/20. The corrected negative-control gate fails, so corrected Linear A
   profiles are not scored. The old Hittite result and shuffle diagnosis are reproducible in code.
3. **The spelling-correspondence threshold is now reachable.** With 9,999 null draws, the
   discovery-selected rule's held-out result is p = 0.1653. The pre-named `-re`/`-ru` → `-ro`
   pattern has 12 pairs against a null mean of 2.9244, p = 0.0001 (zero null exceedances).
   That is stronger exploratory evidence under this null, not independent confirmation: the
   complete set was already examined before the original protocol.
4. **One structural question was tested.** A sign-only ending model predicts whether unseen
   words precede numbers, compared with a bag-of-signs/length baseline. It retains unknown
   sign identities and withholds inscriptions and complete word types. Only 4/20 Linear B
   controls at the target size pass, against 18 required; shuffled controls pass 0/20. Linear A
   is not scored. This retires that simple ending model, not structural approaches generally.

The separate trade-word control also had an unreachable threshold (50 null draws). Its
999-draw correction is recorded in the audit report. No repaired result identifies a language.

## Correspondence source audit, 2026-09-24

The [12-pair audit](../experiments/linear-a-correspondence/PAIRS.md) keeps literal sign labels,
uncertainty and word boundaries. Six candidates survive. For example, the old code made
`a-ka-ru` match `]a-qa-ro` by merging q and k; HT 117a has conflicting `te-*56-re` and
`te-ja-re` source layers. The valid qa-qa pair is restored to `qa-qa-ru → qa-qa-ro`.

The [frozen conditional ending tests](../experiments/linear-a-correspondence/REPORT.md) keep
real stems and observed ending frequencies. Six matches beat the length-conditioned null
(mean 1.5342, p = .0011), but not the null also preserving final onset (mean 4.2191,
p = .1605). Both were required. The historical normalised inventory still passes both
(p = .0001 and .0049), so source quality materially changes the conclusion.

Archive the specific re/ru → ro adaptation lead after its statistical gate fails. The six pairs
remain unresolved spelling observations; broader overlap is not refuted. Reopening needs
new source adjudication or independent context-compatible attestations, not another split
of these exposed words. The audit uses 208 strict A types and 2,984 B types of length >=3;
its conservative filters also reduce coverage and power. No language is identified.

## Accounting-program pilot, 2026-09-24

The [new pilot](../experiments/linear-a-ledger/REPORT.md) uses anonymous words to choose among
fixed arithmetic operations on commodity-specific integer quantities. It passes software
checks on synthetic accounts, including prediction on unseen objects and target-number hiding.
All 34 focused Linear A tests pass.

The natural-data preflight is **not evaluable**: the strict parser yields seven eligible
Knossos objects, below the minimum ten and insufficient for the 33-object A target. Neither
natural corpus was fitted and no null simulations ran. This is a source-representation and
coverage limit, not evidence against accounting structure. Units, fractions and damaged-name
lines need a source-checked parser/benchmark before a meaningful control is possible.

The older ku-ro check's 8/37 matches use integer parts only. HT 13's displayed 130=130 does
not check the complete quantities because fraction signs were ignored. The new pilot treats
fractions as barriers; it does not solve their values or claim a new word meaning. The
[follow-on benchmark plan](../experiments/linear-a-ledger/BENCHMARK_PLAN.md) specifies the
needed quantities, source-supported boundaries, units and evaluator function labels.

## Source-checked accounts and parentage feasibility, 2026-09-24

The [account benchmark](../experiments/linear-a-account-benchmark/REPORT.md) contains nine
accounts on eight objects, with published boundaries/function labels and exact relative units.
Two balance strictly; six require uncertain readings or incomplete text; one oil/jar negative
is dimensionally incompatible. Conditional readings give five balances and three mismatches,
including the published PY Jn658 error. This is supplied-boundary development, not function
recovery or a replacement for the 33-object control. All inspected objects remain development.

The [parentage check](../experiments/linear-a-kinship/REPORT.md) supplies five source-backed
Linear B fixtures on three objects. It distinguishes a named child-parent relation from
“X and daughter”, where the child is unnamed. The Etruscan prototype depends on supplied name
spans and morphology, neither established for these Linear A slots. A strict graphic scan
finds 48 rows on 22 objects; both apparent two-word rows contain a transaction/logogram in
per-occurrence source annotations. No kinship meaning follows. Name-slot substitution remains
a research direction; it needs independently reviewed person/designation slots and more
known-language controls. All 44 focused Linear A tests pass.

## HT85/117 person-slot source audit, 2026-09-24

The [follow-up audit](../experiments/linear-a-person-slots/REPORT.md) reads the full
Davis–Valério chapter and checks GORILA images/transcriptions against live SigLA. Its
43-row inventory covers four faces of two objects: 32 counted word entries, three counted
logograms, five headers, two totals and one divider. HT117's 17 word entries remain possible
people/designations, without certified name or gender labels.

The edition supports `te-ja-re` provisionally, while SigLA's conflicting `te-*56-re` remains
in the frozen source. This supplies a documented working reading, not a retest of the failed
sound-correspondence lead. HT85b has eight counted words and three logograms, not eleven names.
`qa-A310-i` has count 1 on HT85b but 3 on HT122a, so a one-unit entry does not certify a person.

Repeated headers and `di-ki-se` connect HT87/117, but no parentage formula distinguishes
kinship from occupation, responsibility or place affiliation. This led to the completed
`qi-tu-ne` heading/count comparison below. Literal
concordances are source-key counts, not independently deduplicated physical objects:
PH31a and PH(?)31a share museum inventory HM1609. All 49 focused tests and seven freezes
verify; no semantic classifier was fitted. Independent review of the manual inventory is pending.

## Qi-tu-ne heading/count comparison, 2026-09-24

The [HT7 follow-up](../experiments/linear-a-qi-tu-ne/REPORT.md) confirms `qi-tu-ne 1` on
HT7b .1 in the photograph, drawing and published transcription. It is the same sign sequence,
AB21f–AB69–AB24, that occurs in the HT87/117b headings. Three reviewed occurrences therefore
show one count-associated use and two heading uses on three physical objects. HT7a has VIR;
whether its personnel unit carries onto face b remains an assumption. Overwriting is preserved.

The contrast rejects exclusive unnumbered-heading use, but it cannot choose a semantic class:
a counted person can head a group under their responsibility, while an occupational class can
be counted through one member and head a list of members. Place/institution and household
readings also remain possible. No parentage or gender label follows. Before further semantic
modelling, use source-labelled known-language controls that distinguish names and categories
despite shared positional roles. All 49 focused tests and eight follow-up freezes pass.


## Name versus designation control, 2026-09-24

The [known-language control](../experiments/linear-b-person-role/REPORT.md) freezes 38
source-labelled cases on 19 Linear B objects before evaluation (`2574eef`). Its 19 personal
names and 19 designation cases include both classes in headings and counted entries. Roles,
entry boundaries and quantity bins are supplied, while spelling, morphology, site and series
are hidden. Each held-out object excludes all its faces from training; every training object
supplies one vote per exact signature, irrespective of roster length.

The primary layout rule achieves 37.4% class/object-balanced recall, 39.4% coverage and
95.1% weighted conditional accuracy: 15 correct, one wrong, 22 abstentions. Recall and coverage
fail their 90% gates. The forced-majority diagnostic reaches only 57.4% recall. A name and an
occupation can share all four features (role, quantity, position, repetition); these collisions
put the in-sample deterministic ceiling at 84.9%, below the gate even with perfect fitting.
This is a limit of the declared features on this sample, not all linguistic methods.

Zero of 177 evaluable whole-object label-swap negatives pass; 22 of the 199 draws lack the
required class/role representation. The negative-control gate passes, the positive one fails.
No Linear A is scored and no parentage/gender label follows. Labels derive from Ventris &
Chadwick (1956) and modern DĀMOS transcriptions, without independent specialist adjudication;
this is a curated development challenge. Further work needs new evidence such as relational
frames and morphology with known-language rival controls. All 57 focused tests and nine
follow-up freezes pass; earlier results remain unchanged.


## Relational family-reference control, 2026-09-24

The [relational follow-up](../experiments/linear-b-relations/REPORT.md) adds 19 source-labelled
expressions on 16 objects, with names/name-derived spans, clause boundaries, line joins and
terminal -qe segmentation supplied. Seven FAMILY cases occur on six objects; 12 OTHER cases
cover service, occupation and ordinary co-listing. Seven unresolved cases stay outside scoring.
FAMILY includes patronymics; it does not itself identify an immediate parent-child edge.

The form-only diagnostic recognizes the three intact i-*65 expressions from other objects,
reaching 50% family recall and 41.7% balanced recall. It misses the daughter forms and calls
the masked patronymic OTHER. The primary ordered frame/form rule abstains on all 19 cases:
11 signatures are unseen, eight have only one supporting object, and two were required.
The gate fails coverage. All 196 evaluable label-swap negatives also abstain everywhere;
zero negative passes therefore provide no semantic validation. Three draws fail representation.

The expanded corpus retrieval records 59 hits: 12 intact exact forms, 22 intact attached-tail
candidates and 25 uncertain-text candidates. It repairs the earlier search's omission of
literal i-*65/i-*65-qe without making an exhaustive kinship census or changing old frozen outputs.
MY Au102 i-jo-qe and KN Vs1523 i-jo remain disputed. MY Oe106's daughter reference leaves
argument binding unresolved. Four family cases describe unnamed children, two have disputed
binding, and one permits a wider-lineage reading; zero directed edges were inferred.

Sources and code were frozen at 095dbdc before scoring. All 66 focused tests and ten follow-up
freezes pass. No Linear A scoring or Etruscan modification occurred. This led to the Theban
attached-u-jo versus *65/FAR comparison below; an ending match alone cannot resolve it.


## Theban *65/FAR source audit, 2026-09-24

The [Theban follow-up](../experiments/linear-b-theban-65/REPORT.md) preserves 13 selected objects
and compares the proposed son reading with a separate commodity interpretation. Gp227's -u-jo
provides a published linguistic parallel, but its name differs from the Fq counterpart; identity
is not independently established. Original TOP surfaces and the full AGS response were not
obtained, so published spacing claims remain attributed rather than newly verified.

Three of six conversions in James's published quantity table disagree with its printed unit
components: Fq252 186 versus 178 Z, Fq254[+]255 169 versus 86 Z, Fq277 525 versus 641 Z.
The audit preserves both columns, gaps and qualifications. These are conditional component
calculations, not corrected tablet totals. The two selected accounts, Fq214 and Fq254[+]255,
have conditional visible disputed-allocation shifts of 8 and 16 Z, but neither has a complete
secure body, total and scope. Zero accounts qualify for balance adjudication; zero semantic
predictions or directed family edges are exported.

Frozen at `aeb16d9` before generated outputs, after exploratory source inspection. All 74 focused
tests and 11 follow-up freezes pass; the result reproduces exactly. The next evidence needed is
TOP's Fq236 pp94–95 drawing/transcription comparison, relevant Gp124/Gp227/Fq254 surfaces and
the direct AGS2003 response. No Linear A scoring or earlier frozen-file change occurred.


## Theban image-access stopping point, 2026-09-24

The [bounded image search](../experiments/linear-b-theban-images/REPORT.md) obtained no full-context
image of Fq236, Gp124, Gp227 or Fq254[+]255. Judson2016 reproduces a Gp124 glyph drawing, but
its neighbours are absent. LiBER explicitly excludes Theban archival documents; the four pinned
DĀMOS records supply catalogue links rather than photographs. A targeted museum-volume check
also supplied no identified target image.

Later discussions sharpen the qualification: Pierini2018 reports Fq236's reassignment from hand310
to304, while Judson2016 and Pierini disagree about whether this damaged tablet's spacing and
series context establish syllabic use. No distances were measured or labels changed. The lead
is parked until the [specified scans](../experiments/linear-b-theban-images/ACQUISITION.md) arrive;
repeat searches and modelling of transcription whitespace are not the next experiment. This
manual source-access record adds no freeze; all 11 earlier follow-up freezes remain intact.

## Data and licences

Linear A: Navarre-AI collation of SigLA and lineara.xyz (CC BY; SigLA fields CC BY-NC-SA 4.0).
Linear B: DĀMOS, 5,932 documents (CC BY-NC-SA 4.0). Lexicons: Wiktionary via kaikki.org (CC BY-SA),
LAMAN (CC BY-SA 4.0), Oracc (CC0). Running text: TLHdig Beta 0.3 (Zenodo 20328284, CC BY 4.0).
Keftiu names: Peet 1927, *Essays in Aegean Archaeology* (public domain). Results derived from SigLA
and DĀMOS carry CC BY-NC-SA 4.0. Locally archived sources sit in `artifacts/linear-a-sources/` (git-ignored) and
are hashed in the relevant manifests/freezes. Web-only readings are identified in source notes;
no local PDF or complete automatic restoration is claimed for those sources.
