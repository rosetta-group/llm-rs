# Decipherment-method benchmark and Voynich evidence

> **How to read this file.** It is the append-only research record: every experiment, its
> fixed settings, result, limits and implication, in the order the work happened. For the
> short version read [docs/OVERVIEW.md](docs/OVERVIEW.md); for one table of results read
> [docs/RESULTS.md](docs/RESULTS.md); for terms read [docs/GLOSSARY.md](docs/GLOSSARY.md).
>
> Sections: overall goal · evidence required · current work (newest first) · image studies ·
> archived matched-context suite · what we tried and found · what we know · recovery evidence · resources.

Updated: 2026-09-23. This is the current project overview and research record.
Older reports preserve earlier experiments and may contain superseded next steps.

## Overall goal

The current deliverable is a validated decipherment-method benchmark with positive
controls and honest Voynich negatives. Translation into English or Italian remains the
long-term motivation; it is not in reach on the current evidence. A methods contribution
may be publishable, but acceptance is not established by these experiments.
Linear B remains deferred until the method earns further testing.

**Meaning recovery:** an interpretation of the source that can be checked against independent evidence.
**BPC:** bits per normalized transcription character; lower means better prediction.
**Validation:** pages used to develop and compare methods.
**Final test:** reserved pages that have not been scored or used to choose models.

The initial hypothesis was to adapt an LLM's language-dependent layers while preserving
its higher-level knowledge. We have not established a clean separation between those
functions or shown that selective adaptation recovers meaning. TransformerLens and sparse
autoencoders remain possible diagnostic tools, not sources of verified word meanings.

## Evidence required before a translation claim

```text
Build reliable text preparation and evaluation
Archive the closed prediction track; reopen only for a specific falsifiable mechanism
Record the completed matched-context comparison and its limits
Run controlled recovery with verified ciphertext and hidden original passages
Improve word recovery on fresh passages, then reduce supplied cipher assumptions
Test which parts of the method can work without paired Voynich plaintext
Ground Voynich interpretations in independent evidence
Render supported meanings in English or Italian, with uncertainty and alternatives
```

We do not know an accepted Voynich passage in English or Italian to use as a label.
There may be multiple interpretations compatible with the same text statistics.
The project therefore needs evidence outside next-character prediction: verified
recovery on known examples, consistent mappings on unseen passages, and independent
text/image associations where suitable annotations exist.

Success is not a plausible English paragraph. A proposed reading must use consistent
rules, account for repeated forms, and make checkable predictions beyond the material
used to invent it. We must also record contradictions and competing readings.

## Current work: benchmark, not another prediction sweep

**2026-09-24: Linear A repair audit and one sign-only structural test completed.**
Branch `codex/linear-a-audit`; protocols and source/code hashes committed before corrected
runs. Historical code and results remain intact; rounds one to three still verify.
- Corrected duplicate-sensitive profile features and sampled unique types with exact common
  length quotas. Luwian self-identification improves from 7/20 to 19/20; Palaic from 0/20 at
  189 types to 20/20 at 164 types. Other sampling/scaling corrections also changed, so this
  is not a deduplication-only ablation. Real and shuffled Linear B both identify as Greek
  20/20; the negative gate fails and corrected Linear A is not scored.
- Corrected probe 6's unreachable threshold with 9,999 null draws and the original split,
  discovery stage and random streams. The discovered re → ro rule's held-out result remains
  negative: 2 pairs versus 0.7265 expected, p = 0.1653. The pre-named re/ru → ro pattern has
  12 pairs versus 2.9244, zero null exceedances, p = 0.0001. That strengthens an exploratory
  correspondence under the existing null, not independent confirmation or language identification.
- Also repaired the trade-word control: 50 null draws could not reach p < 0.05/7. With 999
  draws per sample, 3/20 pass versus 18 required; gate still fails.
- Made the old profile's formerly interactive shuffle, pseudo-word, reference-size and o/u
  diagnostics reproducible with explicit seeds. [Audit report](experiments/linear-a-audit/REPORT.md).
- Tested one structural question: whether final sign/sign-pair features predict numeric-entry
  position for unseen types on withheld inscriptions, above a sign-bag/length baseline. Unknown
  sound values remain sign IDs; damaged runs are excluded. Only 4/20 size-matched Linear B
  samples pass (required 18); shuffled controls 0/20. Mean balanced-accuracy gain is 0.53 points.
  Linear A is not scored. [Structural report](experiments/linear-a-structure-v2/REPORT.md).
- The first structural driver stopped on a missing NumPy import before reporting any score.
  The import-only v2 was frozen and committed before the run; an added driver test covers it.
  All 16 focused tests pass. No paid compute or new downloads.

The audited methods are retired. The earlier “information, not compute” conclusion is narrowed:
these tests fail, but no general information limit was established. The full-set correspondence
remains an exploratory finding that would need independent evidence for confirmation.

**2026-09-23: Linear A track closed after five rounds; no language identified.**
Branch `linear-a`. Linear A's sign sounds are roughly known and its language is not, so each
method had to find Greek in Linear B (DĀMOS, 5,932 documents) at Linear A's size first
(696 readable word types). Sign values pass a sanity check: 2 of 14 Linear B Cretan place names
occur in Linear A, chance 0.025. Summary: [docs/LINEAR_A.md](docs/LINEAR_A.md).
- Round one, whole-language lexicons under Linear B spelling: Greek found in 10% of control
  draws against a 90% gate. 37% of random Linear-A-shaped words match some Greek lemma.
  [Report](experiments/linear-a/REPORT.md).
- Round two, plus name-versus-tablet-position agreement: 0% of 20 on Knossos.
  [Report](experiments/linear-a-context/REPORT.md).
- Round three, entry words against four proper-name lists: 0% of 20; only 9 of 32 Mycenaean
  names are in the classical list. [Report](experiments/linear-a-names/REPORT.md).
- Round four, seven targeted probes: none below p = 0.007. `-re`/`-ru` → `-ro` gives 12 pairs
  against 3.1 (p = 0.0099), but 100 null runs could not reach 0.007 and the rule was seen before
  the protocol. [Report](experiments/linear-a-probes/REPORT.md).
- Round five, grammar profiles against TLHdig languages: gate passed, Linear A → Hittite 20 of 20.
  Shuffled syllables also go to Hittite 20 of 20, so the match reflects syllable and *o*/*u*
  frequencies, not words. [Report](experiments/linear-a-tlhdig/REPORT.md).

No method can identify or rule out a language for Linear A at this size. The limits are the
spelling, the corpus and the name lists, not compute. CPU only; downloads approved by the owner.
Next candidate: Rongorongo, which has a known language and about 15,000 glyphs
([docs/UNDECIPHERED.md](docs/UNDECIPHERED.md)).

**2026-09-23: length scaling: 10,400 letters cuts Naibbe CER 40%; 20,800 fails on lexicon growth.**
Development only; round four unchanged; three texts; nested 5,200 / 10,400 / 20,800-letter prefixes.
Mean polished CER was 5.62%, then 3.39%, then 4.18%. The declared rule (20,800 at most half of 5,200)
is not met, so no sealed long round follows. The cause: absolute count thresholds admit frequent
concatenations as pieces. Spurious pieces grew from 32–75 to 295–408, and split tokens read as one
whole piece became the main error. Refinement's fixed 300 s cap was also hit at 20,800 (declared
confound). Missing true pieces fell from about 45 to about 20. Next: a length-aware lexicon
(thresholds and refine cap scaled with length) under the same rule. CPU, 80 min in three processes.
[Report](experiments/length-scaling/REPORT.md).

**2026-09-23: Voynich transcription suspects: near-hapax forms far above natural text.**
Training pages only, no decoding. In v101, 10.4% of tokens occur once and sit one edit from a
form seen at least 5 times; in EVA, 6.8%. Equal-size Latin, Italian, Old French and German samples
give 0.9–2.6%. So most such forms are the manuscript's own variation, not misreadings, and the
filter must not be used to correct the text. EVA substitution pairs mix visually close glyphs
(a/o, f/k, k/p, n/r, c/s) with word-ending alternations (o/y, s/y, d/y). v101 and EVA differ by
2 or more certain-space words on 447 of 2,878 lines. The output is a 164-candidate list, with
the top 50 given as folio loci for human review against the scans. Also defined: an exclusion
set for sensitivity checks of any later Voynich scoring. [Report](experiments/voynich-suspects/REPORT.md).

**2026-09-23: language-ID control: the true language wins in 5 of 5 Naibbe ciphertexts.**
The pipeline must not assume Italian for the Voynich text, so this was declared before
encryption. Five languages (medieval Latin ITTB, Old French, German, English, Italian with
Compagni) were each encrypted once with Naibbe. Each was decoded under five equal-budget
order-5 priors (606,976 letters) with round four's shared stages: 25 decodes, no caps hit,
no download. Score: bits per letter minus the prior's held-out entropy. The true language
ranked first 5 of 5, with a median margin of 1.28 bits per letter (range 0.75 Italian to 1.33).
The true-prior decode sits 0.23–0.48 bits above natural text; wrong priors sit 1.2–2.5 above.
This licenses comparing candidate languages this way for Naibbe-class text. It does not
establish that Voynich is Naibbe-class or which language it is. A Voynich run remains gated.
[Report](experiments/language-id/REPORT.md).

**2026-09-23: round five, v3 segmenter in Naibbe recovery: WER 45.8% to 41.5%.**
Declared and frozen before any passage existed. Eight sealed ciphertext-only cases: four
fresh Dante and four unused Compagni passages, each with a new key and Naibbe seed. The
round-four stages were shared. Arms: A round four; B A's letters with v3 segmentation; C
v3 in polish and segmentation. Pooled: A 5.63% CER / 45.81% WER, B 5.63% / 41.46%, C
5.81% / 41.69%. The primary C-vs-A endpoint passed (−4.12 WER points, +0.18 CER). The
gain is all segmentation. B improves all 8 cases, and the v3 polish cost worsens CER in 5 of 8.
Adopt v3 for segmentation only. Round four's letter error reproduces on new text. No
case passes the 1%/10% gate, and no cap was hit. CPU, 71 min. Remaining word error comes
mostly from letter errors; the candidate lexicon is still the limit.
[Report](experiments/joint-recovery-v5/REPORT.md).

**2026-09-23: v3 segmenter passes fresh transfer on a new author; word gate not met.**
With the user's approval, Dino Compagni's *Cronica* (Wikisource) and UD Italian ParTUT
were pinned after the method freeze and before any passage existed (`bf5dc9a`). ParTUT
test was too short, so dev was appended; this deviation is recorded. Four passages per
source, perfect letters, paired with the round-four baseline, predictions saved before
grading. Compagni WER fell 24.01% to 17.46% (−6.55); ParTUT 7.23% to 6.05%. All eight
passages improved. The 3-point transfer threshold passed. No Compagni passage reaches
10% WER, so the word gate is not met. Post-grading attribution: 344 of 391 remaining
extra spaces are inside missing forms. Recurring archaic forms dominate (`giano` ×27,
`erono` ×15, `-orono` verbs); the spelling model is mostly modern Morph-it types. This
licenses a separately declared paired Naibbe comparison with only the segmenter changed.
Compagni and ParTUT are now released. CPU, about 40 s; about 1 MB downloaded.
[Report](experiments/word-segmentation-v3-fresh/REPORT.md).

**2026-09-23: letter-level unknown-word model passes development; frozen.**
Declared four candidates and committed them before scoring (`5d58648`). Each replaces
the flat unknown cost with $-\log p_{unk} - \log P_{spell}(w)$. $P_{spell}$ is a
Witten–Bell letter n-gram (order 3 or 5) on lexicon types. $p_{unk} = 2.34\%$ is a
training-only Good–Turing rate. An optional rule joins elided prefixes. The selected
candidate, order 5 with elision, takes perfect-letter WER from 15.27% to 8.65% on
historical prose, 28.49% to 20.68% on Petrarca and 6.89% to 5.61% on modern ISDT. All
four candidates were eligible. Prose extra spaces fall from 92 to 39. These are
development numbers on the diagnosed streams, not transfer. Frozen before any fresh
source is named. Next: the protocol's fresh test on a new historical author and a
new modern corpus, which needs the user's choice and a download. CPU, 24 s.
[Report](experiments/word-segmentation-v3/REPORT.md).

**2026-09-23: extra spaces come from missing word forms; rubric-free extractor ready.**
A new v3 extractor drops Wikisource chapter rubrics stored as ordinary paragraphs.
On the released Villani source it removes exactly the 36 audited rubric paragraphs
and leaves all 37 body paragraphs unchanged. A diagnosis of the saved v2 baseline
development predictions (perfect letters) attributes each wrong space. 90% of
historical extra spaces (83/92), 87% modern and 83% verse fall inside words absent
from the 407,341-form lexicon; 57 of 64 missing historical forms are split. Released
Villani is similar (670/762), descriptive only. A labelled oracle that only adds each
stream's missing forms cuts WER from 15.27% to 2.29% historical, 6.89% to 2.76% modern,
28.49% to 6.89% verse. Cause: the flat unknown-word cost $15+3\ell$ nats is always
beaten by a split into known pieces (`melano` 33.0 vs `me la no` 19.4). Elided forms
merged by `normalize` are a minority (6/64 historical). No setting chosen, no fresh
text, no downloads, CPU under ten seconds. Next: declare a letter-level unknown-word
model on training text, then freeze before any new historical author or modern corpus.
[Diagnosis](experiments/word-segmentation-v3/REPORT.md).

**2026-09-23: verse word model passes development, fails fresh transfer threshold.**
Added only designated Petrarca training poems to the frozen prose segmenter's word
counts, transitions and vocabulary. Tried weights 1, 4 and 16, with unchanged scoring
and search. Weight 1 won the declared development selection: historical prose WER
15.27% to 14.92%, Petrarca 28.49% to 14.04%, modern ISDT unchanged at 6.89%.
Committed the method, extraction and grading code in `5c86127` before fetching a new
historical author and modern corpus. Both methods then segmented identical perfect
letters from four Villani Book I passages and four UD Italian VIT test passages.
Predictions were saved before reference spaces were opened. Pooled Villani WER was
28.37% to 27.35%; modern 8.49% to 8.38%. The 1.02-point historical gain misses the
declared 3-point transfer threshold. No Villani passage meets the 10% word gate;
two of four modern passages do. Candidate not promoted; round four stays the baseline.
Post-grading audit found chapter rubrics encoded as ordinary paragraphs: seven
included rubrics contribute 107/4,783 historical words (2.24%). This deviates from
the planned prose-only extraction; treat the historical evaluation as a paired
diagnostic with that caveat, not a clean confirmatory prose test. Frozen outputs
are preserved; no favourable subset or corrected challenge was substituted. The
size-based packer also skipped 30 short historical residual paragraphs, with IDs
recorded. Raw sources, licences, full graded records and graphs are committed;
VIT is CC BY-NC-SA 3.0. All 120 tests passed. CPU only, no paid compute, no Voynich
text or new Naibbe case. Next: fix rubric extraction in a new version and diagnose
missing forms versus incorrect splitting on existing development data before
another frozen candidate. [Report](experiments/word-segmentation-v2/REPORT.md).

**2026-09-23: segmentation audit separates two bottlenecks; fixed-key reparse rejected.**
On the three existing 5,200-letter development streams, the frozen word segmenter,
given perfect letters, has WER **15.3% historical prose, 6.9% modern, 28.5% Petrarca**.
With the true cipher-piece inventory supplied as an explicitly labelled oracle,
the polished decoder reaches CER **0.38%, 0.42%, 0.65%**, but WER is still **17.2%,
10.03%, 29.7%**. Better character recovery alone does not pass the historical word gate.
One declared candidate searches known whole/split readings with a 128-state beam
under the fixed round-four key, full five-gram context, length and homophone costs.
Paired mean CER improves only **4.97% to 4.67%**, WER **37.3% to 36.0%**; historical
CER worsens slightly. It fails the predeclared one-percentage-point improvement
requirement. No setting sweep, no fresh passages consumed, no new modern corpus or
historical author evaluated. All reported stage caps are false. This is a development
negative, not a new sealed round or a general cipher impossibility result. Round four
remains the baseline; the next separately frozen development comparison should fix
historical word boundaries before spending fresh passages. No Voynich text, paid
compute or model downloads. [Audit and graphs](experiments/segmentation-audit/REPORT.md).

**2026-09-22: object-and-relation image pilot.** Directly inspected 24 varied panels,
recording objects, count bounds, evidence rectangles and relations. The deterministic
`object_relation_v1_b9cbf0c3a6dc` analysis derives multi-label domains and compound patterns:
two panels with figures in linked basins, five with vessels beside botanical groups,
two with an animal inside a circular diagram, and a tentative linked-circle graph on
the Rosettes. CSV/JSONL, provenance and an evidence gallery are archived. This is one
AI observer's unmasked development annotation, not an automatic semantic detector,
independent ground truth or text association result. Two human reviewer templates and
an agreement scorer are implemented; reviewed text masks and the human reviews remain
pending. The 228-panel pixel baseline is unchanged. CPU only; no paid service or model
download. [Pilot report](data/folios/object-pilot/REPORT.md).

**2026-09-22: history rewritten once to remove a leaked token.** An early local commit had placed
a Hugging Face access token in `config.py`; the next commit removed it, but GitHub's push protection
rejected every push. The token was revoked and the one blob replaced by a placeholder in a single
`filter-branch` pass before anything was pushed. Commits after it changed hash; the reports and this
log now cite the new hashes, and [docs/COMMIT_MAP.md](docs/COMMIT_MAP.md) maps old to new. No
tree other than that one blob changed; all `verify` commands and tests pass unchanged.

**2026-09-21: codebook-free Naibbe recovery reaches partial success in development.** The
standard-method report had left the Naibbe gate untested: its one-letter model class cannot
express Naibbe's one/two-letter units. New code in `voynich/variable_units.py` and
`voynich/joint_segments.py` adds a variable-length key beam with a description-length score,
iterated local search, annealing, and a trigram HMM whose token parses are latent. Findings on
development text (Novellino/Decameron dev tales, ISDT dev), all CPU:

- With the true segmentation supplied as a diagnostic, the true key scores fewer bits than every
  beam or annealing result, yet only EM finds it: 3.2% CER at 5,200 letters after refinement.
  At 1,300 letters EM fails with 24 restarts. The earlier 1,200–1,800-letter benchmarks were
  below the practical unicity distance of this class; their Naibbe negatives were partly a length effect.
- Without the codebook, joint segmentation-and-decipherment EM plus refinement gives about
  12% CER (modern, 5,200 letters) and 15% (historical, 10,400 letters), from 300%+ before.
  Segmentation is 85–89% right and dominates the residual error. Fixed deterministic
  segmentation collapses EM at every length; re-segmentation under a fixed key and warm EM
  restarts did not help. The artificial variable-homophonic control stays unsolved and is out of class.
- **Round four, frozen at `b4648f6` and run on four new sealed Dante passages** (5,213–5,263
  letters, 22 CPU minutes): lexicon repair between the second joint EM and the refinement. An
  oracle with the true piece lexicon had shown the same EM at 97% agreement and 0.5% CER on
  development text; the repair drops whole pieces whose count matches a prefix+suffix bigram
  under the decoded key and admits complements of known halves for unparsed tokens. CER 4.4%,
  5.2%, 6.3%, 7.0%; WER 38–50%. Pooled: **5.7% Dante**, from 10.5%; WER 45% from 56%. Parse
  agreement 90% → 93% on every case. No modern passages: the ISDT test split holds fewer than
  1,000 fresh letters. Stop rule (Dante below 8%) passed; gate still fails.
  [Round-four report](experiments/joint-recovery-v4/REPORT.md), [development](experiments/joint-development-v4/REPORT.md).
- **Round three, frozen at `2221e83` and run on four new sealed passages** (5,211–5,469 letters,
  20 CPU minutes): the round-two decoder followed by a lexical polish that re-scores shortlisted
  letters against the frozen segmenter's word cost. CER 8.2% and 9.3% on modern ISDT, 10.1% and
  10.8% on Dante; WER 54–58%. Pooled: **8.8% modern, 10.5% Dante**. Paired against the same
  decoder before the polish (9.3% / 11.5%), the polish gains 0.5–1.5 points on every case.
  Letter error inside correctly parsed tokens is 0.5–1.4%; mis-parsed tokens hold 10–14% of the
  letters and nearly all the remaining error. Gate still fails; the residual is segmentation, not
  the key. [Round-three report](experiments/joint-recovery-v3/REPORT.md),
  [development](experiments/joint-development-v3/REPORT.md).
- **Round two, frozen at `699ab78` and run on four new sealed passages** (5,237–5,286 letters,
  10 CPU minutes): usage pruning of the candidate piece lexicon and a prior that adds Petrarca's
  Canzoniere (pinned from Wikisource; Dante never used for fitting). CER 10.7% and 8.3% on modern
  ISDT, 9.0% and 11.5% on Dante; WER 49–63%. Pooled: **9.5% modern, 10.3% Dante**, from 12.5% and
  33.5%. Gate still fails. The verse prior transferred across authors; segmentation agreement is
  88–91%. [Round-two report](experiments/joint-recovery-v2/REPORT.md),
  [development](experiments/joint-development-v2/REPORT.md), [verse sources](experiments/verse-prior/sources.json).
- **Round one under the frozen protocol** (commit `983542c`, four sealed passages of
  5,201–5,529 letters, 14.5 CPU minutes): CER 9.5% and 15.5% on modern ISDT, 30.1% and 36.7% on
  Dante; WER 49–88%. Gate CER ≤ 1% / WER ≤ 10% not met on any case. Modern matches development;
  Dante is worse than development prose, which points at the prose-trained prior rather than
  segmentation (agreement 85–91% on all cases). [Fresh report](experiments/joint-recovery/REPORT.md),
  [development record](experiments/joint-development/REPORT.md). Voynich text untouched; final test sealed.

**2026-09-21: image descriptions reopened by explicit request.** Downloaded the complete
213-image Yale scan set at 1,800 pixels on the longest side (121.2 MB), indexed all
228 Voynich.nu panels, and reviewed 39 registered foldout crops. Inspected all scans
in contact sheets and enlarged examples spanning figures, containers, diagrams and
text. The [archive and method](data/folios/README.md) contain sources, hashes, reviewed
crop coordinates and a reproducible CPU pipeline. Pixel Layout v1 produces CSV/JSONL
rows keyed by `(analysis_id, folio_id)`, plus a searchable gallery. It describes colour,
patch distribution, geometry and dark-mark bands; subject domains remain unassigned.
The [visual review](data/folios/VISUAL_REVIEW.md) records direct observations and the
false-circle detector failure found during development. This is an exploratory
description baseline, not a fourth association study or evidence of meaning. All
images were visible during development; no independent masked annotation is claimed.
All 92 repository tests passed. A full rerun reproduced the archived files exactly;
the 231 generated output checksums also matched staged Git contents. No paid compute.


**Standard-method comparison completed.** Method committed as `752c3fa` before generating
16 fresh encodings of four passages. CPU decoding took 13.3 minutes; no paid compute.
The [report and graphs](experiments/standard-decipherment/REPORT.md) and
[methods note](experiments/method-benchmark/METHODS_NOTE.md) are the latest records.

- MDL reranking repairs both historical substitution cases: 145.82% → 0% character error.
- Published beam candidates give exact selected letters in 7/8 substitution/homophonic
  controls; the remaining case has 0.63% error. All eight pass the letter threshold.
- Historical segmentation improves 37.67% → 24.20% on the same new passages; modern
  segmentation improves 7.38% → 6.02%. The historical <10% gate still fails.
- Variable-length controls and Naibbe fail. Naibbe selected character error remains about
  304–319%; even the oracle best generated candidate has roughly 70–71% error.
- The HMM comparator completed eight restarts on each eligible case; it is a bounded
  adaptation, not a reproduction of the million-restart paper.
- Five cases retain an oracle-selection gap. One is a small same-length homophonic error:
  7,173 bits for eight wrong letters versus 7,185 for the exact 1,265-letter candidate.
  MDL removes the expansion defect but does not guarantee the true reading.
- The exact evaluated records are committed after grading, including seeds and references.
  Those source IDs are now disclosed; exclude them from all future fresh evaluations.

The Naibbe gate is closed. The standard one-letter model class does not generally express
Naibbe's units; its failure is not a universal cipher rejection. No Voynich-facing run,
new image study, or final-test scoring occurred. All 83 unit tests and freeze audits passed.


**2026-09-21 priority change.** Archived work is committed as `49a4522`.
The [standard-method protocol](experiments/standard-decipherment/PROTOCOL.md) now governs
CPU development: total description length, published homophonic comparators, and a
non-Dante historical lexicon. A separate commit must freeze the method before fresh grading.

**Image association tests parked.** Image acquisition/description is now reopened above.
All three catalogue-based studies are preserved. No fourth association study
on Grove/Stolfi will run. Reopening requires text-masked annotation, two annotators,
agreement statistics, and a design that can separate image domain from hand and quire.

**Voynich mechanism reserved.** The intact-versus-within-line-shuffle comparison stays
closed until codebook-free Naibbe meets the frozen letter and word recovery gates.
A standard one-letter homophonic model cannot express every Naibbe unit. Failure of that
model alone cannot rule out an unknown variable-length cipher or meaning in Voynich.


**Prediction track formally closed, 2026-09-21.** No additional BPC sweeps, longer runs,
or larger-model comparisons. A future exception must name a mechanism, falsifiable
contrast, fixed budget, and the interpretation it could reject. Lower BPC alone is
insufficient. Existing results, failures, and unused diagnostic options remain archived.

The [bounded CPU plan](experiments/METHOD_BENCHMARK_PLAN.md) supersedes earlier next-step
suggestions. Its one-week budget is a cap, not a reason to run machines unnecessarily.
No new GPU rental, model download, or Voynich final-test scoring is authorized by it.

| Work | Status and measured result | Record |
|---|---|---|
| Cloud cleanup | Deleted stopped pod and attached temporary volumes; zero network volumes; $8.89 balance at check | [Audit](experiments/cloud-cleanup.json) |
| Lexicon segmentation | Frozen after non-Dante tuning; 24 fresh passages. Modern word error 15.9% → 6.1%; historical 60.9% → 39.9%. Historical gate failed | [Report and graphs](experiments/segmentation/REPORT.md) |
| Codebook-free recovery | Completed: modern substitution passes; historical candidate selection, broader control, and Naibbe fail | [Protocol and results](experiments/codebook-free/REPORT.md) |
| Joint segmentation + EM, round one | Fresh 5,200-letter Naibbe: CER 12.5% modern, 33.5% Dante (from 300%+); WER 58% / 87%; gate not met | [Report](experiments/joint-recovery/REPORT.md), [development](experiments/joint-development/REPORT.md) |
| Round two: usage pruning + verse prior | Fresh 5,200-letter Naibbe: CER 9.5% modern, 10.3% Dante; WER 54% / 60%; gate not met | [Report](experiments/joint-recovery-v2/REPORT.md), [development](experiments/joint-development-v2/REPORT.md) |
| Round three: lexical polish | Fresh 5,200-letter Naibbe: CER 8.8% modern, 10.5% Dante; WER 54% / 56%; polish gains 0.5–1.5 points paired; error inside correct parses 0.5–1.4%; gate not met | [Report](experiments/joint-recovery-v3/REPORT.md), [development](experiments/joint-development-v3/REPORT.md) |
| Round four: lexicon repair | Fresh 5,200-letter Naibbe, Dante only: CER 5.7%, WER 45%; agreement 90% → 93%; oracle with true lexicon 0.5%; gate not met | [Report](experiments/joint-recovery-v4/REPORT.md), [development](experiments/joint-development-v4/REPORT.md) |
| Independent evidence pilot | 59 visual descriptions across six folios. Text gain +1.45 accuracy points, p=0.348; no established association | [Report and graphs](experiments/association/REPORT.md) |
| Complex image associations | Completed: 123 descriptions, three text kernels, four endpoints; no reliable gain in 12 corrected tests. Nonlinear synthetic control passes | [Report and graphs](experiments/association-complex/REPORT.md) |
| Broad image domains | Complete: 63 folio groups / three domains. Character text 72.2% balanced accuracy; hand/layout 94.4%; no incremental text gain | [Domain report](experiments/image-domains/REPORT.md) |
| Deliverable | Auditable recovery benchmark and limitations; no Voynich translation claim | [Combined report](experiments/method-benchmark/REPORT.md) |

**Segmentation finding:** the relative word-error reduction exceeds the declared 20%
target on each corpus (61.4% modern, 34.5% historical). Only modern Italian meets the
<10% word-error gate. Correct boundary characters do not imply recovered historical words.
No tuning followed exposure of these evaluation answers. Fresh Dante passages still
share an author/work with the earlier benchmark; this is not cross-author validation.

**Codebook-free finding:** all 12 cases from four further fresh passages completed in
83.5 seconds on CPU. The two modern substitution cases pass the recovery gate (0% letter
error, 5.72% pooled word error). Both historical substitution cases contained an exact
letter candidate, but the selection score chose a wrong expansion. Naibbe and the
artificial variable-length positive control fail even under post-hoc best-candidate
checks. This decoder is not validated for broader cipher recovery. That is a method
negative, not proof that every possible unknown-cipher solver or Voynich interpretation
must fail. No further tuning was performed on these now-exposed references.

**Image finding:** existing Grove/Stolfi visual descriptions provide a narrow source
outside text statistics, but annotators were not blinded to text. Root-versus-plant
labels are confounded with folio; leaf/flower coloration has too few examples. The
eligible root-color endpoint does not pass its permutation/interval gates. The larger
herbal-page study requires new independent visual annotation and unseen evaluation.

## Complex image-association extension

The user requested relationships beyond simple visual attributes. We froze an exploratory
extension with 35 botanical descriptors plus pairwise co-mentions, three text models
(additive, interaction, radial), and four outcomes: root color, joint-profile prediction,
within-page description matching, and relationships between objects. The source yields
123 clear whole-plant descriptions on six training folios; 121 support same-page matching.
Depending on the held-out folio, 58–70 joint-profile dimensions have training support.

**Finding:** no reliable improvement in any of 12 predeclared comparisons. The largest
apparent relational gain is +0.0719 correlation (raw p=0.153; interval −0.0821 to +0.2466).
All Holm-adjusted p-values are 1.0. Same-page matching rank is 52.28–53.85% with text
versus 54.65% for nonlinear controls. All three text models worsen joint-profile error.
The nonlinear machinery detects a planted interaction at 100% held-out accuracy versus
50% additive/control accuracy; this validates that check, not arbitrary manuscript patterns.

**Limit:** this reuses the earlier catalogue; it is exploratory, not new independent
confirmation. The targets are mentions in image descriptions, not raw pixels or known
biological absences. Shape and texture coverage is thin: one triangular-shape example,
two hairy/fuzzy examples, five striped examples. Independent masked-image annotation
and fresh folios are the next data requirement. No new BPC work or cloud computation.

Records: [protocol](experiments/association-complex/PROTOCOL.md),
[full report and graphs](experiments/association-complex/REPORT.md),
[all scores and null distributions](experiments/association-complex/results.json).

## Broad illustration-domain study

The earlier image studies were restricted to plants/roots. The user requested broader
image domains. The new frozen study uses the transcription's conventional illustration
categories and includes paragraph, label, and circular text. It audits 162 training
pages/panels, then compares 142 illustrated pages in 63 physical groups: 50 botanical,
7 people/bathing, and 6 celestial/diagram groups. Zodiac, astronomy, and cosmology remain
separate in the coverage audit but share a broad classification target. Stars in text
margins and text-only pages are not automatically labelled celestial. These are dominant
domains, not exhaustive object-presence tags; zodiac pictures also contain people.

**Finding:** held-out-folio character text scores 72.22% balanced accuracy, word text
57.14%, and hand/layout controls 94.44%. Adding either text view gives 88.89%, with no
supported incremental gain. Raw accuracy is misleading because 50/63 groups are botanical.
The strict hand/layout/quire/position baseline also scores 94.44%.

**Transfer limit:** the people/bathing examples all belong to quire M, so their cross-quire
transfer cannot be assessed. Botanical/celestial cross-quire scores are 83.33% for controls,
83.33% after adding character text, and 58.33% after adding words. Hand-conditioned labels
cannot move at all in this binary comparison; that conditional test is unidentifiable.
Conditioning on both hand and quire also leaves no domain exchanges in the full sample.
This does not establish semantic independence, individual word meanings, or absence of
an image–text relationship. It demonstrates why section recognition is an insufficient
meaning benchmark.

Records: [protocol](experiments/image-domains/PROTOCOL.md),
[coverage, controlled comparisons, and graphs](experiments/image-domains/REPORT.md),
[all scores and per-folio predictions](experiments/image-domains/results.json).
A stronger next comparison would link text regions to independently annotated people,
stars, containers, plant parts, or pipes/pools within matched manuscript contexts.
No new BPC study, paid computation, or final-test scoring occurred.

## Archived matched-context completion

**Completed: matched-context training, all 18 local runs verified.** Each model trained
and evaluated with 8, 32, or 128 preceding characters. We compared intact and within-line shuffled
Voynich, with three training seeds. Model size, initial weights within a seed, optimizer,
updates, and scored-character exposure were matched. Primary results use the fixed final
epoch, not whichever checkpoint looks best on validation.

- Question: does extra history help intact text more than its shuffled control after
  the model learns to operate at that history length?
- Why: the previous memory test changed history only at evaluation; the Naibbe curve
  showed that shortening history can disrupt a model's learned operating conditions.
- Budget: local Apple GPU only; no model downloads or new cloud rentals. Twenty passes
  per run; two-hour run cap and twelve-hour suite cap. Preserve 20 GiB free disk.
- State and results: [fixed protocol](experiments/MATCHED_PLAN.md),
  [completed-run table](experiments/MATCHED.md),
  [machine-readable results](experiments/matched-results.json), and
  [illustrated final report](experiments/report-matched/REPORT.md).
- Runtime details: `artifacts/matched/status.json`, `artifacts/matched.log`, and
  `training_run_outputs/matched-*`. Failed or incomplete runs stay labeled as such.

This comparison tested whether extra history benefits intact text more than its shuffled control. It cannot produce a word
translation or prove the manuscript contains language. Completing every possible
linguistic-statistics experiment is not a prerequisite for the known-plaintext benchmark.

Early diagnostic: the first intact 8-character run reached 2.4689 validation BPC at
epoch 6 but finished at 2.6542 at epoch 20. The training loss kept falling. This is
evidence of overfitting in that run, not evidence against meaning. We retain the fixed
final-epoch comparison and show the full curves and secondary best-epoch scores.

Completion: **18/18 runs verified**. Each model completed 20 passes, 9,820 updates,
and 2,511,320 scored training targets. Checkpoint reload errors were zero. Frozen input
hashes, saved scores, and equal exposure were checked; the suite released its training lock.
The suite took 4.92 hours, produced about 276 MiB of model/run outputs, and finished
within its original deadline with 56.8 GiB free. No new cloud compute or models were used.

| Seed | Intact gain (BPC) | Shuffled gain (BPC) | Extra intact gain | 95% source-folio interval |
|---|---:|---:|---:|---|
| 42 | 0.1195 | 0.1544 | -0.0350 | [-0.0501, -0.0212] |
| 43 | 0.1157 | 0.1192 | -0.0035 | [-0.0256, +0.0199] |
| 44 | 0.1330 | 0.1362 | -0.0033 | [-0.0327, +0.0258] |

**Finding:** longer history helped both versions, without an established extra benefit
for intact Voynich. Mean gains were 0.1227 BPC intact and 0.1366 shuffled. One seed
favored shuffled text; two were inconclusive. The mean interaction was −0.0139 BPC.

**Limit:** all 18 models peaked on validation between passes 5 and 9 and deteriorated
by pass 20. Equal exposure does not equalize convergence or overfitting. These results
do not measure semantic content or imply that the manuscript is meaningless. The folio
intervals do not include every source of training/design uncertainty. Final-test pages remain sealed.

The prediction suite is finished and its monitor is paused. The later CPU benchmark
work is recorded above; the suite is not a reason to restart prediction training.

A reporting error stopped the suite after run five: bootstrap and training seeds used
the same dictionary field. The report now records both separately. A regression test
passes; the original runner and manifest are archived in `artifacts/matched/repairs/`.
A recorded manifest amendment permits this reporting/resume repair. The five completed
models were verified and retained; training resumed under the original suite deadline,
with unchanged training code, data, settings, and exposure.

**Completed alongside training: language statistics and blind decipherment.** Both used
local CPU work. The statistics compare 148 Voynich training pages against eleven pinned
historical/modern corpus samples. The recovery benchmark tests six hidden Dante passages
under two random keys and three supplied cipher conditions. Neither uses Voynich test pages.
See the [statistics report](experiments/language-comparison/REPORT.md) and
[decipherment report](experiments/decipherment/REPORT.md), with five figures in total.

## What we tried and found

Voynich model scores below are validation results. New corpus statistics use training
text; controlled decipherment has separate hidden original passages. Different texts, transcriptions, and evaluation
window settings are not interchangeable. Follow the linked reports for exact configurations,
page scores, uncertainty, and limitations.

| Experiment | Result | What it tells us | Record |
|---|---|---|---|
| Repair the original pipeline | Fixed data scrambling, optimizer-step limits, target masks, grouped splits, and saved-checkpoint verification | Later model comparisons use auditable data and scoring | [Implementation plan](RESEARCH_PLAN.md), [tests](tests/) |
| Frequency, spelling, layout, and copying | GC frequency 4.2450 BPC; best copy baseline 2.6985; copy character accuracy 44.93% | Local regularity explains substantial predictability | [Initial results](experiments/RESULTS.md) |
| Boundaries, transcription, and quire holdout | Copy: merged GC 2.6499; separated GC 2.6441; quire split 2.7815; ZL 2.0691 | Representation and split matter; these different target strings cannot be directly ranked | [Baseline matrix](experiments/RESULTS.md) |
| Short Qwen layer pilots, 400 updates | Frozen 4.3505; outer 2.9373; middle 3.1549; random 2.8783 | Adaptation helped, but all pilots lost to copying; outer layers were not uniquely effective | [Pilot settings and scores](experiments/RESULTS.md) |
| Longer Qwen training, 3,000 updates | Outer 2.4819; random 2.4828 | More training beat copying; no clear outer-versus-random layer advantage | [Learning curves](experiments/LEARNING_CURVES.md) |
| Five Qwen training seeds | Mean 2.4833 BPC; sample SD 0.0017 | The selected configuration's prediction gain was stable across these seeds | [Replications](experiments/REPLICATIONS.md) |
| Qwen on shuffled and synthetic controls | Shuffled 2.6202 vs copy 2.8644; Timm 2.0170 vs copy 2.0909; Naibbe 1.8061 vs layout 1.8683 | Beating simple baselines also happens on controls; it is not a meaning detector | [Control results](experiments/REPLICATIONS.md) |
| Cloud Qwen 1.7B-Base versus 8B-Base | 2.4771 vs 2.4689; gain 0.0083, folio interval [-0.0016, 0.0175] | This one-seed comparison did not establish a reliable larger-model gain; the gated cloud follow-up was not run | [Cloud results](experiments/CLOUD_RESULTS.md) |
| Small character models trained from scratch | Three-seed GRU mean 2.3321; character transformer mean 2.5656 | A small model can outperform these Qwen runs on prediction; training exposure and architecture differ, so this does not isolate pretraining's effect | [Twelve character runs](experiments/CHARACTERS.md) |
| Character models on controls | GRU: shuffled 2.5159, Timm 2.0807, Naibbe 1.7366; Qwen remains better on Timm | The model ranking varies by control | [Character control results](experiments/CHARACTERS.md) |
| Saved-GRU memory test, 30 evaluations | Intact mean: 2.5482 BPC at 8 characters, 2.34898 at 64, 2.32857 at 128; 64 captures 90.7% of the measured 8-to-128 gain | Extra history helps; shuffled text benefits almost as much. Naibbe worsens at intermediate histories, exposing a training-mismatch question | [Context report and graphs](experiments/CONTEXT.md) |
| Learned symbol groups | Training-only BPE and tiny-model plumbing checks passed | Infrastructure exists; no pretrained-tokenizer replacement or semantic result was claimed | [Tokenizer usage](README.md#learned-symbol-groups) |
| Matched-context retraining | All 18 runs verified; mean 8-to-128 gain 0.1227 intact vs 0.1366 shuffled | No established extra intact gain; all runs overfit by the fixed final epoch | [Final report](experiments/report-matched/REPORT.md) |
| Historical/modern corpus statistics | v101: 25,723 forms, mean length 3.865, adjacent-length correlation +0.218; EVA: mean 4.975, correlation +0.169 | Length clustering survives transcription and boundary checks; genre/layout confounds prevent language identification | [Statistics and graphs](experiments/language-comparison/REPORT.md) |
| Blind substitution recovery | 100% normalized letters and words with spaces; 100% letters without spaces | Hidden-key recovery works with known Italian and a supplied simple cipher family | [Recovery benchmark](experiments/decipherment/REPORT.md) |
| Restricted Naibbe recovery | 99.44% letters with structural codebook supplied | A positive control under strong declared assistance, not unknown-cipher discovery | [Recovery benchmark](experiments/decipherment/REPORT.md) |
| Word-boundary failure and secondary repair | Primary space-free word error 99.32%; independently calibrated penalty reduces it to 60.33% for substitution and 61.53% for Naibbe | Correct letters do not guarantee correct words; secondary reuse is exploratory | [Boundary diagnostic](experiments/decipherment/boundary-diagnostic.json) |
| Frozen lexicon segmentation | Fresh modern WER 6.15%; historical 39.90%; both improve >20% relatively, historical gate fails | Correct space insertion remains a separate, unresolved historical-language problem | [Fresh benchmark](experiments/segmentation/REPORT.md) |
| Codebook-free recovery | Two modern substitution cases pass; historical selection, variable-length control, and Naibbe fail | Prior language score can prefer a wrong expansion even when exact letters are available | [Negative result](experiments/codebook-free/REPORT.md) |
| Joint segmentation + EM on Naibbe | Development: oracle segmentation 3.2% CER at 5,200 letters, 77% at 1,300. Fresh round one: 12.5% modern, 33.5% Dante. Round two with pruning and a Petrarca-augmented prior: 9.5% / 10.3%. Round three with a lexical polish: 8.8% / 10.5%. Round four with lexicon repair: 5.7% Dante; gate failed | Search, not the objective, blocked earlier attempts; length below ~2,600 letters is unrecoverable; the prose prior, not segmentation, limited Dante; the candidate lexicon, not the model, limits parsing (oracle 97% / 0.5%) | [Round four](experiments/joint-recovery-v4/REPORT.md), [round three](experiments/joint-recovery-v3/REPORT.md), [round two](experiments/joint-recovery-v2/REPORT.md), [round one](experiments/joint-recovery/REPORT.md) |
| Text–image pilot | 59 labels / six folios; +1.45 balanced-accuracy points; p=0.348 | Existing root-color descriptions do not establish a text association; independent annotation is still needed | [Pilot report](experiments/association/REPORT.md) |
| Complex text–image associations | Three kernels × four endpoints on 123 descriptions; no corrected signal | Joint, nonlinear, and relational tests remain negative on this limited annotation source | [Extended report](experiments/association-complex/REPORT.md) |
| Broad illustration domains | Botanical / people / celestial: text-only character balanced accuracy 72.22%; hand/layout 94.44%; no incremental gain | Domain and manuscript production are strongly entangled; some conditional effects are unidentifiable | [Domain report](experiments/image-domains/REPORT.md) |

The frozen [completed-experiments report](experiments/report-completed/REPORT.md) and
[PDF](output/pdf/voynich-completed-experiments.pdf) cover the Qwen, cloud, and character
experiments. The [earlier report](experiments/report/REPORT.md) is an interim snapshot.
The exact-context test used stride 1; earlier GRU reports used stride 64. That change
explains why its 128-character reference differs from the earlier 2.3321 mean.

## What we know, and what remains open

1. **Predictive structure exists in these transcriptions.** Both simple copying and
   learned models predict better than character frequency. This does not identify
   a source language, cipher, word boundary, or meaning.
2. **Larger language models are not yet the bottleneck we have demonstrated.** The
   8B comparison gave an uncertain gain, while a small GRU predicted better under
   a different training budget. We have no evidence that another large model will translate Voynich.
3. **Our initial layer hypothesis is unconfirmed.** Outer and random layer adaptations
   were nearly equal after longer training. No activation intervention or SAE study
   has established a language-only subset of layers or a Voynich concept mapping.
4. **Controlled content recovery now works under declared assistance.** The spaced
   substitution benchmark recovered its normalized Italian originals exactly. Known
   language and cipher structure are substantial hints; word segmentation remains poor.
   Without the codebook, Naibbe is about 94% recoverable by letter on fresh 5,200-letter
   Dante passages (5.7% character error after four frozen rounds; 8.8% on modern Italian after
   three); below 2,600 letters this class is not recoverable by any method tried. Given the true
   piece lexicon the same solver reaches 0.5%, so the remaining error is in the candidate
   lexicon, not the model. Letters are not words: word error stays above 38%.
5. **Voynich meaning remains unvalidated.** No verified Voynich word, passage translation,
   or controlled image association has been produced. Corpus resemblance is not a language label.
6. **The same methods cannot name Linear A's language.** Lexical tests find Greek in at most 10%
   of Linear B control samples at Linear A's size; the one profile match (Hittite) also holds for
   shuffled syllables. [Summary](docs/LINEAR_A.md).

The validation set contains 29 pages from 15 folio groups, with an uneven A/B mix.
The GC representation is v101; the independent ZL transcription is EVA. Timm and Naibbe
are each one published sample, split into chronological blocks, not independent
generator replications. Baseline robustness was measured; the corresponding full
neural transcription/quire robustness matrix has not been run.

## Recovery evidence and remaining limits

**Known plaintext:** original text whose relationship to the ciphertext is independently verified.
The first controlled benchmark is complete: the solver used unrelated modern Italian
frequencies and received no paired original passages or true letter keys. The Naibbe
condition supplied its structural codebook. Predictions were frozen before grading.
The original boundary failure and the secondary calibration are both preserved.

```text
Improve boundary recovery using independent development material
Freeze the method and score fresh, unseen original passages
Check original content, not just letter accuracy or fluent output
Reduce supplied Naibbe structure and test ambiguity explicitly
Test which recoverable assumptions are defensible for Voynich
```

Do not tune further against the six disclosed Dante passages and call that blind progress.
A new recovery result needs fresh evaluation passages. A useful target is whole-word and
passage recovery with source spaces removed. English or Italian rendering must preserve
recovered content; it must not hide a wrong decoding. The new CPU protocols above test these ideas. No additional training or cloud job is
part of this benchmark. Further improvement needs a newly frozen method and fresh cases.

For Voynich, independent anchors could include repeatable text/image relationships
tested within section and hand, or mappings that predict unseen passages consistently.
Plant identities, medical uses, and source-language guesses remain hypotheses rather
than training labels known to be correct. We will use TransformerLens or SAEs only
when a concrete intervention can test a specific mapping or mechanism.

## Evidence, resources, and maintenance

- [README](README.md): setup and commands. [Research plan](RESEARCH_PLAN.md): methods and evidence requirements.
- `experiments/*-plan.json`: fixed machine-readable settings; `experiments/*-results.json`: numerical results.
- [Language sources](experiments/language-sources.json) and [Naibbe sources](experiments/decipherment-sources.json): immutable revisions, hashes, attribution, and licenses; about 210 MiB downloaded.
- `experiments/splits/`: fixed page groups. `experiments/sources.json`: pinned downloads and hashes.
- `artifacts/data/`: prepared text and metadata. `artifacts/results/`: baseline page scores.
- `training_run_outputs/`: model configs, training curves, weights, scores, and run status. These large files are git-ignored.
- [Literature and source links](RESEARCH_PLAN.md#key-reading): motivation, not independently reproduced findings.
- Runpod pod `6o8irlwqlzhsb1` and attached non-network volumes were deleted on
  2026-09-21 after the explicit user request. The audit log confirms deletion; network
  storage is 0 GB. Balance was $8.89 at check. Local results remain preserved.
  [Cleanup audit](experiments/cloud-cleanup.json). No new cloud run is planned.
- The matched-context suite is complete and its progress monitor is paused.
  [Final verification](experiments/report-matched/verification.json) records checkpoint, exposure, hash, and resource checks.

After each experiment, add its question, fixed settings, result, limitations, and
implication for meaning recovery here. Preserve old reports and failed attempts.
Distinguish planned, running, completed, and verified work. Do not turn a prediction
milestone into a translation claim, or keep expanding the statistics track indefinitely.

## 2026-09-24: Linear A correspondence source audit completed

**Question:** does the exploratory re/ru → ro association survive literal source readings
and nulls preserving real stems? Pattern name: conditional ending test.

**What was done**

- Traced all 12 historical pairs to pinned source inscriptions; retained six under uniform
  whole-corpus filters. Kept q/k and numbered signs distinct, rejected marked or broken runs,
  and required A word/sign/glyph agreement. HT 117a has te-*56-re versus te-ja-re; KH 41's
  ka-ta-re is excluded conservatively because gaps bound it. Valid qa-qa labels are restored.
- Froze code, tests, full inventories, accepted attestations, pair evidence, source hashes and
  protocol in commit `75128e8`, before real-data permutations. Eight new tests pass; all 24
  focused Linear A tests pass. Original rounds 1–3 and both earlier repair/structural freezes
  still verify. Old frozen code/results are unchanged.
- Strict inventory: 208 A types, 2,984 B types, length >=3. With 9,999 draws each, six matching
  stems exceed the length-null mean 1.5342 (p = .0011), but not the length+final-onset mean
  4.2191 (p = .1605). Both were required below .05/7, including Wilson upper bounds.
- Historical sensitivity: 12 stems versus 2.7487 (p = .0001) and 7.2026 (p = .0049);
  the old normalised data pass both. Full simulations and intervals are archived in
  [the report](experiments/linear-a-correspondence/REPORT.md).

**Why:** the prior p = .0001 under a syllable null neither checked the source readings nor
isolated the vowel change from an existing final-r association. The source-checked specific
adaptation lead fails the committed gate and is archived; simpler spelling overlap remains.

**Limits and next action:** the filters reduce coverage and power, and collated source layers
are not independent readings. This audit does not disprove individual loans or identify a
language. Preserve the six pairs as unresolved observations. Reopen only with new source
adjudication or independent dated/context-compatible evidence, not another split or ending
sweep on exposed words. No paid compute was used and no further statistical rerun is scheduled.

## 2026-09-24: accounting-program feasibility pilot completed

**Question:** can anonymous words choose arithmetic operations that predict quantities on
unseen physical tablets? Pattern name: the tablet as a program.

**What was done**

- Implemented commodity-specific integer vectors, sum-before, sum-after and balance-before
  programs, opaque word IDs, five whole-object folds, independent-object support, conflicting
  rule abstention and quantity-shuffled refitting controls. Target quantities are never read
  by the prediction expressions. Units/fractions/damage are barriers, not silently ignored.
- Archived source extraction, exclusion reasons, evaluator vocabulary and seven source-review
  candidates. Coverage was inspected before the freeze; no arithmetic fits or success scores
  were inspected. Frozen at `8793ad6`; the subsequent run returns not_evaluable.
- Found 7 eligible KN control objects (34 rows), 30 other-B development objects (150 rows),
  and 33 A objects (134 rows). The control cannot reach the minimum 10 objects or match the
  33-object target without replacement. No real-data fitting or null simulations were run.
- Passed all 34 focused Linear A tests, including ten new ledger tests. A 30-object synthetic
  case recovers its hidden total marker and predicts all held-out-object totals; this is a
  software check, not linguistic power evidence. Earlier repair, structural and correspondence
  freezes remain valid. [Report](experiments/linear-a-ledger/REPORT.md).

**Why:** the arithmetic route needs trustworthy quantities, boundaries and control cases before
it can support a function claim. This is a parser/coverage limitation, not a failed test of
arithmetic structure. The accounting approach remains open.

**Correction to the motivating example:** HT 13's old 130=130 result compared integer parts;
its source contains fractions in entries and the total. The old code explicitly documented
that limitation, but my earlier recommendation did not explain it. It is not a verification
of the full quantities. The old frozen result is preserved and the current summaries clarify it.

**Next:** build the [source-checked account benchmark](experiments/linear-a-ledger/BENCHMARK_PLAN.md):
separate uncertainty in names from quantities, cite unit conversions, justify account boundaries
and label functions independently. A to-so/to-sa token alone is not a verified sum equation.
Keep inspected objects in development; do not lower the gate or duplicate cases to force a pass.
No language claim, paid compute or further real-data scoring was made in this pilot.

## 2026-09-24: source-checked accounts and parentage feasibility

**What was done**

- Built and froze a nine-account/eight-object development benchmark at `f048a56`, with
  source-supported boundaries, function labels, commodity inheritance and exact units.
  Strict arithmetic: two balances, six uncertain/incomplete, one dimension negative.
  Conditional readings: five balances, three mismatches, one dimension negative. The
  PY Jn658 mismatch is independently discussed by the edition. [Report](experiments/linear-a-account-benchmark/REPORT.md).
- Built five parentage/pairing/collective-children controls on three Linear B objects;
  froze descriptive inventories at `1c3dd37`. “X and daughter” needs an unnamed participant,
  unlike a two-name CHILD_OF template. Concurrent Etruscan files were inspected read-only.
- Counted 48 graphic Linear A rows on 22 objects; the two pair-shaped rows are transaction/
  logogram ambiguities, not personal-name pairs. No kinship label was inferred. HT117a
  remains excluded for its explicit *56/ja conflict. [Report](experiments/linear-a-kinship/REPORT.md).
- Passed 44 focused tests, including seven exact-account and three ordered-inventory tests.

**Why:** source evidence must distinguish damaged names from damaged quantities, and personal
names from ordinary graphic slots. These development fixtures support further work without
turning conditional arithmetic or repeated names into translations.

**Limits:** the accounting control still lacks 33 matched objects and ten function-positive
objects. No real-data word-function learner was fitted. The kinship inventory is descriptive,
not a recovery gate; the Etruscan prototype supplies name spans and morphology unavailable
for these Linear A candidates. No paid compute or edits to the concurrent Etruscan branch.

## 2026-09-24: HT85/117 person-slot source audit

**What was done**

- Read the full Davis–Valério chapter and inspected GORILA photos, facsimiles and
  transcriptions for both target tablets, plus contextual HT87/88/122a transcriptions.
- Froze the manual inventory and descriptive code at `74610d3`: 43 logical rows on four
  faces of two objects; 32 counted words, three counted logograms and eight other rows.
  HT117 contributes 17 possible individual/designation entries, without gold name labels.
- Recorded an edition-supported preference for `te-ja-re`, preserving SigLA's conflicting
  `te-*56-re` and all earlier frozen inputs. Reviewed four structural leads against
  occupation, ownership/responsibility, place and kinship. No parentage formula identified.
- Passed 49 focused tests and verified all seven follow-up freezes. Literal concordance
  review found a catalogue alias: PH31a and PH(?)31a share HM1609. Raw source-key counts
  must not become independent-object support. [Report](experiments/linear-a-person-slots/REPORT.md).

**Why:** numeral 1 does not certify a name: qa-A310-i has count 1 on HT85b and 3 on HT122a.
Source roles and repeated contexts must be checked before a parent-child model receives labels.

**Next discriminating check:** qi-tu-ne is a reviewed header on HT87/117 but a counted entry
in the pinned HT7b text; image collation of HT7b could establish that role contrast. The
inventory still needs independent epigraphic review. No gender, translation, statistical
significance or recovery claim follows, and no previous failed gate was reopened.

## 2026-09-24: qi-tu-ne heading/count contrast confirmed

**What was done**

- Collated HT7a/b against GORILA I pp.14–15 photographs, drawings, transcriptions and
  apparatus. Confirmed qi-tu-ne 1 on HT7b .1; the front's qe-ti VIR supplies personnel
  context but does not automatically establish a unit for the reverse.
- Rechecked HT87/117b headings. Froze an eight-row HT7 inventory, three occurrence cases
  and explicit rival assignments at `9d47059`, then generated descriptive summaries.
- Confirmed AB21f–AB69–AB24 across three objects: one count-associated occurrence and
  two headings. Preserved overwritten-text qualifications and physical-object grouping.
- Passed 49 focused tests and verified eight freezes. [Report](experiments/linear-a-qi-tu-ne/REPORT.md).

**Why:** the newly verified count tests the earlier heading-based category hypothesis.
It establishes role mobility, but both a responsible person's name and an occupational
class can produce the observed count/heading pattern.

**Decision:** no semantic winner or parentage label. Retain the contrast as a counterexample
to assigning name/category labels from position alone. A known-language name-versus-category
control is needed before further semantic modelling; more unlabelled repetitions alone do
not distinguish these rivals. No old frozen input, failed gate or concurrent Etruscan file changed.

## 2026-09-24: Linear B name/designation structural control failed

**What was done**

- Built a curated challenge with 38 source-labelled cases on 19 objects: 19 personal names
  and 19 occupation/title/group/status designations. Inspected the printed source, retained
  DĀMOS text, recorded exclusions and source discrepancies, and separated anonymous features
  from the gold key. Labels still lack independent modern specialist adjudication.
- Froze code, sources, labels, settings and eight new tests at `2574eef` before scoring.
  Used whole-object holdout and one normalized vote per training object/signature.
- Primary layout: 37.4% balanced recall, 39.4% coverage, 95.1% weighted conditional accuracy;
  raw 15 correct, one wrong, 22 abstentions. Recall and coverage fail the 90% thresholds.
  Forced-majority layout recall is 57.4%; the in-sample feature ceiling is 84.9%.
- Ran 199 whole-object label-swap negatives: 177 evaluable, zero passes, 22 excluded by
  representation requirements. Passed 57 focused tests and verified all nine follow-up
  freezes. [Report](experiments/linear-b-person-role/REPORT.md).

**Why:** known personal names and occupational labels can have identical role, quantity,
position and repetition features. The name o-wo-to on PY An 261 is misclassified from five
unanimously designation-labelled supporting objects; training agreement does not certify meaning.

**Decision:** retire this structural classifier for semantic transfer. No Linear A was scored,
no parentage/gender inference was made, and no earlier failed gate or frozen input changed.
Any next experiment needs additional evidence, such as relational frames and morphology, plus
known-language tests separating parentage from ownership/responsibility and occupation.

## 2026-09-24: Relational family-reference control and expanded son inventory

**What was done**

- Reviewed Hiller (1989), Duhoux (2007), Godart (2024) and the Oe106 translation disagreement.
  Added 19 scored expressions on 16 objects, with seven unresolved cases outside gold.
  Supplied onomastic spans, fragment boundaries, cross-line joins and terminal -qe segmentation;
  hid spellings, Greek meanings, grammatical case, gender and argument roles from the learner.
- Froze code, cases, source PDFs and all 5,932 DĀMOS snapshots at `095dbdc` before scoring.
  The expanded literal inventory adds three intact i-*65/i-*65-qe occurrences and preserves
  two doubtful counterparts separately. Overall retrieval: 59 hits, including 22 intact
  attached-tail candidates and 25 uncertain-text candidates. These are not semantic labels.
- Form-only diagnostic: 50% family recall, 41.7% balanced recall, 50% coverage; seven correct,
  one wrong, 11 abstentions. The primary ordered frame/form control abstains on all 19 cases:
  11 unseen patterns, eight supported by only one other object, where two were required.
- Gate failed. Zero of 196 evaluable negatives pass, but all abstain because structural support
  is absent; that negative result is uninformative. Three of 199 swaps fail representation.
  Passed 66 focused tests, verified ten freezes and reproduced results/inventory exactly.
  [Report](experiments/linear-b-relations/REPORT.md).

**Why:** relational wording adds evidence absent from position alone, but recognizing a family
expression does not determine its participants. MY Oe106's named person can be read as daughter
or parent; the anonymous two-name patronymic also collides with ordinary co-listing.

**Decision:** no parentage edges or Linear A meanings inferred. Four supplied family cases have
an unnamed child, two have disputed binding, and one has a lineage qualification. MY Au102
and KN Vs1523 readings remain unresolved. Further controls need independently supported repeated
constructions; the next source question is the Theban attached-u-jo versus *65/FAR comparison.
Earlier frozen inputs, failed gates and concurrent Etruscan files remain unchanged.

## 2026-09-24: Theban *65/FAR source and quantity audit

**What was done**

- Preserved 13 selected objects, including one joined Fq254[+]255 and two external commodity
  controls. Reviewed Palaima, Duhoux and James; original TOP surfaces and the direct AGS
  response remain unavailable. Son and commodity readings stay explicit rivals.
- Froze exploratory source annotations/code at `aeb16d9` before generating comparisons.
  Three of six published total conversions disagree with their printed components:
  Fq252 186/178 Z, Fq254[+]255 169/86 Z, Fq277 525/641 Z (published/calculated).
  Neither column is silently corrected; gaps and doubtful readings remain unresolved.
- Conditional visible allocation shifts: Fq214 8 Z; Fq254[+]255 16 Z. Both accounts fail
  complete-body/secure-total/scope eligibility, so no balance verdict or family edge follows.
- Passed 74 focused tests, verified 11 freezes, reproduced output exactly and checked overwrite
  protection. [Report](experiments/linear-b-theban-65/REPORT.md).

**Why:** a food amount after a person does not distinguish “son” from a commodity marker.
The reading must be tested against sign placement and a secure accounting scope, not damaged
or inconsistent totals. Published -u-jo parallels motivate further source review, not gold labels.

**Decision:** retain the Theban son proposal as a qualified lead; no Linear A transfer or new
parentage/gender labels. Next inspect TOP pp94–95 and the related tablet surfaces, alongside
the full AGS2003 reply. Earlier failed gates and frozen files remain unchanged.

## 2026-09-24: Theban image search reaches its source-access stopping point

**What was done**

- Searched for full-context images of Fq236/Gp124/Gp227/Fq254[+]255. LiBER excludes Theban
  archival texts; the four DĀMOS image fields are catalogue links. Publisher and book-preview
  routes did not supply the requested pages. No claim that scans do not exist elsewhere.
- Pinned Palaima2006, Judson2016, a museum-volume reference and the LiBER coverage snapshot.
  Inspected Judson's Gp124 glyph reproduction: it lacks adjacent signs needed for spacing.
- Recorded later scholarly disagreement over Fq236 and the reported hand310-to304 reassignment.
  Prepared an unsent scan-request packet. No measurements, new code/tests, labels or freeze.
  All 11 earlier follow-up freezes remain unchanged.
  [Report](experiments/linear-b-theban-images/REPORT.md).

**Why:** a glyph crop cannot establish attachment, and damage may prevent even a full image
from deciding the reading. The later literature narrows what the proposed check can establish.

**Decision:** park this image-dependent lead until full-context scans arrive. Do not rerun the
same source searches or treat transcription whitespace as a photograph. No Linear A transfer,
family edge or gender inference follows. No external request was sent.
