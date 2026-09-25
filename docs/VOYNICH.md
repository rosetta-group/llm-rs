# The Voynich track: what was done, what was found, what it means

Status on 2026-09-24. This document summarises the whole Voynich effort in this repository. Every
number comes from a frozen, linked record. The [research log](../RESEARCH_LOG.md) is the full
chronology, the [results table](RESULTS.md) lists every result, and the [glossary](GLOSSARY.md)
defines terms.

**Headline.** No Voynich word, passage or language has been identified, and no claim of one is made.
What exists is a validated decoding method for one candidate cipher family (Naibbe), tested on sealed
Italian text. It now recovers about **98% of letters** from ciphertext alone, on sealed 20,800-letter
historical passages. It also has three checks that any Voynich claim must pass, two of which the
Naibbe hypothesis only partly passes.

---

## 1. The problem and the rules of evidence

The Voynich manuscript is an early-15th-century book in an unknown script. There is no answer key:
nobody knows a single word for certain. So every method here follows the same rules.

- **Validate where the answer is known.** Methods are developed on synthetic ciphertext made from real
  text, then tested on **sealed** passages. The originals stay hidden until predictions are saved.
- **Declare before running.** Each experiment commits its protocol, settings and pass mark before any
  result exists. Failures are kept as failures, and a result just short of its bar is a miss.
- **Keep development and test apart.** Development texts are reused; test passages are used once and
  then released and excluded. Leakage found after grading is corrected in the record (see §6).
- **Never score the reserved Voynich pages.** The final-test pages of the manuscript have never been
  scored. The "Voynich mechanism test" stays closed until a recovery gate is met.

**Recovery gate:** 1% character error (CER) and 10% word error (WER) on a sealed case.

**Metrics:**
- **BPC:** bits per character, for prediction; lower is better.
- **CER / WER:** edit distance per letter or per word, against the hidden original.

---

## 2. Tracks at a glance

| Track | Status | One-line finding |
|---|---|---|
| Prediction with language models | closed | Voynich is predictable, but every gain also appears on shuffled and synthetic controls |
| Corpus statistics | complete | Word lengths cluster, which rules out plain spaced substitution; the language is not identified |
| Text–image association | parked | No association beyond scribe hand and layout; annotation pilot awaits human review |
| Cipher recovery (Naibbe) | active | From 300%+ CER to 1.83% on sealed text; gate not met |
| Mechanism checks on the Voynich text | active | Near-duplicate words point to a mechanical process; Naibbe fits only with heavy letter pairing |

---

## 3. Prediction (closed)

Qwen 1.7B and 8B were adapted with LoRA to predict Voynich characters, on fixed page splits with
controls. Small GRU and transformer models were trained from scratch for comparison.

| Result | Number | Record |
|---|---|---|
| Frequency baseline / best copy baseline | 4.245 / 2.699 BPC | [RESULTS](../experiments/RESULTS.md) |
| Qwen 1.7B adapters, 5 seeds | 2.483 BPC (SD 0.002) | [REPLICATIONS](../experiments/REPLICATIONS.md) |
| Outer vs random layers | 2.4819 vs 2.4828 | [LEARNING_CURVES](../experiments/LEARNING_CURVES.md) |
| Qwen 8B vs 1.7B | 2.469 vs 2.477; the interval crosses 0 | [CLOUD_RESULTS](../experiments/CLOUD_RESULTS.md) |
| Character GRU from scratch, 3 seeds | 2.332 BPC | [CHARACTERS](../experiments/CHARACTERS.md) |
| Same gains on shuffled, Timm and Naibbe controls | yes | [report](../experiments/report-completed/REPORT.md) |
| Matched-context retraining, 18 runs | intact gain 0.123 vs shuffled 0.137 BPC | [report](../experiments/report-matched/REPORT.md) |

**What it means:** prediction measures structure, not meaning. A small model beats the pretrained one.
Pretraining on real languages does not help. The track closed on 2026-09-21.

---

## 4. Corpus statistics and text–image work

- **Word-length adjacency** ([report](../experiments/language-comparison/REPORT.md)): Voynich gives
  +0.218 (v101) and +0.169 (EVA); 11 natural languages give −0.18 to +0.07. This rules out plain
  substitution of a European language that keeps its word spaces. It does not identify the text.
- **Image association** ([pilot](../experiments/association/REPORT.md),
  [complex](../experiments/association-complex/REPORT.md),
  [domains](../experiments/image-domains/REPORT.md)): no signal beyond scribe hand and layout. Picture
  type, hand and quire move together in this manuscript. A 24-panel
  [object–relation pilot](../data/folios/object-pilot/REPORT.md) exists as AI development annotations.
  Independent human review is still pending.

---

## 5. Cipher recovery: the Naibbe benchmark

### Why Naibbe

Naibbe (Greshko 2025) is a published cipher that turns Latin or Italian letters into Voynich-like
glyph strings. Each letter, or each pair of letters, becomes one of several glyph strings, chosen by
drawing cards. It reproduces many of the manuscript's statistics, so it is a demanding, realistic
test cipher.

### Setup

The solver gets the ciphertext and a character model of Italian. It gets no codebook, no key and no
paired examples. Each passage is encrypted with a fresh hidden letter permutation and a fresh Naibbe
seed.

### The decoding method, as it now stands

```
joint EM over hidden splits and letters  (thresholds scaled with √length)
usage pruning, joint EM again
lexicon repair: drop bigram concatenations, admit complements of known halves
refine the key (iterated local search)
repeat 2×: context reparse of every token (5-gram beam, width 128) → refit the key
lexical polish of the key
word segmentation with the v3 segmenter (letter-level unknown-word model)
```

### Sealed results, in order

| Round | Change | Letters per case | CER | WER | Record |
|---|---|---:|---:|---:|---|
| Standard methods | MDL, Nuhn beam, BK&K HMM | 1,200–1,800 | 300%+ | — | [report](../experiments/standard-decipherment/REPORT.md) |
| One | joint EM over splits and letters | 5,200 | 33.5% Dante | 87% | [report](../experiments/joint-recovery/REPORT.md) |
| Two | + usage pruning, + Petrarca verse prior | 5,200 | 10.3% Dante | 60% | [report](../experiments/joint-recovery-v2/REPORT.md) |
| Three | + lexical polish | 5,200 | 10.5% Dante | 56% | [report](../experiments/joint-recovery-v3/REPORT.md) |
| Four | + lexicon repair | 5,200 | 5.7% Dante | 45% | [report](../experiments/joint-recovery-v4/REPORT.md) |
| Five | + v3 segmenter (paired) | 5,200 | 5.63% | 45.8% → **41.5%** | [report](../experiments/joint-recovery-v5/REPORT.md) |
| **Six** | + √-scaled thresholds, + context reparse with key refit (paired) | 20,800 | 3.85% → **1.83%** | 34.1% → **26.8%** | [report](../experiments/joint-recovery-v6/REPORT.md) |

Round six's valid cases are two Dante and two Compagni passages; all four improve. Its two modern cases
were invalid (§6). No case has met the gate. The best historical case was 1.52% CER.

### How each improvement was found: the development chain

| Step | Finding | Record |
|---|---|---|
| Search, not scoring | the true key scores best under every objective; only EM reaches it | [dev](../experiments/joint-development/REPORT.md) |
| Length is a hard limit | 77% CER at 1,300 letters, 4.6% at 2,600, 3.2% at 5,200 with true splits | same |
| The lexicon limits parsing | with the true piece list, the same EM reaches 97% agreement and 0.5% CER | [dev](../experiments/joint-development-v4/REPORT.md) |
| Longer text helps up to a point | mean CER 5.62% → 3.39% → 4.18% at 5,200 / 10,400 / 20,800 letters; false pieces explode at 20,800 | [dev](../experiments/length-scaling/REPORT.md) |
| Thresholds must scale with length | linear scaling fails (5.14%); √ scaling gives 3.19% / 3.25% | [dev](../experiments/length-scaling-v2/REPORT.md) |
| Context reparse must refit the key | fixed-key version rejected (−0.29 points); with refit, 1.79% at 20,800 | [dev](../experiments/length-scaling-v3/REPORT.md) |
| False pieces become harmless | pruning them (178–200 → 3–6) moves CER only 1.79% → 1.63% | [dev](../experiments/length-scaling-v4/REPORT.md) |
| Missing pieces are the bottleneck | 33–38 true pieces are still absent after all repairs | same |

### Word segmentation (letters to words)

Correct letters still leave word boundaries to guess, because the ciphertext has no reliable spaces.

| Step | Finding | Record |
|---|---|---|
| Audit | even perfect letters give 15.3% WER on historical prose, 28.5% on Petrarca | [audit](../experiments/segmentation-audit/REPORT.md) |
| Diagnosis | 83–90% of wrong spaces fall inside words missing from the lexicon; a flat unknown-word penalty always loses to a split | [diagnosis](../experiments/word-segmentation-v3/REPORT.md) |
| **v3 segmenter (adopted)** | letter-level spelling model for unknown words; fresh Compagni WER 24.0% → 17.5%, transfer passed | [fresh test](../experiments/word-segmentation-v3-fresh/REPORT.md) |
| v4 (not adopted) | historical spelling mix; fresh Sacchetti 8.82% → 6.87%, 0.05 points short of its bar | [fresh test](../experiments/word-segmentation-v4-fresh/REPORT.md) |

### Negative results, kept on purpose

| Candidate | Outcome | Record |
|---|---|---|
| Verse word model for the segmenter | Villani 28.4% → 27.3%, below the 3-point bar | [report](../experiments/word-segmentation-v2/REPORT.md) |
| Self-inclusive glue test | CER 35–87%; it also dropped true one-letter pieces | [report](../experiments/length-scaling-v3/REPORT.md) |
| v3 cost inside polish | CER worse in 5 of 8 cases; the gain was segmentation only | [report](../experiments/joint-recovery-v5/REPORT.md) |
| Linear threshold scaling | loses rare true pieces; 5.14% | [report](../experiments/length-scaling-v2/REPORT.md) |
| Context admission of rare pieces | about 1,000 admitted, 2–3% true; CER tripled | [report](../experiments/length-scaling-v5/REPORT.md) |

---

## 6. Corrections recorded after grading

- **ParTUT train is ISDT training text.** 1,753 of its 1,781 sentences appear verbatim in UD_Italian-ISDT
  train, which fits the character prior and the segmenter. The modern cases of round six and of the v4
  segmentation test are therefore invalid. A 20-word overlap check missed them because most of these
  sentences are shorter than 20 words ([audit](../experiments/partut-overlap-audit.json)). Round six is
  reported on its four historical cases. The earlier v3 test's modern half overlaps ISDT test and dev,
  not train; this is a caveat, not leakage. **New rule:** reject any sentence found verbatim in a fitting
  corpus, whatever its length.
- **Villani rubrics.** Chapter headings survived extraction in the v2 test, 2.24% of words
  ([report](../experiments/word-segmentation-v2/REPORT.md)). A rubric-free extractor followed.
- **Commit order.** Round six's freeze and v4's development record were committed a moment out of
  order, both before any passage existed. The hashes match and the reports say so.

**Fresh modern text now:** UD_Italian-PUD, 1,000 news and Wikipedia sentences. None is verbatim in, or
shares an 8-word run with, any corpus used so far ([manifest](../experiments/modern-fresh-sources.json)).

---

## 7. Checks on the Voynich text itself (no decoding)

### Which language? A control first

Should the manuscript be decoded against an Italian prior? Not by default. A
[language-ID control](../experiments/language-id/REPORT.md) encrypted known Latin, Old French, German,
English and Italian with Naibbe, and decoded each under all five priors, which were fitted on equal
amounts of text. The true language fit best **5 of 5** times, by 0.75–1.33 bits per letter. So the
pipeline can compare candidate languages without assuming one. That doesn't make any of these five the
manuscript's language.

### Rejection and transfer with a fixed key

The [control screen](../experiments/rejection-transfer-v2/REPORT.md) asks whether the
five-language decoder can answer "none of these" and apply a learned key to a fresh
passage without changing it. Each shuffled or copy/mutate control gets its own full
five-prior search. Removing the true language supplies a third, paired negative.

On the first three fresh pairs, the true language wins both rankings in 3/3 cases.
The fixed rule accepts Italian and Latin, but rejects English solely because its
fit excess is 0.584 bits per letter against the 0.50 ceiling. English transfer excess
is 0.374, with 6.91% character error; Italian and Latin transfer errors are 5.85% and
9.37%. Thus the English false rejection is a score-calibration problem on this case,
despite its key transferring. This does not establish a calibrated alternative rule.

The [first attempt](../experiments/rejection-transfer/REPORT.md) stopped because three
wrong-language refinements hit their caps. Its already graded passages were excluded
before the fresh resource-repair freeze. Both attempts retain their original records;
the cap stop is not counted as a correct rejection. See the new report for actual
negative denominators, resource accounting and limits. No manuscript text was used.

### Three development follow-ups completed

The [follow-up report](../experiments/rejection-followups/REPORT.md) records three
changes tested on released cases. Removing only the fit-excess ceiling makes the
candidate accept 3/3 genuine ciphers, versus 2/3 for the original rule. It retains
language agreement, both margins, transfer-score, coverage and cap requirements.

Three new controls preserve each passage's exact token frequencies but favor local
copying. All three are rejected without caps, with 98.47–98.64% transfer coverage;
their best transfer excesses are 3.000, 2.999 and 3.137 against the 0.50 ceiling.
Their rejection therefore survives removal of the earlier missing-piece weakness.
The old Latin mutation control remains inconclusive; it was not reclassified.

A new refiner bounds candidate-key memory and scores only affected n-grams for
one-letter changes, rechecking close minima with the old full objective. On the
released copying fixture, a warmed exhaustive swap batch falls from 4.817 to 0.155
seconds (31.1×), with the same winning swap and exact score. Peak worker RSS falls
from 1,694 to 331 MiB. Chunking alone saves memory but is slightly slower here.
The complete stronger-control run uses 15 fits and 1.028 fit-worker hours; this is
not a matched end-to-end speed comparison. All 180 tests and saved-key replays pass.

These are three shared development blocks, not fresh confirmation. The
[costed confirmation proposal](../experiments/rejection-followups/CONFIRMATION_PLAN.md)
requires new sources/keys and a work-budget audit. A 90-block design would require
1,800 fits: about 123 fit-worker hours at the measured copying-control cost, before
allowances for harder inputs. No such study or manuscript run was started.

### Near-duplicate words: a mechanical signature

A **near-hapax** is a word seen once that is one glyph away from a word seen at least 5 times.
Measured on training pages in 10,000-token windows
([suspects](../experiments/voynich-suspects/REPORT.md)):

| Source | Near-hapax share of tokens |
|---|---:|
| Voynich v101 / EVA | **12.2% / 8.3%** |
| Naibbe (published sample) / Timm–Schinner generator | 5.9% / 5.6% |
| Old French, Finnish, German, Italian, Latin | 1.2–2.2% |

Cleaning the transcription does not remove this: merging uncertain spaces and dropping disputed lines
gives 12.0% and 7.8%. Agglutinative Finnish does not produce it. Mechanical processes do. So the excess
points to a cipher-like or generated text, not to transcription errors or rich morphology. It is not
proof of either.

### Can Naibbe produce it?

A [90-setting grid](../experiments/naibbe-near-duplicates/REPORT.md) over plaintext language, letter
pairing, deck size and space removal:

- **Only heavy pairing matches.** Naibbe reaches the EVA figure (8.33%) only when 75% of letters are
  encrypted in pairs (RESPACING 9); the published setting pairs 53%. The best setting gives 9.36%.
- **It never reaches the v101 figure** of 12.2%.
- **A second statistic disagrees.** At the matching settings, hapax forms are 55–62% of distinct forms,
  against 71% in the manuscript.
- **The decoder does worse there.** On heavily paired development text, the method gets 3.30% CER,
  against 1.79% at the published setting ([report](../experiments/length-scaling-v4/REPORT.md)).

### Transcription suspects

A ranked list of 164 likely misreadings is ready for review against the scans, with the top 50 given as
folio loci (for example `dairn`/`daiin` at f102v1.8). 447 of 2,878 lines differ in word count by 2 or
more between v101 and EVA. These form an exclusion set for robustness checks.

---

## 8. What it all means

1. **No Voynich meaning has been recovered, and none is claimed.**
2. **The decoding method works under known conditions.** For a Naibbe-class cipher of Italian, about
   98% of letters and 73% of words come back from 20,800 letters of ciphertext alone. With perfect
   letters, word error on historical text is still 17–20%.
3. **Naibbe as a Voynich mechanism is constrained, not confirmed.** It needs about 75% letter pairing to
   match the near-duplicate rate, it still undershoots the v101 figure and the share of one-off forms,
   and it is the regime where the decoder is weakest.
4. **The language is open.** Latin is at least as plausible as Italian for an early 15th-century
   scholarly manuscript. Any Voynich run must rank candidate languages with the control's score, and
   must be willing to answer "none of these".

---

## 9. What remains open

| Item | Why | Cost |
|---|---|---|
| Plan fresh rejection confirmation | the three development follow-ups passed their comparisons, but share only three source/key blocks | source audit and representative work-budget checks before a new freeze |
| Admit missing true pieces without false ones | the letter bottleneck (33–38 missing at RESPACING 17, 52–56 at 9) | about 2 h CPU per development run |
| Develop and test at RESPACING 9 | the only Naibbe regime that matches the manuscript | same |
| Historical spelling model, retry | v4 missed by 0.05 points; needs a new author and its own protocol | minutes |
| More candidate languages | Occitan/Catalan, Czech, Hebrew in transliteration, Latin of other genres | downloads, then about 2 h |
| Human review of the 50 suspect loci and the image pilot | needs a human eye | reviewer time |
| Voynich scoring run | still gated: no sealed case has met 1% / 10% | a deliberate decision |

---

## 10. Reproducing

Commands for each experiment are in [REPRODUCE.md](REPRODUCE.md), and the code layout is in
[REPO_MAP.md](REPO_MAP.md). Recovery rounds keep a `PROTOCOL.md`, a `freeze.json` with hashes, and a
`results.json` or `summary.json`. Sealed rounds also keep evaluated records. Development runs are CPU
only. A 20,800-letter decode takes about 70–80 minutes per case. Downloaded sources are pinned by commit
and SHA-256 in each experiment's `sources.json`.
