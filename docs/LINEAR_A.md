# Linear A: five rounds, a repair audit, and one structural test

This page summarises the Linear A track (branch `linear-a`, 2026-09-23). It does not contain a
translation. It records whether word-matching methods can tell which language Linear A is, and
why the tested methods cannot at present. A [repair audit](../experiments/linear-a-audit/REPORT.md)
and [structural test](../experiments/linear-a-structure-v2/REPORT.md) followed on 2026-09-24.
These methods are retired; the evidence does not establish a limit on all possible methods.
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

## Data and licences

Linear A: Navarre-AI collation of SigLA and lineara.xyz (CC BY; SigLA fields CC BY-NC-SA 4.0).
Linear B: DĀMOS, 5,932 documents (CC BY-NC-SA 4.0). Lexicons: Wiktionary via kaikki.org (CC BY-SA),
LAMAN (CC BY-SA 4.0), Oracc (CC0). Running text: TLHdig Beta 0.3 (Zenodo 20328284, CC BY 4.0).
Keftiu names: Peet 1927, *Essays in Aegean Archaeology* (public domain). Results derived from SigLA
and DĀMOS carry CC BY-NC-SA 4.0. All sources sit in `artifacts/linear-a-sources/` (git-ignored) and
are hashed in each round's `sources.json`.
