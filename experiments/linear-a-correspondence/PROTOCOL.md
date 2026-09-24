# Correspondence source audit and conditional ending tests

Date: 2026-09-24. Status: freeze before real-data permutations. No paid compute.

## Question and prior exposure

Do literal Linear A words ending in `re` or `ru` share stems with Linear B words ending in
`ro` more often than expected after preserving the real stems and ending frequencies?
The historical 12 matches and their source readings have already been inspected. This is a
robustness audit of an exploratory lead, not a new independent confirmation or a language test.
The old syllable null gave 12 versus 2.9244, p = .0001 after its simulation-budget repair.

Pattern name: **conditional ending test**. A stem is every sign except the last; for example,
`qa-qa` is the stem of `qa-qa-ru`. A type is a distinct complete sequence of literal sign labels.
The null is exchangeability of final labels among the declared Linear A type slots. It does not
assert that languages form words randomly, or that matching tokens denote the same name.

```text
Audit all 12 historical pairs against pinned source records
Extract strict types across the entire corpus using the same rules
Commit code, source evidence, inputs, protocol and hashes
Permute endings within length; then within length and final onset
Require the strict dataset to pass both declared comparisons
If either fails, archive the statistical adaptation lead
If both pass, retain only for genuinely independent evidence
```

## Sources and extraction, fixed before simulation

Use the existing pinned Navarre corpus (revision `3a83a327505bd5e01c41f05185fdcd74a40fb062`)
and all 5,932 archived DĀMOS JSON items. The new freeze hashes those files, the complete extracted
inventories, accepted attestations, evidence for every original pair, code, tests and this protocol.
`PAIRS.md` gives the human-readable source decisions. No new attestations are introduced.

Primary **strict** inventory:

- Deduplicate types; require at least three syllabic labels. Keep `q` distinct from `k`, `a3`
  from `a`, and `ta2` from `ta`. An alphabetic label may have a numeric sign suffix; unknown
  sign IDs and marked readings are excluded. No inferred sound changes within stems.
- Linear A: require `word_source == sigla`, exact agreement with a complete source word,
  an intact Unicode run, and corresponding contiguous source sign IDs whose readings agree
  and whose occurrences are certain syllabograms. Every matching ID span in that inscription
  must meet the certainty/reading criterion. This conservative rule avoids transferring
  certainty from another occurrence. A conflict naming the candidate word excludes it;
  unrelated conflicts do not exclude the whole inscription.
- Unicode parsing retains the old fixed 80% corpus logogram boundary rule. Numerals, fractions,
  dividers, whitespace and line edges bound runs. A gap or other editorial mark adjoining a
  run invalidates it; it is never a word separator. This may omit genuine complete words at a
  broken edge. It is a reproducible source-quality filter, not a specialist epigraphic judgement.
- Linear B: accept only whole unmarked tokens from `content`, split on whitespace, comma or
  slash. Never strip underdots, question marks or brackets to create a word. An open bracket
  (`[`, `⟦`, `<`, `{`) excludes following material until its close on the same line;
  unmatched openings exclude the rest of that line. Broken edges are marked independently
  on each transcription line (e.g. KN Dc 1272 .A ends in `19̣[`, while .B starts in intact
  `a-ti-ro`). Attached closing marks also exclude tokens. No multiline restoration is inferred.
- SigLA-derived word/sign fields and lineara.xyz glyphs are collated layers, not independent
  witnesses. No claim of autopsy, dating adjudication or independent reading of photographs.
- Multiple occurrences, tablet faces, and `re`/`ru` variants of one stem get only one vote.
  Both source inventories are filtered in full, including nonmatching words.

Secondary **historical** sensitivity inventory: reproduce the existing normalised types of
length >=3, including q/k conflation, collapsed numbered signs and damaged readings. Its results
cannot override the strict decision. No third inventory or alternative threshold is selected.

## Statistic and two nulls

Count distinct Linear A stems with at least one `re` or `ru` ending and an identical Linear B
stem ending in `ro`. Linear B remains fixed. For each null draw, permute complete final-sign
labels over the original Linear A type slots. Keep each stem, its length, the slot count, and
the final-sign multiset. Synthetic duplicate words are possible; they do not add votes.

1. **Length:** permute within exact word length. Controls for existing stems and ending counts.
2. **Length + onset:** additionally condition on the final sign's onset, obtained by removing
   its last vowel and numeric suffix (`re`, `ru`, `ro`, `ra2` have onset `r`). This tests the
   specific final-vowel pattern beyond a generic final-r correspondence. It deliberately
   conditions away a possible broader r correspondence; failure does not refute that pattern.

Each dataset/null uses 9,999 draws. Seeds: strict length `24092401`, strict length+onset
`24092402`, historical length `24092403`, historical length+onset `24092404`. Store every score,
its frequency distribution, mutable-slot counts, and the observed matches. Fixed singleton or
single-ending buckets remain fixed. No adaptive extension or rerun with favourable seeds.

Use the one-sided plus-one p-value `(1 + count(null >= observed)) / 10000`; ties count against
the lead. Retain the original seven-probe threshold `alpha = .05/7` for continuity, not as a
claim that it corrects all prior exploration. Require **both strict** p-values and both Wilson
95% upper bounds on null exceedance probability to be below alpha. These intervals quantify
Monte Carlo error only, not uncertainty about readings, model assumptions, or the hypothesis.
Conjunction of two gates is conservative; the secondary historical results are descriptive.

## Validation and decision

Before freezing, test literal q/numbered-sign distinctions, uncertainty and bracket spans,
Linear A gap/certainty/source conflicts, preserved permutation inventories, duplicate-stem
votes, equivalence of optimised scores to full permutations, a planted synthetic association,
and an all-ties negative case. Synthetic trials are software checks, not linguistic controls.

If either strict comparison fails, archive the claimed statistical support for the specific
`re/ru -> ro` adaptation. Individual cognates, loans or names remain unresolved. If both pass,
seek dated, context-compatible and independently read attestations not already used here.
Do not call another split of this already-inspected corpus independent evidence.

The old frozen implementations/results remain unchanged. Any correction needed after this
freeze requires a new version and an explicit deviation record, even if scores have not run.
Derived source transcriptions retain CC BY-NC-SA 4.0 attribution to SigLA/DĀMOS; see `PAIRS.md`.
