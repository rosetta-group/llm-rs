# Rejection and frozen-key transfer: bounded screening round

Declared 2026-09-24 before generating or decoding any new challenge. CPU only, at most
five single-threaded workers. No Voynich text. This changes the research question, not
the existing 1% CER / 10% WER recovery gate.

## Question and scope

Can the existing five-language decoder accept a supported cipher, reject inputs without
an ordered source message, reject a missing source language, and transfer its learned
key to a new passage without refitting? This is a **screen**, not a population estimate
of 90% sensitivity or 5% false acceptance. A failure parks this version of the method;
a pass licenses a larger, separately frozen control study, never a manuscript run.

The tested decoder is the frozen `language-id` pipeline: round-four joint EM, pruning,
lexicon repair and refinement, without an Italian-only lexical polish. Its five priors
and settings are reused byte-for-byte. Round six is newer and better at long Italian
recovery, but it has not been validated for five-language scoring. No decoder retuning
or switch to round six is allowed after observing this screen.

## Development calibration

The five released `language-id` cases are development evidence for this question, not
fresh confirmation. Their correct-prior excesses were 0.23–0.48 bits per letter and
their winning margins 0.75–1.33. Fix the acceptance ceiling at **0.50 bits per letter**,
the winning margin at **0.25**, and transfer token coverage at **95%**. These are
engineering thresholds, not estimates of a null distribution. They are not adjusted
against the new negative controls. Record this calibration before the challenge exists.

## Sources and independent units

Use new UD test splits for Latin ITTB, Old French PROFITEROLE, German GSD and English EWT,
at the repository revisions already pinned in `language-sources.json`. Use the newly
pinned Italian PUD test corpus. Fetch and hash these before freeze. For each language,
reserve four disjoint passages of 5,200–6,000 normalized letters: two fit/transfer pairs.
Only sentence counts, exclusion counts and available lengths are examined before freeze.

Exclude every normalized sentence occurring in the existing language-source training
files, historical prior material, or earlier released Voynich recovery passages. Reject
8-word overlap with those sources and between retained challenge sentences, including
short exact-sentence duplicates. The whole source sentence is reserved even if a passage
ends there. Record source sentence IDs privately until grading. Check source SHA-256.

There are at most **10 independent passage/key blocks**, two per language. Each block
gets an independent random letter permutation; the fit and transfer passages share that
permutation but have separate encryption seeds. The Naibbe tables and RESPACING remain
the published default, as in the language-ID pilot. This does not validate RESPACING 9.

Run blocks in this fixed order: English, Italian, Latin, German, Old French, repeated.
Public fit/transfer IDs are opaque. Gold text, language, transformations, permutation
and seeds are evaluator-only. Procedural separation is not adversarial isolation.

## Three paired inputs per block

1. **Positive:** fresh Naibbe ciphertext from the two source passages.
2. **Shuffle:** independently permute all tokens of each positive passage; preserve its
   token inventory and frequency exactly. No ordered source passage remains.
3. **Copy/mutate:** generate the same number of tokens from the fit passage's empirical
   token inventory. Seed 32 tokens by sampling that inventory. At each subsequent step,
   copy a token from the last 50 with probability 0.6; copy and make one random glyph
   insertion/deletion/substitution with probability 0.2; otherwise sample the inventory.
   Fit and transfer outputs have independent random streams but the same empirical
   inventory. This is a declared stress generator, **not** an implementation of Timm's
   published generator and not an exhaustive model of meaningless text.

Both negative fit passages get a complete independent five-prior key search. Merely
applying a key learned from meaningful text to nonsense would be too weak a control.

## Frozen-key transfer and acceptance

For each fit passage, learn five independent role-specific keys with the old decoder.
Save keys and fit predictions before reading transfer ciphertext. On the transfer
passage, only choose among known whole-token / prefix+suffix readings using the existing
context-reparse beam (width 128). Never admit a piece, change a letter assignment, or
refit a key. Unknown tokens produce a gap and reset language context. Report token and
glyph coverage; reject coverage below 95%. Do not silently concatenate across gaps.

Score each language with the old excess: prior bits per recovered letter minus that
prior's frozen held-out entropy. Transfer scores sum the prior's bits over covered runs,
including their length codes; missing tokens are explicitly governed by the coverage
gate. Empty/entirely uncovered text is unscorable and rejected.

```text
Fit five keys on the fit ciphertext and seal them
Decode transfer ciphertext with those keys held fixed
Rank all candidate languages by excess separately on fit and transfer
Accept only if both rankings have the same winner
Require fit and transfer excess <= 0.50 and both margins >= 0.25
Require transfer token coverage >= 95% and no compute cap hit
Otherwise return none of these
Repeat the decision after removing the true language for the absent-language control
```

The absent-language control uses the positive's four wrong-prior fits, without any
rerun or retuning. It is paired with its positive, not a further independent key.
Report CER on positive fit and transfer passages after predictions are sealed; missing
tokens count as errors, not omissions. Also report in-sample-only decisions to expose
what the transfer requirement changes. No decoded prose is treated as evidence itself.

## Gates, stopping and resources

At the planned ten blocks require >=9/10 correctly accepted positives, and **0/10 false
acceptances in each** of shuffle, copy/mutate and absent-language controls (at most 5%
of ten means zero). Do not pool controls to dilute a failed class.

Process each block positive first, then shuffle, then copy/mutate. Finish and seal the
five-prior predictions for an input before grading it. **Stop immediately** when a
negative is accepted, a positive is accepted as the wrong language, or two positives
are rejected: the final declared screen can no longer pass. Do not decode remaining
cases merely to accumulate results. Report the actual denominator, omitted cases and
reason. Early-stop proportions are descriptive, not unbiased population estimates or
fixed-sample confidence intervals. A cap/resource failure yields an inconclusive stop,
not a successful rejection.

Keep the existing per-fit cap (3,600 seconds), stage caps and settings, run no more than
five workers, and budget at most 24 aggregate fit-worker hours. Stop inconclusively if
that budget is exhausted. Checkpoint every fit and transfer; never overwrite predictions
or grade incomplete input bundles. All earlier frozen files remain unchanged.

Even a complete 10/10, 0/10 result would be imprecise: zero failures in ten independent
blocks has a one-sided 95% binomial upper bound of about 26%. Zero in 59 would be needed
to bring that bound below 5%. A subsequent confirmation must budget enough independent
keys/passages and account for shared authors and paired negative controls.

## Reproduction and records

```text
sources       pin downloads, build the exclusion inventory, report counts only
freeze        hash protocol, code, priors, sources and development calibration
commit        commit the freeze before preparing any ciphertext
prepare       generate and seal all ten blocks
run           fit, seal, transfer, seal, grade, stop by the declared rule
report        preserve results, partial denominators, decisions and released records
```

Driver: `python -m experiments.rejection_transfer`. Tests must cover missing-language
rejection, known-key transfer, gaps and coverage, no transfer refitting, leakage checks,
stopping boundaries, and provenance failures before the freeze is committed.
