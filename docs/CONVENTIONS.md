# Conventions: how an experiment is run here

These rules keep results auditable. They are enforced by the drivers where possible and by
the record otherwise.

## The recovery cycle

```text
develop     tune on development text only; write results.json and PROTOCOL.md
freeze      hash code, inputs and settings into freeze.json
commit      the freeze must be in git before any sealed passage exists
prepare     draw fresh passages, encrypt with hidden keys, keep answers evaluator-only
solve       decode from ciphertext alone; checkpoint per case
evaluate    open answers once; write results.json; archive evaluated-records.tar.gz
report      REPORT.md: what was done, why, result, limits; update the log
```

`verify` fails if any frozen file changed or the freeze is not committed. `prepare`, `solve`
and `evaluate` refuse to overwrite existing outputs.

## Rules

1. **Development, evaluation and Voynich text never mix.** Development text: held-out tales,
   ISDT dev, Petrarch dev poems. Evaluation text: ISDT test and Dante, excluded by source ID
   after use. Voynich final-test pages are never scored.
2. **Every sealed passage is used once.** After grading, its source IDs go into the exclusion
   list of every later round.
3. **Declare hints.** A protocol states exactly what the solver receives (language, alphabet,
   token class) and what it does not (codebook, key, trace, length).
4. **Fix the gate before running.** The pass mark and secondary measures are written in the
   protocol; a failed gate is reported as failed, with the shortfall.
5. **Do not edit frozen files.** New behaviour goes in a new module (`_v2`, `_v3`) so earlier
   rounds still verify.
6. **Record negatives.** Ideas that were tried and rejected stay in the development report
   with their numbers.
7. **Four passages are four units.** No significance claims; blinding is procedural on one
   machine, not an independent evaluator.
8. **CPU only, no paid compute, pinned downloads.** Every external text has a revision, a hash
   and a licence in a `sources.json`.

## Adding a round

1. Copy the previous `joint_development_vN.py` and `joint_recovery_vN.py`; bump `N`.
2. Put changed algorithms in a new library module; import unchanged ones.
3. Add the previous round's `artifacts/joint-recovery-vN/evaluator-only/answers.json` to `excluded_ids`.
4. Write `experiments/joint-development-vN/PROTOCOL.md` before development results exist.
5. Run development; commit protocol, results and code.
6. `freeze`, commit, `verify`, `prepare`, `solve`, `evaluate`, `REPORT.md`, update `RESEARCH_LOG.md`,
   `README.md`, `docs/RESULTS.md`; archive records; commit.

## Writing

Reports follow one shape: what was done (bullets), why (two sentences), results (tables, then
numbered points), what it does and does not establish, records and reproduction. Every claim
carries a number and a link. No translation claims.
