# Proposed fresh confirmation: not executed

The three follow-ups are development on three released source/key blocks. A larger
study requires a new protocol, new passages, new keys and a source-availability audit.

## Proposed endpoints and sample size

Plan 90 independent source/key blocks covering all five languages and multiple
sources/genres. The intended population is the predeclared language/source mixture;
these counts do not establish a separate error bound for each individual language.
Each block includes a supported positive, a shuffled input, copy/mutate input,
frequency-preserving copying input, and an absent-language decision reusing the
positive's wrong-prior fits. Fit every input under all five priors; transfer sealed keys.

A conservative reference calculation splits alpha=0.05 over five endpoints (positive
sensitivity and four negative classes), alpha=0.01 each. Under independent Bernoulli
trials, zero acceptances in 90 negatives gives a one-sided upper bound
1 - 0.01^(1/90) = 4.9881%. A sensitivity gate of at least
88/90 correctly accepted positives rejects sensitivity <=90% at alpha=0.01
by the binomial tail. Source clustering or nonidentical sampling can invalidate that
simple calculation: define the sampling population and statistical analysis before
freezing, and retain per-language/source results. Paired negative classes are never
pooled to dilute a failure.

## Cost estimate and remaining uncertainty

The completed stronger controls averaged 246.6 fit-worker seconds per fit.
Ninety blocks × four fitted inputs × five priors = **1,800 fits**. Applying the same
cost to all inputs gives **123.3 fit-worker hours**,
or at least 24.7 wall hours with five continuously
busy workers, excluding overhead. A preliminary two-times allowance plus one reserved
five-hour batch is **251.6 fit-worker hours**.

This is a planning estimate, not an agreed run budget or measured cost for the whole
study. The estimate comes from frequency-preserving controls on only three languages;
mutation controls, other sources and harder keys can cost more. Measure representative
released examples of every fitted input class before selecting the final budget.
Per-fit caps and work limits must remain explicit, with cap outcomes inconclusive.

The released mutation fixture has 893 used units, or
418,817 proposals per complete sweep (all 23 letter choices and all
pair swaps). A 20-million-proposal allowance permits only
47 such sweeps. The smaller frequency-preserving controls
can require many more sweeps to complete the inherited 30 kicks. Therefore do not
carry this work allowance into fresh confirmation without first checking convergence
on released mutation examples. No completed follow-up limit was changed in response
to its outcomes.

## Preconditions

```text
Audit enough independent fresh sources and exclude all consumed source IDs
Check projected cost on every development input class
Declare language/source sampling, endpoints, multiplicity and stopping rules
Freeze and commit the candidate rule and selected decoder
Generate new keys and passages only after that freeze
Run the confirmation once and report every failure, cap and omitted case
```

The existing manuscript recovery gate is unchanged. Even a rejection-study pass would
not establish that Voynich belongs to the tested cipher family or identify a word.
