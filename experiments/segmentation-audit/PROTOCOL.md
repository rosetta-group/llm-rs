# Segmentation audit and one bounded reparse experiment

Declared before running. CPU only. No Voynich text, fresh evaluation answers, or
paid services. Existing frozen modules remain unchanged.

## Diagnostics

Use the existing three development streams: Novellino/Decameron, ISDT development,
and Petrarca development. Retain the first 5,200 letters and reference word breaks;
if the cut splits a word, exclude that terminal fragment from word scoring. Recreate
the existing development encryption (seed 6200).

1. Give the frozen word segmenter perfect letters; record word error and boundary F1.
2. Run the frozen round-four decoder; record character and word error.
3. Supply the true cipher-piece inventory to joint EM, then refine and polish with
   the same settings. This is a labelled diagnostic, never a candidate method.

## Targeted change

Keep the round-four key fixed. Search whole-token and prefix+suffix readings whose
role-specific pieces already have a decoded letter. A beam of 128 states scores
full character context (the frozen five-gram prior), plaintext length and the cost
of choosing among homophones within each role. Transmitting the fixed key and one
whole/split bit per token adds constants. No new pieces, key fitting, word-model
changes or parameter sweep. This conditional fixed-key score is not the general
Voynich MDL score. If the beam candidate costs more than the unchanged parse under
this same score, retain the unchanged parse.

Select the change only if mean development character error improves by at least
one percentage point, no source worsens by more than one point, and mean word
error does not increase. Otherwise record a negative result and do not consume
fresh passages. Diagnostics can motivate a later separately declared experiment;
they must not turn into an undeclared search in this run.

If selected, freeze before preparing fresh cases. Compare baseline and candidate
on identical passages and keys, including an unseen historical author and a newly
pinned modern corpus. Passage independence matters more than repeated keys. The
existing recovery gate stays CER <= 1%, WER <= 10%; passing this Italian/known-class
control still would not identify Voynich's language or cipher family.

## Provenance

Archive settings, hashes, development ciphertext, predictions and evaluator-only
diagnostics after the run. Source splits and the existing round-four prior remain
unchanged. No claim of independent test performance from this development study.
