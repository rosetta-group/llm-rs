# Historical word segmentation: add a training-only verse vocabulary

Declared before development scoring. CPU only. No cipher decoding, Voynich text,
image association modelling or paid compute. The older word model stays frozen.

## Hypothesis and bounded comparison

The character prior already includes Petrarca verse; the frozen word segmenter
includes modern Italian and historical prose but no verse training. Test whether
adding training-only verse word counts, word transitions and word forms improves
historical word boundaries without harming modern Italian.

Keep the existing segmentation algorithm and all its parameters unchanged. Add
Petrarca training poems to the frozen word model with multiplicities 1, 4 and 16.
No other parameters, spelling rules or candidate models are tried. Every fifth
poem remains development; no missing development word is manually inserted.

Use the three existing audit streams, ending at the last complete word within
5,200 letters. Score baseline and three candidates on identical perfect letters.
Select the eligible candidate with lowest mean WER on historical prose and verse;
ties use modern WER, then lower multiplicity. Eligibility: at least 3 percentage
points lower historical mean WER, neither historical stream worsens by over 1
point, and modern WER worsens by at most 1 point. If none qualifies, record the
negative and do not consume fresh evaluation text.

## Fresh evaluation, conditional on development selection

Commit model, decoder parameters, protocol, extraction and grading code before
preparing any fresh passage. Use four historical passages from a previously unused
historical author (Giovanni Villani's Nuova Cronica) and four modern passages from
a newly pinned corpus (UD Italian VIT test split). Download only after the method
freeze; record revisions, source hashes and licenses. Extract historical prose
paragraphs, dropping navigation, headings, footnotes and editorial material.
Archive raw sources. Fetch whole chapters in numerical order; retain all usable
paragraphs without selecting by decoder scores. If sources are unavailable or
insufficient, report the blocker rather than silently changing corpora.

Build four disjoint passages per source of 5,200–6,000 letters, cutting only between
complete paragraphs/sentences (split an overlong paragraph between whole words).
Reject exact shared 20-word sequences with fitting/development texts, as an overlap
check, not a claim of complete provenance independence. Save reference spaces in
an evaluator-only file; expose only dense letters and opaque IDs to the solver.
Save all baseline and candidate predictions before opening answers. This is
procedural blinding on one machine, not independent third-party evaluation.

Compare paired pooled WER, per-passage WER, boundary precision/recall/F1 and exact
character preservation. Report each source separately and all failures; no tuning
after grading. Four passages from one author are not four independent authors.
A useful transfer result requires historical pooled WER improvement >= 3 points
and modern worsening <= 1 point. Separately report the stricter WER <= 10% gate
per passage; character error is zero by construction. Passing segmentation alone
never opens the Naibbe/Voynich recovery gate.

## Records

Store all tried settings and development outputs. Fingerprint frozen code, source
manifests, baseline and selected model bytes. Preserve old experiments. Freeze
before fresh source preparation and verify that Git contains the exact freeze.
Release graded references only after predictions are saved. Future evaluations
must exclude these source IDs. No paid model or model-weight download is needed.
