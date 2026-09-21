# Standard-method comparison and length-aware selection

Status: development protocol; freeze.json and a Git commit must precede fresh evaluation.
The archived experiments stay unchanged. CPU only; no cloud or model downloads.

## Question and gate

Can a fixed decoder recover fresh Italian without a supplied cipher family or key?
This is a method benchmark, not evidence that Voynich is Italian or a substitution cipher.

```text
Pin non-Dante historical prose; split whole tales
Fit priors and word lexicons on training tales and modern ISDT training only
Tune on new non-Dante development passages
Commit code, protocol, source manifest, development results, and freeze
Generate fresh modern and Dante challenges, excluding all earlier source sentence IDs
Save ciphertext-only predictions before opening evaluation references
Grade letters, words, selection failures, and model-class limits
If every Naibbe case has CER <= 1% and WER <= 10%, propose the reserved Voynich test
Otherwise keep the Voynich test closed
```

No image study is authorized. Reopening needs independently made, text-masked labels,
two annotators, agreement statistics, and identifiable hand/quire controls.

## Sources and historical words

Anonymous *Novellino*: 100 tales. Boccaccio *Decameron*: the first two days, 20 tales.
Original Italian Wikisource transcriptions; original works are public domain. Source URLs,
page revisions, attribution, raw and extracted hashes are in sources.json. Transcription
reuse follows Wikisource's CC BY-SA terms. These are edited texts with variable proofread
status, not diplomatic manuscript readings. They must not be described as exact medieval
word-boundary truth. Navigation, headings and notes are removed; prose rubrics remain.
Every fifth tale is development. No tale crosses the split. No Dante enters fitting/tuning.

Fit the existing trie segmenter on modern training plus historical training, with a
modern Morph-it lexicon. Try historical count weights 1, 4, 16 and alpha .1, 1;
bigram .5, unknown penalty 3, beam 8, maximum word length 32 stay fixed.
Choose minimum development WER across 12 modern and 24 historical 80-word windows.
Report the frozen modern segmenter separately on the same fresh evaluation passages.
A historical gain may reflect lexical coverage AND word counts; do not call it a lexicon-only ablation.

## Candidate description length

Minimize total bits, never bits per recovered character:

L = L(plaintext length) + L(plaintext | character prior) + L(key and inventory)
  + L(cipher choices | plaintext, key) + L(representation, unit count, layout).

The prior is an interpolated conditional character model of order 5, strength 5;
unigram pseudocount .1. Lower orders encode the initial prefix. Train on modern ISDT
plus historical training text once each. Alphabet and normalization match the archived
benchmark (21 letters; j→i, k→c, w→uu; accents removed).

The inventory is sorted and encoded as UTF-8 strings with Elias-gamma byte lengths.
Each key entry uses one length bit and five bits per plaintext letter (one or two).
At each recovered-text offset, count all key entries whose expansion matches the
remaining plaintext prefix. A uniform arithmetic code selects the actual cipher unit.
This residual pays for homophones and variable-length ambiguity; collapsing many symbols
to one letter is not free. Transmit unit count and plaintext length with Elias-gamma
codes. Character-mode removed spaces use an enumerative boundary code; unit-mode spaces
are reconstructed between units. Whitespace is normalized before comparison. This is an
ideal code-length criterion, not an implemented binary compressor. All terms are reported.

## Published comparators

1. [Nuhn, Schamper & Ney 2013](https://aclanthology.org/P13-1154/): beam over partial keys.
   Our independent implementation uses their capacity limits, with [their 2014 longest-known-context score](https://aclanthology.org/D14-1184/).
   Unknown positions break context. Use the 2014 extension-order beam (100 states),
   order-1 through order-5 weights (0,1,1,2,3), adapted to our shorter prior. Above 64
   cipher types use the 2013 frequency extension order to bound order-search cost. For each public representation try homophone capacities
   1, 2 and max(2, ceil(inventory/21)*2), deduplicated. The latter is a generic allowance, not a supplied key.
   Develop beam widths 128, 512, 2048 and 8192 on four non-Dante passages, substitution and genuine
   one-letter homophonic controls. Choose mean CER, then smaller width on ties.
   Keep the best complete mapping from each search, then rank candidates by the fixed MDL.
   Cap each beam invocation at 60 seconds (checked between key extensions).
   Differences from the papers: Italian corpus, order 5 interpolated prior, bounded beams,
   no sentence boundary symbol, adapted order-search weights, no Zodiac replication.
2. [Berg-Kirkpatrick & Klein 2013](https://aclanthology.org/D13-1087/): fixed trigram HMM,
   emission EM, uniform random normalized initial emissions, .1 count smoothing, 200 iterations,
   posterior decoding, best likelihood restart. Run eight CPU restarts with a 60-second cap
   checked between restarts. This cap can overrun by one restart. Use the finest supplied
   space-delimited units, otherwise characters; inventory >64 is outside this bounded run.
   Uniform interpolation of the fitted unigram/bigram/trigram conditionals; unlike the paper,
   those component conditionals themselves use smoothing. No Zodiac-specific training text.
   Eight restarts do not reproduce the paper's million-restart study or establish saturation.
   Posterior decoding may use different letters for one symbol, so report this comparator
   separately; do not force its output into the deterministic-key MDL selector.

Both standard model classes emit one letter per observed symbol. They can express the
new genuine homophonic controls. Neither directly expresses Naibbe's one/two-letter units.
Naibbe runs are out-of-class stress tests, not a valid universal cipher rejection.

The archived annealer remains as an explicit candidate-generation ablation (4 restarts,
12,000 proposals each, 40-second cap per candidate). Its old per-character score selects
one baseline answer; the new MDL selector reranks exactly those same candidates as a
selection-only comparison, then also considers published-beam candidates. No old source is
edited. Report whether correct letters existed among candidates but selection lost them.
The one/two-letter annealer is the generic extension-capable candidate source; it is not
presented as a published homophonic solver. A failure there remains a bounded-search failure.

## Fresh evaluation

Two modern ISDT and two Dante passages, each 1,200–1,800 normalized letters.
Exclude source sentence IDs from every earlier decipherment, segmentation and codebook-free
challenge; reject exact training/development sentence overlap. This adds a cross-author
historical test after training on anonymous prose and Boccaccio. Dante is a previously
studied evaluation author; these are fresh passages, not a newly discovered author holdout.

Encode every passage four ways: substitution, genuine one-letter/two-homophone cipher,
variable one/two-letter homophonic cipher, and pinned Naibbe encoder with a hidden global
alphabet permutation. Cipher symbols are opaque. Public files contain only IDs/ciphertext;
answers, traces and keys stay evaluator-only. Verify encoder round trips. Decoder receives
no family, source, gold length, key, trace or plaintext. Public whitespace is observed,
not oracle word spacing. This is a procedural blind split on one machine, not an independent
third-party evaluation; file permissions do not prevent a determined solver reading gold.

Report CER/WER per passage and family, aggregated by dataset, selection ablation, oracle
candidate CER (diagnostic only), exact-letter segmentation with old/new lexicons, budgets,
and all capped/skipped runs. Four source passages are the independent units; 16 encodings
are not 16 independent language samples. No significance or broad-language claim.

## Reserved Voynich mechanism

Not run unless the Naibbe gate above passes. A later committed protocol must freeze candidate
languages, equal training amounts, decoder budgets, comparable lossless coding terms, training
pages only, and within-line shuffle seeds. Selection among languages and restarts must receive
the same treatment for intact and shuffled text. A lower intact MDL rejects that specified
shuffle null for this decoder; it does not establish a translation or prove the cipher family.
A null can also reflect an unsuitable language/model/representation. Final Voynich test sealed.
