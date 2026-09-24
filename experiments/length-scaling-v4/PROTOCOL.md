# Post-reparse pruning, and the decoder on heavily paired ciphertext (development only)

Declared 2026-09-24, before any run. CPU only. No download, no sealed passage, no Voynich text.
Same three development texts, key and seed (7000) as the length-scaling tests, at 20,800 letters.
The baseline is the recorded reparse pipeline R of [length-scaling-v3](../length-scaling-v3/REPORT.md):
mean polished CER 1.79%.

## Candidate P: post-reparse pruning

Run R up to and including the two reparse and key-refit rounds. Then:

1. Drop every candidate piece used fewer than `repair_usage_floor` times in the reparsed segmentation.
   With square-root scaling at 20,800 letters that is 2 uses.
2. Rerun joint EM on the kept pieces.
3. Refine, then two more reparse and key-refit rounds, then polish.

**Selection, fixed now:** P is adopted for a later sealed test if its mean polished CER is at least
0.3 points below R's 1.79%, with no text worse by more than 0.3 points.

## Check H: R on heavily paired ciphertext

The [near-duplicate grid](../naibbe-near-duplicates/REPORT.md) found that Naibbe matches the Voynich
near-duplicate rate only at RESPACING 9, where 75% of letters are paired. Every decoder test so far
used 17. H runs pipeline R unchanged on ciphertext encrypted at RESPACING 9.

H is descriptive; nothing is selected from it. It reports CER, WER, parse agreement and lexicon
counts next to R's values at RESPACING 17.
