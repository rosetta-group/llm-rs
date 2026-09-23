# Linear A, round five: the Hittite profile match is a syllable-frequency artefact

Status: complete, 2026-09-23, branch `linear-a`. Protocol, sources and driver committed before the
run ([PROTOCOL.md](PROTOCOL.md), [sources.json](sources.json)); results in
[results.json](results.json). **The protocol's gate passed for Hittite; post-hoc checks show the
match does not depend on word structure, so it is not a lead.** No reading of Linear A is proposed.

## What was done

- Extracted word forms by language tag from TLHdig Beta 0.3 (`linear_a/tlhdig.py`) and spelled
  them with the Linear B rules: Hittite 22,114 types, Hurrian 6,980, Akkadian 5,004, Hattic 2,382,
  Luwian 963, Palaic 375. Greek running words from Wiktionary as before (41,194).
- Compared profiles (alternation rates, final vowels, entropy, length, vowel-initial share) with
  length-matched references, 20 samples of 300 types.
- Read Peet's 1927 transcription of the Keftiu writing board from the page image (p. 92) and
  tested the 9 names as consonant skeletons.

## Why

Round four's name-profile control worked, and TLHdig supplied the missing Anatolian and Hurrian
running text in one format.

## Results

| Check | Result | Needed |
|---|---|---|
| Linear B → Greek | 20 of 20 | ≥ 18 |
| Held-out half → own language | Hittite 20, Hurrian 20, Akkadian 20, Hattic 18 | ≥ 18 |
| Held-out half → own language, failed | Luwian 7 (13 Hurrian); Palaic 0 (20 Hurrian, 189 types only) | ≥ 18 |
| Linear A → nearest | Hittite 20 of 20; entry words Hittite 20 of 20 | ≥ 18 |
| Keftiu names (Peet) | 8 of 9 match some word; chance 7.4; p = 0.35 | exploratory |

Post-hoc checks, not in the protocol:

| Check | Linear A → nearest |
|---|---|
| Pseudo-words from Linear A's syllable bigrams | Hittite 18, Greek 2 |
| Linear A syllables shuffled across words | Hittite 20 |
| Hittite reference cut to 700 types | Hurrian 10, Greek 5, Hattic 5 |
| *o* merged into *u* in every list | Hittite 13, Greek 7 (Linear B still Greek 20) |

1. **The match survives destroying the words.** Shuffled syllables and bigram pseudo-words land on
   Hittite as reliably as real Linear A. So the profile reflects syllable and vowel frequencies,
   not stems, affixes or vocabulary.
2. **Cuneiform does not write o.** Hittite, Hurrian, Hattic and Akkadian in transliteration have
   almost no final *o*; Linear A has 3%, Greek 28%. Merging *o* into *u* drops Linear A below the
   gate, while Linear B stays Greek.
3. **List size matters.** Cutting Hittite to 700 types moves Linear A away from it, so the largest
   reference list is favoured when differences are small.
4. **Luwian and Palaic cannot be identified even from themselves** at this size, so this method
   cannot test the Luwian hypothesis.
5. **Peet's Keftiu names are consonant skeletons.** Peet rejects reading vowels from the Egyptian
   group writing, and 2–5 consonants match something in any 700-word list.

## What this does and does not establish

Linear A's syllable statistics resemble those of cuneiform-written languages, mainly because
Linear A, like cuneiform transliteration, rarely shows *o*. That is a fact about the scripts and
their conventions, not evidence that Linear A is Hittite. The protocol should have included a
shuffled-syllable negative control; it now belongs in every profile test.

## Records and reproduction

```bash
.venv/bin/python -m experiments.linear_a_tlhdig      # refuses to overwrite results.json
```

The post-hoc checks were run interactively; their numbers are recorded above.
