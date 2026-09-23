# Scope: Rongorongo

Status: scoping only, 2026-09-23, branch `rongorongo`. Nothing is downloaded yet. Sources and
risks are in [BACKGROUND.md](BACKGROUND.md) (AI-compiled; items tagged [U] unverified).

## Why Rongorongo, and what changed on scoping

It is the one undeciphered script where the language is almost certainly known (Old Rapa Nui,
Polynesian) and the corpus is not tiny ([UNDECIPHERED.md](../../docs/UNDECIPHERED.md)). That is
the Naibbe setup: language given, key unknown.

Scoping found that the planned known-answer control has already been run and failed.
[Rochala 2026](https://github.com/rochal/rongorongo) (section 23), as reported there:

| Measure | Value |
|---|---:|
| Adjacent pairs among the 40 commonest signs, one copy per parallel family | about 2,600 |
| Real Rapa Nui text disguised as signs, recovered under a 6,000-syllable Rapa Nui model | 5% |
| Same, under 5.4 M syllables of Māori and Tahitian | 20% |
| Held-out Māori text: syllables needed for full recovery | about 8,000 (62% at 4,000) |

It also finds that whole glyphs behave like a word vocabulary, not a syllable stream: 649 head
sign types, of which the 55 commonest cover only 62% of tokens.

## Data

| Need | Source | Size | Licence |
|---|---|---:|---|
| Glyph transcription | CEIPP numerical transliteration (Barthel numbers), [kohaumotu.org](http://kohaumotu.org/rongorongo_org/translit/) | 26 objects, about 15,100 glyph tokens; 8,700 with one copy per parallel family | free copying for non-profit use with acknowledgement |
| Language model (proxy) | Māori 1868 Bible and Grey's collections, archive.org scans | about 2.8 M syllables | public domain |
| Old Rapa Nui | Thomson 1891 (archive.org), Métraux 1940 (HathiTrust) | about 6,000 syllables | public domain; needs OCR |
| Modern Rapa Nui examples | Kieviet 2017 grammar, LaTeX source on GitHub (langsci/124) | 2,000+ glossed examples | CC BY 4.0 |

## What we could add

Our solver differs from Rochala's annealing search. It is a joint HMM with latent segmentation
and variable-length units, which recovered Naibbe at 5,200 letters when its piece list was
right. So an independent check of the bound is worth one round, control first:

```text
download CEIPP pages, the Māori corpus, Thomson 1891; pin hashes
build a Māori syllable model; hold out passages
control: disguise held-out Māori and Rapa Nui text with a random one-to-one sign key
         at 2,600, 5,200 and 8,000 syllables; also with 20% extra homophone signs
gate: at the tablets' size (2,600) recover >= 90% of syllables in 4 of 4 passages
only if the gate passes: run on the CEIPP frequent-sign stream
```

Expected outcome, stated before running: the control fails at 2,600, as Rochala's did. The
result would then be an independent confirmation that the tablets are too short for
statistical sign-to-syllable assignment, with the length at which our solver succeeds.

## Risks

1. **Too little text**, per the table above, and the corpus cannot grow.
2. **Probably not a pure syllabary** (Davletshin 2022; Rochala 2026). A one-sign-one-syllable
   model may be the wrong model.
3. **Inventory choice drives statistics.** Barthel, Horley (about 130) and Pozdniakov (52)
   catalogues give different pictures; any result must be repeated across them.
4. **Cultural sensitivity.** None of the tablets is on Rapa Nui, and the community seeks the
   return of its heritage. Any output is a method test, not a reading, and should not be
   published as one without consultation.
