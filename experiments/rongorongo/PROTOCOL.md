# Rongorongo, round one: how much text does sign-to-syllable recovery need?

Status: written before any code or result, 2026-09-23, branch `rongorongo`. CPU only.
Scope: [SCOPE.md](SCOPE.md). Downloads approved by the owner on 2026-09-23 and pinned in
`sources.json`.

## Question

If Rongorongo wrote Rapa Nui one sign per syllable, could the key be recovered from a text the
size of the tablets? Answered first on text with a known answer. The tablets are touched only if
the control passes.

## Data

- **Language model:** Māori, 1868 Bible (`kotepaiperatapua00barl`) and 1841 New Testament
  (`kotekawenatahou00yategoog`), archive.org OCR, public domain. Reduced to Rapa Nui shape as in
  Rochala 2026: macrons dropped, `wh` → `h`, `w` → `v`, `ng` → `ŋ`; a word is kept only if it is a
  sequence of (consonant) vowel syllables over `h k m n ŋ p r t v`. Words are joined without
  spaces, because the tablets have no word dividers.
- **Held-out passages:** Grey's *Ko nga mahinga* (1854) and *Ko nga moteatea* (1853), same
  reduction: traditional narrative and song, closer in genre to recitation than scripture. Never
  used for the model.
- **Syllables:** each vowel with its preceding consonant, if any: at most 50 types.

## Solver

Bigram hidden Markov model (Knight et al. 2006; the project's Berg-Kirkpatrick-Klein HMM with a
bigram prior, because a trigram over 50 syllables is too slow on CPU). Transitions fixed from the
Māori model (add-0.5 smoothing); emissions learned by EM, 200 iterations, 20 random restarts;
best restart by likelihood; Viterbi decoding.

## Control

```text
for N in 1,300, 2,600, 5,200, 8,000 syllables:
  for 4 contiguous held-out passages (2 narrative, 2 song):
    one-to-one: each syllable type gets a random sign number
    homophones: 20% of syllable types get a second sign, used half the time
    solve; accuracy = share of syllable tokens decoded correctly
```

**Gate:** one-to-one accuracy of at least 90% in 4 of 4 passages at N = 2,600, the tablets'
count of adjacent pairs among frequent signs (Rochala 2026). If the gate fails, the tablets are not
decoded, and the result is the length at which this solver succeeds.

Stated before running: Rochala's annealing search needed about 8,000 syllables on held-out Māori.
We expect failure at 2,600.

## Records

`experiments/rongorongo_round_one.py`; results in `experiments/rongorongo/results.json`.
