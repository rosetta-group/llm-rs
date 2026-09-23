# Linear A, round five: Anatolian and Hurrian running text, and Peet's Keftiu names

Status: written before any result, 2026-09-23, branch `linear-a`. CPU only. Scope:
[SCOPE.md](../linear-a-next/SCOPE.md).

## Data

- **TLHdig Beta 0.3** (Zenodo 20328284, CC BY 4.0), downloaded 2026-09-23 with the owner's
  approval. Word forms by language tag (`linear_a/tlhdig.py`): Hittite, Luwian, Palaic, Hurrian,
  Hattic, Akkadian. Forms with logograms, digits or damage marks are dropped. Each form is spelled
  with the round-one Linear B rules.
- **Greek running words:** round four's Wiktionary inflected forms.
- **Peet 1927**, pp. 90–99 (archive.org scan, public domain), read from the page image of p. 92.
  Keftiu names as Peet transcribes them, consonants only: ꜣšḥr, Nsy, ꜣkš, ꜣkšt, ꜣdm, Pnrt, Rs,
  Bndbr, ỉknw. Sn-nfr and Snt-nfrt are Egyptian and excluded.

## A. Length-matched grammar profiles (gated)

Round four's profile features (`linear_a/probes.profile`). For each target sample of 300 word
types, every reference sample is drawn to match the target's syllable-length distribution
(lengths above 6 pooled). 20 samples. Nearest reference by Euclidean distance on features
standardised over all samples.

```text
references = Greek, Hittite, Luwian, Palaic, Hurrian, Hattic, Akkadian
control 1: held-out half of each TLHdig language (split by hash) -> own language, >= 18 of 20
control 2: Linear B words -> Greek, >= 18 of 20
if both pass: Linear A words -> report nearest counts; a lead needs >= 18 of 20
```

A language whose own held-out half fails control 1 is reported but cannot be a lead.

## B. Keftiu names (exploratory, no control)

Probe 1's consonant-skeleton rule: initial ꜣ or ỉ dropped as a vowel carrier, ḥ dropped, š → s,
b → p, t ~ d; s, m, n, r may drop from clusters. Count names with at least one Linear A match
against 200 null corpora from Linear A's syllable model; one-sided p.

## Not done, and why

Frequent-word matching was scoped but is left out: there is no Greek frequency list pinned, so it
would have no Linear B control, and rounds one to three showed uncontrolled matching misleads.
