# Proto-Elamite, round three: which object signs go with which number system?

Status: written before any code or result, 2026-09-24, branch `proto-elamite`. CPU only.
Round two ([report](../proto-elamite-joint/REPORT.md)) recovered N14 = 10 N01 (counting) and
N14 = 6 N01 (capacity) blind. This round fixes those two ratios and asks whether the object signs
on each tablet match the commodity the literature assigns to each system. Same data and parser.

## Labels from arithmetic only

For each usable tablet (round one's definition), `d = entry sums − total`. A tablet is labelled
when `d` is non-zero only on N01 and N14 (other numeral signs, if any, already balance):

```text
capacity  if d(N01) + 6  d(N14) = 0 and d(N01) + 10 d(N14) != 0
counting  if d(N01) + 10 d(N14) = 0 and d(N01) + 6  d(N14) != 0
otherwise unlabelled
```

No sign other than N01 and N14 is used for the label, and no value other than 6 and 10.

## Object signs

Every non-numeral token on the tablet, with damage marks and variant suffixes (`~a`, `@t`)
removed and compounds (`|A.B|`, `|A+B|`, `|A×B|`) split into their parts. A tablet "has" a group
if any of its signs is in the group.

## Tests (one-sided Fisher exact, share of labelled tablets that have the group)

| # | Corpus | Group | Prediction | Role |
|---|---|---|---|---|
| G | proto-cuneiform | any sign starting ŠE (barley) | more on capacity tablets | **gate**, p < 0.05 |
| 1 | Proto-Elamite | grain: M288, M036, M297 | more on capacity tablets | test |
| 2 | Proto-Elamite | animals and people: M388, M124, M346, M367, M006, M362, M376 | more on counting tablets | test |

Tests 1 and 2 run only if the gate passes; each passes at p < 0.025 (0.05 / 2). Groups come from
the literature summary in [BACKGROUND.md](../proto-elamite/BACKGROUND.md) §1.5 and §4, fixed now.
Also reported, exploratory: every sign on at least 3 labelled tablets, with its counts by label.

## What would count

Passing tests 1 and 2 would show, from arithmetic alone, that the tablets which use the capacity
ratio are the ones that mention the proposed grain signs. That supports those sign readings
independently of how they were first proposed. It would not identify any word.

## Records

`experiments/proto_elamite_round_three.py`; results in `experiments/proto-elamite-signs/results.json`.
