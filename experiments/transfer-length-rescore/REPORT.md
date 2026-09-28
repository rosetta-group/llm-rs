# Transfer length code: a scoring bug, and what fixing it does

Post-hoc development analysis of all released rounds. Sealed outcomes stay as recorded.
Code: `voynich/rejection_v3.py`, `experiments/transfer_length_rescore.py`; tests: `tests/test_rejection_v3.py`.

**Transfer excess:** decoded bits per letter on the transfer passage with the sealed fit key, minus
the model's calibration score; the frozen rule accepts at most 0.50.

## The bug

`voynich/rejection.py` scores each run between unreadable tokens with its own `CharacterPrior.bits`
call, and every call adds an Elias-gamma length code. A transfer passage has a median 59 runs, so it
pays 59 length codes, about 0.13 bits per letter, while the calibration score pays one. Found by an
independent code review; confirmed: scoring one text in two calls costs 14.4 extra bits. The archived
runs can be rebuilt exactly (0 difference on 80 records), so rescoring needs no new decode.

`voynich/rejection_v3.py` keeps the context reset at each gap but charges one length code per passage.

## Rescoring the released rounds (decoder A)

| Round | Correct, per-run → one code | False acceptances with one code | Lowest omitted-language excess |
|---|---|---|---|
| Round one (narrow priors) | 13 → 14 | 1 wrong language + 1 omitted | 0.56 → 0.47 |
| v2 development | 19 → 20 | 1 omitted | 0.56 → 0.47 |
| v2 confirmation | 16 → 20 | 0 | 0.73 → 0.57 |

Every false acceptance is the same case: round-one Catalan block 2 (Desclot, Muntaner), accepted as
Occitan at 0.469 when Catalan is omitted. Old Catalan and Old Occitan are close enough that the
Occitan prior fits this text almost as well.

## Ceiling sweep with one length code (A; correct / false per round)

| Ceiling | Round one | v2 development | v2 confirmation |
|---|---|---|---|
| 0.40 | 14 / 0 | 19 / 0 | 17 / 0 |
| 0.45 | 14 / 0 | 19 / 0 | 17 / 0 |
| 0.50 | 14 / 2 | 20 / 1 | 20 / 0 |
| 0.55 | 14 / 2 | 21 / 1 | 20 / 0 |

## What it shows

1. **The bug acted as a hidden safety margin.** Removing it moves true and closely related languages
   down together.
2. **With honest scoring, 0.45 keeps every released round clean** and gains one block on v2
   confirmation (17 against 16). The gain to 20 lies in the 0.45–0.50 band, where the
   Catalan→Occitan case also sits.
3. **Decoding cost is still the lever.** Lower cost moves true languages away from that band.

Any new rule (scoring and ceiling) must be frozen and confirmed on fresh text.
