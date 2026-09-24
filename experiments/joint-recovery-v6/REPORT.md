# Round six: context reparse halves Naibbe letter error on sealed 20,800-letter text (1.78%); gate not met

Evaluated 2026-09-24. Six sealed ciphertext-only cases, CPU, about 80 minutes each in six parallel
processes. No cap was hit. No Voynich text. Protocol and code are in `9f87b88`, freeze in `f73032a`,
and the ParTUT train pin in `95c70ae`, all before any passage existed.

**Commit order:** a failing unit test delayed the protocol commit, so the freeze was committed a
moment before it. The freeze hashes those exact files, both commits precede the download and the
passages, and `verify` checks the hashes.

## Arms (paired, shared stages up to refinement)

- **S:** round four's decoder with square-root-scaled thresholds, then polish and v3 segmentation.
- **R:** S's refined key, then two rounds of full-context reparse with key refit, then polish and v3 segmentation.

## Results ([results.json](results.json))

| Pooled | S: CER | S: WER | **R: CER** | **R: WER** |
|---|---:|---:|---:|---:|
| All 6 cases | 3.88% | 31.39% | **1.78%** | **22.74%** |
| Dante (2) | 4.05% | 35.09% | 2.07% | 28.50% |
| Compagni (2) | 3.65% | 32.84% | 1.60% | 24.84% |
| Modern ParTUT (2) | 3.93% | 24.53% | 1.69% | 12.26% |

Per case, R: Dante 1.69% / 27.8% and 2.45% / 29.3%; Compagni 1.52% / 23.7% and 1.67% / 26.0%;
modern 1.99% / 15.3% and **1.39% / 9.3%**. Every case improves on letters and words. Parse agreement
rises from 93.2–95.1% to 94.6–96.6%. The first reparse round changes 206–346 parses, the second 10–36.

- **Primary endpoint: passed.** R is 2.10 points below S; the threshold was 0.5.
- **Gate: 0 of 6.** One modern case is under 10% WER (9.3%), but its CER is 1.39%, above 1%.
- **Development held up on fresh text.** R's 1.78% matches its development mean of 1.79%. S did worse
  than on development, 3.88% against 3.25%, so the paired gain grew.

## Progress on this benchmark

| Round | Letters per case | CER | WER |
|---|---:|---:|---:|
| Round one | 5,200 | 33.5% (Dante) | 87% |
| Round four | 5,200 | 5.7% | 45% |
| Round five (v3 segmentation) | 5,200 | 5.6% | 41.5% |
| **Round six (reparse, sqrt lexicon)** | **20,800** | **1.78%** | **22.7%** |

Cases got longer, so round six does not compare directly with the 5,200-letter rounds. Its paired
arms isolate the method's effect at fixed length.

## Remaining limits

- **Letters.** 1.4–2.5% CER remains. The lexicon still holds about 180–200 spurious pieces at this
  length, and the reparse can only choose among readings the lexicon offers.
- **Words.** At 1.8% CER, word error is 23% pooled. Historical text carries most of the rest: its
  perfect-letter segmentation floor is 17–20%.
- **Scope.** Six cases from three sources. Naibbe's structure is known to the method family, not
  supplied to it. This does not show that the Voynich text is Naibbe-class.

Dante sentences, Compagni paragraphs and ParTUT train sentences used here are released and are
excluded from future tests.
