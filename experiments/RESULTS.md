# Local validation results

2026-09-20. All experiments used local hardware. The final test set remains sealed.

BPC is bits per normalized transcription character; lower is better. These are prediction results, not translations.

## Baselines

| Dataset | Frequency | 3-char context | 5-char context | Layout | Copy |
|---|---:|---:|---:|---:|---:|---:|
| gc | 4.2450 | 2.7579 | 3.0577 | 2.7630 | 2.6985 |
| gc-shuffle | 4.2450 | 2.9225 | 3.2494 | 2.9360 | 2.8644 |
| gc-merged | 4.1934 | 2.7099 | 3.0013 | 2.7131 | 2.6499 |
| gc-separate | 4.1668 | 2.7010 | 2.9942 | 2.7054 | 2.6441 |
| gc-quire | 4.2299 | 2.8415 | 3.1748 | 2.8526 | 2.7815 |
| zl | 4.0762 | 2.0810 | 2.2221 | 2.0744 | 2.0691 |
| timm | 3.8757 | 2.2934 | 2.5083 | 2.2955 | 2.0909 |
| naibbe | 3.9612 | 1.8737 | 1.9566 | 1.8683 | 1.9376 |

Compare models within a row. Transcriptions, merged boundaries, and synthetic controls have different target strings.

GC copy accuracy: 44.93%; frequency accuracy: 16.68%. Local spelling and copying suffice for this gain over the frequency baseline.

## Qwen3-1.7B

400 optimizer steps per selection; seed 42; rank 8; four layers; 1,245,184 trainable parameters each.
Training batch 1, accumulation 2, learning rate 0.0001, context 64, stride 32, BF16 on MPS.
Each run sees about 0.27 epochs. The frozen model is shared; all 29 validation pages are scored.

| Layers | Frozen BPC | Adapted BPC | Gain over copy, 95% folio interval |
|---|---:|---:|---|
| outer [0, 1, 26, 27] | 4.3505 | 2.9373 | -0.2388 [-0.2873, -0.1615] |
| middle [12, 13, 14, 15] | 4.3505 | 3.1549 | -0.4563 [-0.5156, -0.3641] |
| random [0, 3, 20, 23] | 4.3505 | 2.8783 | -0.1797 [-0.2191, -0.1281] |

The best Qwen selection in this pilot is **random**. One seed cannot establish a layer-location effect.

Positive gain favors Qwen. Intervals use 2,000 paired draws over 15 validation folio groups.
They do not include variation across training seeds. Qwen token accuracy uses subwords and cannot be compared with character accuracy.

## Limits and next experiments

```text
Compare five seeds at matched training budgets
Evaluate the same adapters at contexts 16, 64, and 256
Repeat neural comparisons on shuffled text, other transcriptions, and published controls
If a gain survives those checks: freeze choices and release the test set
Otherwise: revise the representation or model before testing meaning
```

- GC uses 148 training, 29 validation, and 30 test paragraph pages. Validation has 22 A and 7 B pages; the mix is uneven.
- The quire split has only three validation quires. Its uncertainty will be weak.
- Baseline robustness is measured above. Neural robustness and the five-seed matrix are not yet run.
- Naibbe and Timm each use one published sample with chronological block splits and one omitted block between splits. This is not replication over generator seeds.
- ZL3b comes from a pinned mirror because the publisher returned HTTP 406; it was not byte-verified against the publisher.
- BF16 evaluation can vary slightly with batch shape. Outer evaluation used batch 2; middle/random use batch 1 and reuse the same frozen scores.
- Learned BPE was fitted on training pages and passed a two-step random-model smoke run. It has not been attached to pretrained embeddings.
- No image association, plaintext recovery, SAE experiment, or claimed Voynich translation has been run.

## Reproduce

```sh
.venv/bin/python -m experiments.baselines
.venv/bin/python -m experiments.results
```

The second command requires the three completed `qwen-*-c64-s42-n400` runs. Their settings and per-page scores are preserved in `results.json`.
Generate fresh training configs with `python -m voynich matrix MODEL --contexts 64 --steps 400`; set accumulation to 2 to match these pilots.
See `../README.md` for training and comparison commands and source attribution.
