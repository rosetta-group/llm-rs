"""Report frozen word-segmentation results; no fitting or decoder changes."""
from experiments.word_segmentation_v2 import OUT, read
from experiments.word_segmentation_records import verify


def main():
    verify()
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'word-segmentation-v2'
    import matplotlib.pyplot as plt

    dev = read(OUT / 'development.json'); result = read(OUT / 'results.json')
    release = read(OUT / 'evaluated-records.json'); audit = read(OUT / 'source-audit.json')
    selected = next(r for r in dev['rows'] if r['weight'] == dev['selected_weight'])
    base = dev['rows'][0]; summary = result['summary']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained', sharey=True)
    for ax, labels, values, title in [
        (axes[0], ['Historical\nprose', 'Petrarca\nverse', 'Modern\nISDT'],
         [[r['grades'][s]['wer'] for s in ('historical', 'verse', 'modern')] for r in (base, selected)],
         'Development: used to select the model'),
        (axes[1], ['Villani\nnew author*', 'Modern VIT\nnew corpus'],
         [[summary[s][m]['wer'] for s in ('historical', 'modern')] for m in ('baseline', 'verse')],
         'Fresh passages: no further tuning')]:
        for j, (label, color) in enumerate([('Frozen prose model', '#547d98'), ('Add training verse (weight 1)', '#bc6847')]):
            bars = ax.bar([i+(j-.5)*.35 for i in range(len(labels))],
                          [v*100 for v in values[j]], .34, label=label, color=color)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        ax.axhline(10, color='#555555', ls='--', lw=1)
        ax.set_xticks(range(len(labels)), labels); ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 34); ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel('Word error (%) — lower is better')
    fig.suptitle('Perfect letters: large verse gain, small cross-author gain', fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=2, frameon=False)
    axes[1].text(.5, -.22, '*Chapter rubrics retained; see source audit. Dashed line: 10% gate.',
                 transform=axes[1].transAxes, ha='center', fontsize=8)
    fig.savefig(OUT / 'word-errors.png', dpi=160)
    fig.savefig(OUT / 'word-errors.svg', metadata={'Date': None}); plt.close(fig)
    svg = OUT / 'word-errors.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

    pct = lambda n: f'{n*100:.2f}%'
    dev_table = '\n'.join(f"| {r['weight']} | " + ' | '.join(pct(r['grades'][s]['wer'])
                         for s in ('historical', 'verse', 'modern')) + ' |' for r in dev['rows'])
    fresh_table = '\n'.join(f"| {name} | {s['baseline']['errors']} / {s['baseline']['words']} ({pct(s['baseline']['wer'])}) | "
        f"{s['verse']['errors']} / {s['verse']['words']} ({pct(s['verse']['wer'])}) | "
        f"{100*(s['baseline']['wer']-s['verse']['wer']):.2f} points | {s['verse']['gate_passes']}/4 |"
        for name, s in [('Villani', summary['historical']), ('Modern VIT', summary['modern'])])
    answers = {r['id']: r for r in release['answers']}
    cases = sorted(result['cases'], key=lambda r: (r['dataset'], answers[r['id']]['source_ids'][0]))
    case_table = '\n'.join(f"| {r['dataset']} / `{r['id']}` | {answers[r['id']]['characters']} | "
                          f"{pct(r['baseline']['wer'])} | {pct(r['verse']['wer'])} |" for r in cases)
    boundary_table = '\n'.join(f"| {dataset} / {method} | {pct(v['precision'])} | {pct(v['recall'])} | "
                              f"{pct(v['f1'])} | {v['fp']} | {v['fn']} |"
                              for dataset, s in summary.items() for method, v in s.items())
    report = f'''# Verse vocabulary: a development gain that did not transfer enough

This comparison adds training-only Petrarca words to the frozen Italian word segmenter.
The large Petrarca development gain did not meet the declared transfer threshold on Villani.

**What was done**

- Compared weights 1, 4 and 16 against the frozen prose word model on existing development text.
- Committed weight 1, extraction code and grading rules in `5c86127` before fetching fresh sources.
- Saved predictions on four Villani and four modern VIT passages before opening reference spaces.
- Archived sources, revisions, licences, predictions, references and every tried setting. CPU only.

**Why**

The earlier audit found 28.5% word error on perfect Petrarca letters. This isolates
word-space recovery from cipher errors and tests transfer to a previously unused author.

**WER:** word insertions, deletions and substitutions divided by reference word count.
**Boundary F1:** agreement on space positions; it can be high while many words are wrong.
**Transfer threshold:** at least 3 percentage points lower pooled historical WER and
at most 1 point worse modern WER, declared before fresh source preparation.

```text
Add only designated Petrarca training poems at weights 1, 4, 16
Select on existing prose, verse and modern development streams
Commit the selected model and the full fresh-evaluation pipeline
Prepare new Villani and VIT passages; hide spaces and source labels from solver
Save all paired predictions, then grade
If transfer threshold fails, retain the old baseline and stop
```

## Development selection

Each stream ends at the last complete word within 5,200 letters. The word model adds
counts, word transitions and word forms; the segmentation algorithm, beam and scoring
parameters stay unchanged. Every fifth Petrarca poem remains development and is not fitted.
There are {len(dev['training_poem_ids'])} training poems. Weight 0 is the old baseline.

| Training verse weight | Historical prose WER | Petrarca WER | Modern ISDT WER |
|---|---:|---:|---:|
{dev_table}

Weight **1** minimizes the mean historical prose/verse WER among eligible settings:
{pct(sum(base['grades'][s]['wer'] for s in ('historical','verse'))/2)} to
{pct(sum(selected['grades'][s]['wer'] for s in ('historical','verse'))/2)}.
Weight 4 helps verse slightly more but has a worse historical mean; it was not selected.
No fresh result was used to revise that decision.

![Development and fresh word error](word-errors.png)

## Fresh results: transfer threshold failed

References use normalized original surface words. All predictions preserve every letter;
character error is **zero by construction**, not a cipher-decoding achievement.
Pooled WER sums edit counts and reference words; it is not an unweighted passage mean.

| Fresh source | Baseline errors / words (WER) | Added verse errors / words (WER) | Improvement | Selected model <=10% WER |
|---|---:|---:|---:|---:|
{fresh_table}

Historical improvement is **1.02 points**, below the required **3 points**. The modern
guard passes. No historical passage passes the 10% word gate; only two modern passages
pass it. The selected model is **not promoted** to the cipher decoder.

| Source / opaque case ID | Letters | Baseline WER | Added verse WER |
|---|---:|---:|---:|
{case_table}

| Source / method | Boundary precision | Recall | F1 | Extra spaces | Missing spaces |
|---|---:|---:|---:|---:|---:|
{boundary_table}

## Source audit and protocol deviation

After grading, inspection found that Wikisource encodes chapter rubrics as ordinary
`p` elements, starting with a Roman numeral on its own line. The frozen extractor
removes HTML headings but retained these rubrics. Seven included rubrics contribute
**{audit['evaluated_heading_words']} of 4,783 historical words (2.24%)**. Chapter labels
are `unlabelled` in intermediate rows; paragraph IDs still identify exact source spans.
This violates the intended prose-only extraction. Treat Villani as a fresh paired
diagnostic with this caveat, **not a clean confirmatory prose test**. No corrected
challenge was substituted and no favourable subset was regraded.

The fixed passage packer also skips 30 short historical paragraph residuals when
the next whole paragraph would exceed 6,000 letters. Their IDs are recorded in the
challenge's `excluded_source_ids`; that field combines overlap rejections and packing
residuals. A replay of the source and overlap rules reproduces the exact passages.
This is sequential size-based selection, not random sampling or selection by score.

## What the result means

1. **Coverage helps within an author.** Held-out Petrarca poems improve from 28.49% to
   14.04% WER after fitting other Petrarca poems. This supports the value of relevant
   word data; it does not establish robust historical Italian recovery.
2. **Transfer remains weak.** Villani improves from 28.37% to 27.35% WER. Four passages
   from one author are not four independent authors, and the rubric caveat limits
   the planned prose claim. There is no confidence interval or significance claim.
3. **Spaces still fail on exact letters.** The selected model produces 737 extra and
   190 missing spaces on Villani. These are descriptive counts, not proof that one
   vocabulary change or score parameter causes the errors.
4. **Meaning recovery remains unearned.** This is word segmentation only. The latest
   Naibbe round remains 5.7% character error and about 45% word error on Dante, against
   the 1% / 10% gate. No Voynich text was scored and no translation claim is supported.

## Decision and next bounded work

Keep round four and its old word model as the recovery baseline. Stop this weight
sweep. Before another fresh evaluation, add a source-fixture test for paragraph-encoded
rubrics in a new extractor version; never rewrite the frozen extractor. Use existing
development passages to distinguish missing historical forms from known words split
incorrectly. A later candidate needs its own declared mechanism, development threshold
and committed freeze. Exclude all released source IDs from future hidden evaluation.
Do not spend another Naibbe or Voynich test passage merely on this verse-word addition.

## Provenance, licences and reproduction

- [Protocol](PROTOCOL.md), [development](development.json), [freeze](freeze.json),
  [results](results.json), [evaluated records](evaluated-records.json),
  [source audit](source-audit.json), [archive hashes](archive.json).
- Villani: [Nuova Cronica, Libro primo, revision 3734101](https://it.wikisource.org/w/index.php?oldid=3734101).
  Public-domain original; Wikisource transcription CC BY-SA, with source/history
  attribution retained. Edited transcription, not a diplomatic manuscript edition.
- Modern: [UD Italian VIT](https://github.com/UniversalDependencies/UD_Italian-VIT/tree/12fc5f682a677e87dac1923903ae954d8441f7b7),
  test split at commit `12fc5f682a677e87dac1923903ae954d8441f7b7`. Fabio Tamburini,
  Maria Simi, Cristina Bosco and UD contributors. **CC BY-NC-SA 3.0**; its README and
  licence are included in the source archive. Research redistribution is not an
  unrestricted commercial licence.
- [Source manifest](sources.json) records raw hashes and revisions;
  [source archive](fresh-sources.tar.gz) includes challenge and solver outputs.
  Exact shared 20-word strings with fitting/development text were excluded. This
  check is not proof of complete provenance independence.
- Blinding is procedural on one machine, not independent third-party evaluation.
  The solver received only opaque IDs and dense letters. References were disclosed
  only after both methods' predictions were saved.

With the existing baseline and training archives restored as in
[REPRODUCE.md](../../docs/REPRODUCE.md):

```sh
.venv/bin/python -m experiments.word_segmentation_records restore
.venv/bin/python -m experiments.word_segmentation_records verify
.venv/bin/python -m experiments.report_word_segmentation
.venv/bin/python -m unittest discover -s tests -v
```

Verification replays source extraction and passage selection, regrades saved predictions
and development selection, and checks the committed freeze. It does not refit models
or consume a new test. The original `prepare`, `solve`, `evaluate` sequence is preserved
in `experiments/word_segmentation_fresh.py`; do not rerun `prepare` on the released test.
'''
    (OUT / 'REPORT.md').write_text(report)


if __name__ == '__main__':
    main()
