"""Graphs and interpretation of the frozen exploratory association extension."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/association-complex'
MODELS=('additive','interactions','nonlinear')
LABELS=('Additive','Interactions','Nonlinear')
METRICS=('root_color','profile_error','matching_rank','relational_alignment')
TITLES=('Root-color prediction','Joint visual descriptions','Same-page object matching','Relationships between objects')


def main():
    data=json.loads((OUT/'results.json').read_text());tests=data['tests'];figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,metric,title in zip(axes.flat,METRICS,TITLES):
        scale=100 if metric in ('root_color','matching_rank') else 1
        for i,model in enumerate(MODELS):
            row=tests[model+'/'+metric];lo,hi=np.array(row['interval_95'])*scale;gain=row['gain']*scale
            ax.plot([lo,hi],[i,i],color='#276eab',lw=2);ax.scatter(gain,i,color='#276eab',s=45,zorder=3)
        ax.axvline(0,color='#75828a',ls='--',lw=1);ax.set_yticks(range(3),LABELS);ax.invert_yaxis()
        units='percentage points' if scale==100 else ('standardized error reduction' if metric=='profile_error' else 'correlation gain')
        ax.set(title=title,xlabel='Improvement over controls ('+units+')')
    fig.suptitle('Complex associations: no reliable improvement\nBars are descriptive 95% folio-bootstrap intervals; all 12 Holm p-values = 1.0',fontsize=12)
    for suffix in ('png','svg'):fig.savefig(figures/f'complex-gains.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    mentions=data['audit']['descriptor_mentions'];selected=['root','leaf','flower','stem','root_light','root_dark','leaf_dark','flower_dark','stem_light','large','striped','hairy','triangular','split_roots']
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    axes[0].barh(selected[::-1],[mentions[k] for k in selected[::-1]],color='#276eab')
    axes[0].set(title='What the descriptions actually cover',xlabel='Explicit mentions among 123 objects')
    scores=data['sanity']['balanced_accuracy'];names=['control']+list(MODELS)
    axes[1].bar(['Controls']+list(LABELS),[100*scores[n] for n in names],color=['#75828a','#75828a','#db7840','#276eab'])
    axes[1].set(title='Planted interaction: implementation check',ylabel='Held-out balanced accuracy (%)',ylim=(0,110))
    axes[1].tick_params(axis='x',labelrotation=15)
    for i,n in enumerate(names):axes[1].text(i,100*scores[n]+1,f'{100*scores[n]:.0f}%',ha='center')
    for suffix in ('png','svg'):fig.savefig(figures/f'coverage-and-control.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    table=[]
    for model,label in zip(MODELS,LABELS):
        vals=[tests[model+'/'+m] for m in METRICS]
        table.append(f"| {label} | {100*vals[0]['score']:.2f}% | {vals[1]['score']:.4f} | {100*vals[2]['score']:.2f}% | {vals[3]['score']:.4f} |")
    statistical=[]
    for metric,title in zip(METRICS,TITLES):
        for model,label in zip(MODELS,LABELS):
            row=tests[model+'/'+metric];scale=100 if metric in ('root_color','matching_rank') else 1
            lo,hi=np.array(row['interval_95'])*scale
            statistical.append(f"| {title} | {label} | {row['gain']*scale:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['p']:.3f} | {row['holm_p']:.3f} |")
    report='''# Beyond root color: complex text–image associations

This extends the image-description pilot to joint features, nonlinear models, object
matching, and relationships between objects. None of the 12 planned comparisons
establishes a text–image association on this dataset.

**Joint profile:** a vector of explicit visual mentions and their pairwise combinations;
for example, light roots together with dark leaves. Zero means not mentioned, not absent.
**Interaction model:** one that can use combinations of text features whose individual
contributions are insufficient; the synthetic check plants exactly this situation.
**Same-page matching:** ranking the correct visual description among alternatives on
that held-out page; ties receive half credit and chance normalized rank is 50%.
**Holm correction:** adjusts the 12 planned tests together, so trying more models or
outcomes does not create an easy route to a positive claim.

```text
Preserve the original single-endpoint result
Audit richer descriptions without scoring text associations
Freeze vocabulary, models, grouped evaluation, and 12 tests
Hold out an entire folio and fit only on the remaining folios
Predict joint profiles and compare same-page matches and object relationships
Shuffle complete annotation profiles within pages 999 times and refit
Correct all 12 tests; preserve every result
```

## What was done

- Expanded from 59 root-color examples to **123 whole-plant descriptions** on six folios;
  121 objects have at least two alternative descriptions on the same page.
- Encoded 35 base descriptors and all 595 pairwise co-mentions. Depending on the held-out
  folio, 58–70 dimensions have sufficient training support to enter the joint-profile model.
- Compared additive, pair-interaction, and nonlinear radial text kernels against nonlinear
  length/layout controls, with all sides and panels of each folio held out together.
- Tested root color, joint-description prediction, same-page matching, and relational
  alignment. No model or metric was selected after seeing the outcomes.
- Checked implementation on a planted interaction: nonlinear models recover it, while
  the additive model and controls remain at chance.

## Why it was done

A single visual attribute can miss a relationship carried by combinations of plant
parts or text features. Joint prediction and within-page matching test those richer
possibilities while reducing the opportunity to exploit page identity or layout.

## Results

![Gains and uncertainty](figures/complex-gains.png)

All values below use held-out physical folios. Profile error is lower-is-better;
the other three scores are higher-is-better. The control model is stronger than the
previous pilot's linear baseline, so its score is different.

| Model | Root-color balanced accuracy | Joint-profile error | Same-page matching rank | Relational alignment |
|---|---:|---:|---:|---:|
| Length/layout controls | 61.28% | 0.9565 | 54.65% | 0.0140 |
'''+ '\n'.join(table)+'''

1. **Joint visual features did not improve recovery.** All three text models increase
   profile error relative to controls. These targets include combinations such as root
   color plus leaf color, size plus part, and multiple co-mentioned structures. They
   predict descriptions, not biological presence/absence or translated words.
2. **Same-page matching did not improve.** Text models score 52.28–53.85% normalized
   rank versus 54.65% for controls. These are rank scores, not percentages of objects
   identified exactly. Keeping candidates on the same page removes a simple route to
   success through section, scribe, or page differences.
3. **Relational alignment had the largest apparent improvement.** The interaction
   model improves correlation by 0.0719, with interval [−0.0821, +0.2466] and raw p=0.153.
   This interval crosses zero; even before correcting for multiple comparisons, it does
   not meet the declared gate. All twelve corrected p-values are 1.000.
4. **The nonlinear implementation check passed.** On 24 synthetic observations in six
   held-out groups, the interaction and radial models reach 100% balanced accuracy,
   p=0.001 each. Additive and control models score 50%. This demonstrates detection of
   that planted pattern; it does not establish adequate power for every Voynich relation.

![Coverage and planted control](figures/coverage-and-control.png)

The negative outcome covers a wider set of tests than the original root-color pilot.
It does not rule out more complex relations in the manuscript. The data barely represent
some desired features: triangular shape has one clear mention, hairy/fuzzy two, split
roots one, and stripes five. A larger model cannot supply the missing annotations.

## Full comparison record

Gains for root color and matching are percentage points. Profile gains are standardized
mean-square-error reductions; relational gains are correlations. Intervals are paired
2,000-draw folio bootstraps, descriptive rather than simultaneous confidence intervals.
Permutation tests move entire profiles together, preserving co-mentions and page totals.

| Endpoint | Model | Gain over controls | 95% folio interval | Raw p | Holm p |
|---|---|---:|---|---:|---:|
'''+ '\n'.join(statistical)+f'''

## Method and limits

1. **Visual representation.** Fixed botanical vocabulary over clear clauses in existing
   descriptions, with uncertain/editorial clauses excluded. Pairwise co-mentions are
   targets, not independently observed new objects. At least three mentions and three
   non-mentions in the training fold are required for a dimension. This grammar is a
   limited annotation parser, not a new independent visual assessment. Some descriptions
   omit parts or use wording it does not recognize.
2. **Text and controls.** EVA character 1–4-grams plus skip-bigrams, hashed to 512 bins.
   Controls include label length, word count, relative label index, location group,
   and transcriber. All objects share section and inherited hand label 1. Control
   scaling, radial bandwidths, target support, and target scaling fit training folios
   only. Ridge penalty 1 and every model choice were frozen before scoring.
3. **Inference.** Six physical folios remain a small sample; folio 99 has only two
   eligible plants and no same-page matching trial. Within-page permutations preserve
   the page composition, not every possible confound. Profile support/scales are
   invariant under these permutations because entire pages belong to one fold; cached
   operators therefore refit the specified ridge model exactly. Repeated objects and
   duplicated descriptions are not independent evidence; rank ties get half credit.
4. **Scope.** This is an exploratory reuse of a previously examined catalogue, not a
   new independent confirmation sample. The original annotators could see the writing.
   Targets are human descriptions of images, **not raw-image features**. No species,
   medical use, word translation, or direction of causation has been established.

The full analysis took {data['seconds']:.2f} seconds on CPU. No GPU rental, model download,
or additional prediction/BPC experiment was used. The original Voynich final test
remains sealed. Original pilot and recovery protocols/results are unchanged.

## Next evidence to collect

The useful next extension is a larger **blinded visual-feature dataset**, rather than
more fits to these same descriptions. Annotate consistent image regions with writing
hidden: root branching/count, leaf arrangement and shape, flower structure, texture,
color distribution, and spatial relations. Distinguish present, absent, uncertain, and
unobservable. Use two independent annotators, record agreement, and reserve whole folios
before fitting. A raw-image embedding can then be another declared representation,
with text masking and same-page retrieval controls to prevent reading the glyphs.
This annotation/pixel study has not yet been performed.

## Audit and reproduction

- [Frozen protocol](PROTOCOL.md), [code/source hashes and coverage audit](freeze.json),
  [all metrics, null draws, and per-folio results](results.json), [verification](verification.json).
- Code: `voynich/association_complex.py`, `experiments/association_complex.py`;
  report renderer: `experiments/association_complex_report.py`.
- Source: [Grove/Stolfi 1998 catalogue](https://www.ic.unicamp.br/~stolfi/EXPORT/voynich/98-02-01-lotsa-labels/).
  Raw annotations remain local; no explicit source reuse license was found. Aggregate
  results and source attribution are stored here.
- Run order in a fresh isolated output archive: `python -m experiments.association_complex freeze`,
  then `run`. Existing output is protected against overwrite. `verify` checks all frozen
  hashes; `python -m experiments.association_complex_report` regenerates these graphs
  from saved scores. Original local predictions remain in `artifacts/association-complex/`.
'''
    (OUT/'REPORT.md').write_text(report)
    print('Wrote complex association report and two PNG/SVG figures')


if __name__=='__main__':main()
