"""Render the frozen standard-method comparison without choosing new models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/standard-decipherment'


def main():
    results=json.loads((OUT/'results.json').read_text())
    development=json.loads((OUT/'development.json').read_text())
    summary=results['summary']
    audit=json.loads((OUT/'verification.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    colors=['#a84e44','#377fa1','#358067','#9c73af']
    families=['substitution','homophonic','variable-homophonic','naibbe']
    labels=['Substitution','One-letter\nhomophonic','One/two-letter\nhomophonic','Naibbe']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,dataset in zip(axes,['modern','historical']):
        for j,method in enumerate(['legacy_mean','legacy_mdl','combined']):
            values=[summary[f][dataset][method]['cer']*100 for f in families]
            pos=np.arange(4)+(j-1)*.25
            bars=ax.bar(pos,values,.23,color=colors[j],label=['Old mean selection','MDL, same candidates','MDL + published beam'][j])
            for b,value in zip(bars,values):
                ax.annotate(f'{value:.1f}',(b.get_x()+b.get_width()/2,value),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=7,rotation=90)
        ax.set_xticks(range(4),labels)
        ax.set_title('Modern Italian' if dataset=='modern' else 'Historical Italian (Dante)')
        ax.set_yscale('symlog',linthresh=1)
        ax.set_yticks([0,1,10,100,1000],['0','1','10','100','1,000'])
        ax.set_ylim(0,2000)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Character error (%) — linear below 1%, log above')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9,frameon=False)
    fig.suptitle('Fresh controls: correcting selection is not the same as solving a cipher')
    fig.tight_layout(rect=[0,.10,1,.95]);fig.savefig(OUT/'recovery.png');plt.close(fig)

    segmentation={}
    for dataset in ('modern','historical'):
        subset=[r for r in results['segmentation'] if r['dataset']==dataset]
        words=sum(r['words'] for r in subset)
        segmentation[dataset]={name:sum(r[name+'_errors'] for r in subset)/words for name in ('old','new')}
    fig,ax=plt.subplots(figsize=(7.5,4))
    for j,name in enumerate(('old','new')):
        bars=ax.bar(np.arange(2)+(j-.5)*.3,[segmentation[d][name]*100 for d in segmentation],.28,color=colors[j],label=['Modern prior','Modern + historical prose'][j])
        ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.set_xticks([0,1],['Modern Italian','Historical Italian (Dante)']);ax.set_ylabel('Word error (%)')
    ax.set_ylim(0,max(v for r in segmentation.values() for v in r.values())*125+2)
    ax.set_title('Segmentation control: exact letters supplied, boundaries hidden')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'segmentation.png');plt.close(fig)

    comparator=[]
    for family in ('substitution','homophonic'):
        for dataset in ('modern','historical'):
            cases=[r for r in results['cases'] if r['family']==family and r['dataset']==dataset]
            beam=[]
            for case in cases:
                candidates=[c for c in case['candidates'] if c['method']=='published-beam']
                if candidates:
                    selected=min(candidates,key=lambda c:c['mdl']['total_bits'])
                    beam.append((selected['cer'],case['characters']))
            comparator.append(dict(family=family,dataset=dataset,
                beam_cer=sum(c*n for c,n in beam)/sum(n for c,n in beam) if beam else None,
                beam_cases=len(beam),em=summary[family][dataset].get('hmm_em')))
    fig,ax=plt.subplots(figsize=(9,4))
    names=[r['family']+'\n'+r['dataset'] for r in comparator]
    for j,key in enumerate(('beam','em')):
        values=[100*(r['beam_cer'] if key=='beam' else r['em']['cer']) for r in comparator]
        bars=ax.bar(np.arange(4)+(j-.5)*.32,values,.3,color=colors[j+1],label=['Nuhn beam + MDL selection','Trigram HMM, bounded EM'][j])
        ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set_xticks(range(4),names);ax.set_ylabel('Character error (%)');ax.set_ylim(0,max(2,1.25*max(100*max(r['beam_cer'],r['em']['cer']) for r in comparator)))
    ax.set_title('Published model classes: fresh controls they can express')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'comparators.png');plt.close(fig)

    rows=[]
    for family in families:
        for dataset in ('modern','historical'):
            m=summary[family][dataset]
            rows.append(f"| {family} | {dataset} | {100*m['legacy_mean']['cer']:.2f}% | {100*m['legacy_mdl']['cer']:.2f}% | {100*m['combined']['cer']:.2f}% | {100*m['combined']['wer']:.2f}% | {m['combined']['gate_passes']}/2 |")
    segment_rows=[f"| {d} | {100*r['old']:.2f}% | {100*r['new']:.2f}% |" for d,r in segmentation.items()]
    method_rows=[f"| {r['family']} | {r['dataset']} | {100*r['beam_cer']:.2f}% ({r['beam_cases']}/2) | {100*r['em']['cer']:.2f}% ({r['em']['cases']}/2) |" for r in comparator]
    controls=[r for r in results['cases'] if r['family'] in ('substitution','homophonic')]
    selected_exact=sum(r['methods']['combined']['cer']==0 for r in controls)
    selection_losses=[r for r in results['cases'] if r['oracle_cer']<r['methods']['combined']['cer']]
    beam_caps=sum(r['status']=='budget_exhausted' for c in results['cases'] for r in c['beam_runs'])
    em_runs=[c['em'] for c in results['cases'] if 'restarts' in c['em']]
    md=f'''# Length-aware selection and standard decipherment comparators

Frozen method commit: `{results['challenge']['freeze_commit']}`. Earlier archive: `28ceb16`.
[Protocol](PROTOCOL.md) · [development](development.json) · [fresh results](results.json) · [sources](sources.json) · [audit](verification.json).

## What was done

- Replaced mean-score candidate selection with total description length, including key and ambiguity costs.
- Implemented Nuhn-style key beam search and a bounded trigram-HMM EM comparator.
- Added Novellino and Decameron training/development prose; evaluated fresh modern and Dante passages.
- Kept the image track parked and the Voynich final test sealed. No paid compute.

## Why

The old decoder sometimes generated the exact letters and then discarded them for a longer
wrong answer. Fixing that defect and adding established search models makes the negative
benchmark more informative; it does not establish that Voynich is a cipher or recover meanings.

## Fresh result

The combined selector recovers exact letters in **{selected_exact}/8** fresh substitution and
one-letter homophonic controls. The full Naibbe recovery gate is **{'passed' if results['naibbe_gate'] else 'failed'}**.
No Voynich mechanism experiment was run. All eight substitution/homophonic cases meet
the **letter** threshold; historical word recovery still fails. Historical substitution
CER falls from **145.82% to 0%** by reranking the same candidates. Historical segmentation
WER falls from **37.67% to 24.20%** on identical fresh passages; the previous 39.9% result
was measured on different passages and is not the paired baseline.

**CER:** character insertions, deletions and substitutions divided by reference letters.
**WER:** the same edit count over words. Either can exceed 100% when output expands.
**MDL:** total bits for the proposed plaintext, key, cipher choices and required layout.
**Gate:** CER ≤1% and WER ≤10% on each case; four source passages are the independent samples.

![Selection and recovery](recovery.png)

| Cipher | Source | Old mean CER | Same candidates, MDL CER | + beam CER | + beam WER | Gate |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

All columns here use the new common prior and segmenter. The first two compare selectors
on exactly the same candidate set; the third adds beam candidates. The old algorithm's
search is retained only as an ablation and as the one/two-letter candidate source.
MDL is the selection criterion; it is not the objective of the archived annealer or HMM.
The character prior changes between those algorithms as documented, so this is not a
pure search-only causal comparison.

## What the score repairs

```text
Generate a candidate plaintext and its key
Charge every plaintext letter under a normalized character prior
Charge key entries and cipher-symbol inventory
Charge choices between homophones and ambiguous chunk boundaries
Choose the complete candidate with the fewest total bits
```

The four new development substitution cases all selected exact letters with MDL; mean
selection failed one. In that case the exact candidate cost **2,835 bits**, versus **6,443 bits**
for the expansion, which had **144.8% CER**. Development examples are not held-out evidence.
On fresh grading, **{len(selection_losses)} cases** still have a better candidate by oracle CER
than the MDL-selected candidate. MDL encodes the stated model; it does not guarantee truth.
One historical homophonic case still prefers an eight-error candidate (**7,173 bits**)
over the exact candidate (**7,185 bits**), both 1,265 letters long. That is prior/ranking
error rather than the old free-expansion defect. Inventory/key costs especially matter
for large, weakly shared variable-length codebooks.
The criterion is an ideal arithmetic/enumerative code length, not an implemented compressor.

## Published model comparison

![Published comparators](comparators.png)

| Cipher | Source | Beam-only MDL CER (completed cases) | HMM CER (run cases) |
|---|---|---:|---:|
{chr(10).join(method_rows)}

The beam implementation follows [Nuhn et al. 2013](https://aclanthology.org/P13-1154/)
and the partial-context and extension-order improvements of
[Nuhn et al. 2014](https://aclanthology.org/D14-1184/). Development selected width
**{development['selected_width']}**: all eight development encodings had zero CER, and 8192
added no gain. Our Italian order-5 prior, adapted order weights, and absence of sentence
boundary symbols differ from the published English experiments. Above 64 cipher types,
extension order falls back to frequency. This is an independent implementation, not an
UNRAVEL run or a replication of published Zodiac accuracy.

The HMM follows the model in [Berg-Kirkpatrick & Klein 2013](https://aclanthology.org/D13-1087/):
fixed trigram transitions, learned emissions and posterior decoding. It uses 200 EM iterations,
.1 emission smoothing and up to eight random restarts. The actual range was
**{min(r['restarts'] for r in em_runs)}–{max(r['restarts'] for r in em_runs)} restarts**; the paper's
large-restart result is not reproduced. Its character priors and smoothing differ too.
HMM letters may vary across occurrences of one cipher symbol, so this output is graded
separately rather than forced into a deterministic key for MDL selection.

**Model-class limit:** both published methods emit one letter per observed symbol. The genuine
homophonic controls fit that assumption. Naibbe and the variable-length controls do not,
in general. Their failure is an out-of-class stress result; the generic expansion search
also remains bounded. Neither result rules out all variable-length cipher methods.

## Historical segmentation, with letters supplied

![Word segmentation](segmentation.png)

| Source | Frozen modern segmenter WER | Added historical prose WER |
|---|---:|---:|
{chr(10).join(segment_rows)}

These are the same four fresh passages, stripped of spaces before the segmenters see them.
They isolate word boundaries from cipher recovery. Novellino and the first two Decameron
days contribute **62,756 training words**, **13,668 development words**, **8,436 historical
word forms**, and **2,995 forms** absent from the original lexicon. Modern ISDT contributes
216,579 training words. Whole tales are split; every fifth tale is development.
No Dante is used for fitting or tuning. The selected segmentation model weights historical
training counts 16×, alpha 1, bigram weight .5. Development WER was 10.26% across 12 modern sentences and 24 historical 80-word windows.
This changes word counts and transitions as well as vocabulary; it is not a lexicon-only test.

Dante is a different author from the historical training sources, but the project had already
studied other Dante passages. The new cases are fresh source-ID holdouts, not a previously
unknown author benchmark. Edited orthography, tokenization and poetry/prose differences
remain limitations. Wikisource pages have varying proofreading status; provenance is pinned.

## Audit and limits

- Method, protocol, sources and development results were committed before challenge generation.
- 16 opaque encodings derive from four passages: two modern, two historical; each has 1,200–1,800 letters.
- Earlier benchmark source IDs and exact modern train/dev sentences were excluded; encoder round trips passed.
- Decoder input was ciphertext and unpaired priors. Predictions were saved before opening evaluation answers.
- Exact-letter segmentation controls ran only after cipher predictions were frozen.
- This is procedural blinding on one machine, not an independent evaluator or inaccessible secret store.
- Fresh decoding took **{audit['decoding_wall_seconds']/60:.1f} minutes** on local CPU, excluding setup, development and grading. No paid compute.
- **{beam_caps} beam runs** hit their CPU cap. Capacity skips and HMM inventory/restart limits are in results.json.
- Seeds, hashes, per-case candidates and selector losses are recorded. No significance claim from four passages.
- Exact challenge, predictions and evaluator references are released in `evaluated-records.tar.gz` only after grading. These passages are now disclosed; exclude them from all future fresh tests.
- The 1.8 MB committed source snapshot preserves the exact transcluded Wiki text, not only top-level page revisions.
- All prior frozen studies are unchanged. No fourth image study, BPC sweep, GPU rental, or final-test scoring occurred.

## Decision

{'The Naibbe gate passed; a separately committed multilingual shuffle protocol is required before the reserved mechanism run.' if results['naibbe_gate'] else 'Keep the Voynich mechanism test closed: codebook-free Naibbe has not met the letter-and-word gate.'}
The contribution is a controlled method benchmark, failure decomposition and reproducible
negative evidence. A translation remains out of reach. The standard-method implementations
add a missing comparator, but four passages, edited historical data, and bounded HMM restarts
do not by themselves establish publication readiness.
'''
    (OUT/'REPORT.md').write_text(md)
    (OUT/'comparison-summary.json').write_text(json.dumps(dict(comparators=comparator,segmentation=segmentation,selection_losses=[r['id'] for r in selection_losses]),indent=2)+'\n')
    print('Report and three figures written')


if __name__=='__main__': main()
