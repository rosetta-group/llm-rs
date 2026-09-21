"""Render reports from frozen benchmark results; no training or evaluation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/method-benchmark'


def read(name): return json.loads((ROOT/f'experiments/{name}/results.json').read_text())


def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    seg=read('segmentation');image=read('association');cipher=read('codebook-free')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    blue='#276eab';orange='#db7840';gray='#75828a'
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,dataset,label in zip(axes,('modern','historical'),('Modern Italian: 12 fresh passages','Historical Italian: 12 fresh passages')):
        data=seg['summary'][dataset];values=[100*data[m]['word_error_rate'] for m in ('baseline','lexicon')]
        ax.bar(['Previous dictionary','Frozen lexicon'],values,color=[gray,blue],width=.6)
        ax.axhline(10,color=orange,ls='--',lw=1,label='Declared 10% gate')
        ax.set(title=label,ylabel='Word edit errors / reference words (%)',ylim=(0,70))
        for i,v in enumerate(values): ax.text(i,v+1,f'{v:.1f}%',ha='center')
        ax.legend(frameon=False,loc='upper right')
    save(fig,'segmentation')
    summary=image['summary'];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    vals=[100*summary[k] for k in ('control_balanced_accuracy','text_balanced_accuracy')]
    axes[0].bar(['Length/layout controls','Controls + EVA text'],vals,color=[gray,blue],width=.6)
    axes[0].axhline(50,ls='--',color=orange,lw=1);axes[0].set(ylim=(0,100),ylabel='Held-out balanced accuracy (%)',title='59 labels, six physical folios')
    for i,v in enumerate(vals):axes[0].text(i,v+1,f'{v:.1f}%',ha='center')
    axes[1].hist(100*np.array(image['null_gains']),bins=25,color=gray,alpha=.75)
    axes[1].axvline(100*summary['gain'],color=blue,lw=2,label=f"Observed +{100*summary['gain']:.2f} points")
    axes[1].set(xlabel='Text improvement over controls (percentage points)',ylabel='Within-page permutations',title=f"Page-controlled null: p = {summary['permutation_p']:.3f}")
    axes[1].legend(frameon=False)
    save(fig,'image-association')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    families=['substitution','variable-homophonic','naibbe'];labels=['Substitution','Variable-length control','Naibbe']
    for ax,oracle in zip(axes,(False,True)):
        for j,dataset in enumerate(('modern','historical')):
            values=[]
            for family in families:
                rows=[r for r in cipher['cases'] if r['family']==family and r['dataset']==dataset]
                value=sum(r['characters']*(r['oracle_candidate_cer'] if oracle else r['character_error_rate']) for r in rows)/sum(r['characters'] for r in rows)
                values.append(100*value)
            x=np.arange(3)+(j-.5)*.35
            ax.bar(x,values,width=.35,color=[blue,orange][j],label=dataset.title())
            for pos,v in zip(x,values):ax.text(pos,v+(.7 if oracle else 7),f'{v:.1f}',ha='center',fontsize=9)
        ax.set_xticks(range(3),labels);ax.tick_params(axis='x',labelsize=9)
        ax.set(ylabel='Character edit error (%)',ylim=(0,100 if oracle else 750),
               title='Best candidate after seeing answers (diagnostic only)' if oracle else 'Selected output: actual decoder result')
        ax.legend(frameon=False)
    save(fig,'codebook-free')
    segtable=[]
    for dataset in ('modern','historical'):
        r=seg['summary'][dataset];lo,hi=r['absolute_wer_improvement_interval_95']
        segtable.append(f"| {dataset.title()} | {100*r['baseline']['word_error_rate']:.2f}% | {100*r['lexicon']['word_error_rate']:.2f}% | {100*r['relative_wer_reduction']:.1f}% | {100*lo:.2f}–{100*hi:.2f} points | {r['lexicon']['boundary_f1']:.3f} |")
    segtext='''# Fresh-passage lexicon segmentation

The frozen lexicon method improves both corpora, but historical word recovery remains poor.
The declared improvement gate passed; the “solved” gate failed.

**Word error rate (WER):** inserted, deleted, or substituted words divided by reference words.
**Boundary F1:** precision/recall balance for inserted spaces; it is not word accuracy.
**Freeze:** source hashes, code, model, and parameters recorded before challenge preparation.

```text
Fit word counts/transitions on modern ISDT train and Morph-it! surface forms
Tune 18 configurations on 200 non-Dante ISDT development sentences
Freeze the winning method
Generate 24 fresh passages excluding all earlier challenge sentences
Save predictions without reading originals
Grade once; preserve historical failures
```

![Word recovery](../method-benchmark/figures/segmentation.png)

| Corpus | Previous WER | Lexicon WER | Relative reduction | 95% interval for improvement | Boundary F1 |
|---|---:|---:|---:|---|---:|
'''+ '\n'.join(segtable)+'''

1. **Method.** A lexicon trie and beam-Viterbi decoder combine word counts, smoothed
   word transitions, and unknown-word costs. Morph-it! plus training words supplies
   404,346 normalized forms. The selected pseudocount is 1, bigram weight 0.5, unknown
   cost 15+3×length, beam 8, maximum word length 32. Development WER was 6.14%.
2. **Fresh evaluation.** Twelve modern ISDT test passages and twelve historical
   Italian-Old passages, 400–800 letters each. There is no source-sentence overlap with
   the old six-passage benchmark or exact train/development sentences. The modern
   result is 66 errors / 1,074 words; historical is 533 / 1,336. One modern passage
   was exact; no historical passage was exact. Characters were preserved throughout.
3. **Declared gates.** Both corpora exceed 20% relative improvement. Historical WER
   39.90% fails the predeclared requirement of <10% on both corpora. The method improved
   segmentation; it did not fix historical segmentation. The modern prior is a plausible
   mismatch, but this experiment does not isolate vocabulary, spelling, syntax, or genre.
4. **Limits.** The 2,000-draw paired bootstrap samples passages, not new authors.
   Dante passages are fresh but share author/work with the prior benchmark. Gold remains
   evaluator-only in software; this is not an OS security boundary. No tuning followed
   this evaluation. No Voynich final-test text was scored.

Sources: [Morph-it! documentation](https://docs.sslmit.unibo.it/doku.php?id=resources:morph-it),
Marco Baroni and Eros Zanchetta, version 0.48, under the upstream CC BY-SA 2.0 option;
[pinned mirror](https://github.com/giodegas/morphit-lemmatizer/tree/b99d75d774367e4bedc5c5f339dec384777488ff),
[license](https://creativecommons.org/licenses/by-sa/2.0/).
Normalization changes forms; the derived lexicon/model remain local with source attribution.
Language corpus revisions and licenses: [source manifest](../language-sources.json).

Audit: [frozen plan](../METHOD_BENCHMARK_PLAN.md), [lexicon source/hash](../segmentation-sources.json),
[freeze and development grid](freeze.json), [per-case metrics](results.json).
Commands are in the [README](../../README.md#current-benchmark-commands).
'''
    (ROOT/'experiments/segmentation/REPORT.md').write_text(segtext)
    ci=summary['folio_bootstrap_interval_95']
    imagetext=f'''# Text–image association pilot

Existing visual descriptions provide a narrow test outside text-only statistics.
This pilot did not establish a text–image association.

**Endpoint:** explicit light versus dark root descriptions; silence about a part is unknown.
**Balanced accuracy:** mean accuracy across light and dark classes; 50% is the trivial baseline.
**Grouped evaluation:** each fold holds out an entire physical folio, including its page sides/panels.

```text
Audit existing descriptions before inspecting text associations
Exclude uncertain labels, absent mentions, duplicate readings, and non-training folios
Freeze endpoint, controls, model, and tests
Hold out each folio; compare controls with controls plus EVA label text
Shuffle annotation labels within pages 999 times; refit the same models
Report the observed gain against the controlled null
```

![Image association](../method-benchmark/figures/image-association.png)

1. **Source and eligibility.** Grove/Stolfi's 1998 catalogue has 249 deduplicated plant-related
   objects. Root-versus-plant class is concentrated on folio 99 and cannot support the
   proposed comparison. Leaf/flower coloration has too few eligible examples. The
   root-color endpoint retains 59 confident plant/root labels: 27 light and 32 dark,
   across folios 88, 89, 99, 100, 101, and 102. All have the same pharmaceutical section
   and inherited hand label 1. Historical f101v subdivisions share one folio group.
2. **Controls.** Label length, word count, relative label index, location group, and
   transcriber. The text model adds fixed-hash EVA character 1–3-grams. Ridge penalty 10
   and decision threshold 0.5 were fixed before evaluation. Standardization fits only
   each training fold. The cached linear prediction operator is mathematically equivalent
   to refitting ridge for every permutation; held-out target values have zero influence.
3. **Result.** Controls score {100*summary['control_balanced_accuracy']:.2f}%; adding text scores
   {100*summary['text_balanced_accuracy']:.2f}%. The gain is {100*summary['gain']:.2f} percentage points,
   p={summary['permutation_p']:.3f}, paired folio-bootstrap interval
   [{100*ci[0]:.2f}, {100*ci[1]:.2f}] points. The predeclared promising-result gate fails.
   This is a negative pilot, not evidence that images and writing are unrelated.
4. **Limits and remaining study.** Six folios are few; folio 99 supplies only one eligible
   example. Descriptions are selective and the original annotators could see the text.
   Within-page permutations preserve page composition, not every layout effect. This is
   pharmaceutical-label coverage, not the planned larger herbal-page study. Independent
   visual annotation of crops with text hidden, explicit absent/unknown labels, annotation
   agreement, and a fresh held-out sample are still required. Do not infer plant species
   or word meanings from these results.

Source: [Grove/Stolfi catalogue and format](https://www.ic.unicamp.br/~stolfi/EXPORT/voynich/98-02-01-lotsa-labels/),
[annotation index](https://www.ic.unicamp.br/~stolfi/EXPORT/voynich/98-02-01-lotsa-labels/labtit-98-07-20.idx).
No explicit reuse license was found; raw annotations remain local. The repository stores
retrieval code, source hash, aggregate results, and attribution, not the annotation text.

Audit: [fixed protocol](PROTOCOL.md), [freeze/source hash](freeze.json),
[aggregate and per-folio results](results.json). No Voynich final-test scores were opened.
'''
    (ROOT/'experiments/association/REPORT.md').write_text(imagetext)
    table=[]
    for family in families:
        for dataset in ('modern','historical'):
            r=cipher['summary'][family][dataset]
            table.append(f"| {family} | {dataset} | {100*r['character_error_rate']:.2f}% | {100*r['word_error_rate']:.2f}% | {r['recovery_gate_passes']}/2 |")
    ciphertext='''# Codebook-free recovery: failed broader control

This tests recovery after removing supplied cipher-family labels and the Naibbe codebook.
The decoder is not validated for codebook-free Naibbe: it also fails its broader synthetic control.

**Character error rate (CER):** character edit distance divided by reference length; insertions can make it exceed 100%.
**Positive control:** verified ciphertext with a known original, used to check that the method can recover content.
**Oracle diagnostic:** the best candidate chosen after reading the reference; it cannot count as decoder success.

```text
Freeze the unpaired Italian prior, candidate mappings, score, and CPU budget
Generate four fresh passages under three encodings; keep originals and keys private
Run one decoder on opaque IDs and ciphertext only
Freeze every candidate and the selected prediction
Open originals to grade; report failed controls and oracle diagnostics separately
```

![Codebook-free recovery](../method-benchmark/figures/codebook-free.png)

| Encoding | Corpus | Selected CER | Selected word error | Recovery gate passes |
|---|---|---:|---:|---:|
'''+ '\n'.join(table)+f'''

1. **Assistance removed.** The public cases have only opaque IDs and ciphertext. No family,
   structural codebook, true key, encoder trace, original length, or word boundaries are
   supplied. Italian and its normalized alphabet remain known. The decoder tries observed
   characters and space-delimited units, with bijections and many-to-one one/two-letter
   expansions. These are explicit representation assumptions, not arbitrary cipher discovery.
2. **Controls and separation.** Two fresh modern and two fresh historical passages,
   1,200–1,800 letters each, exclude all earlier benchmark source sentences. Each is encoded
   by substitution, an artificial variable-length homophonic code, and published Naibbe
   with a hidden global letter permutation. The artificial code has two cipher spellings
   per plaintext chunk. All encoder roundtrips passed before solving. There are four
   independent passages, not twelve independent originals. Source/preparation and grading
   can access gold; the solver path cannot. This is auditable software separation.
3. **Selection failure.** The bijection candidate recovered all letters in all four simple
   substitution cases. The selected output preserved those correct candidates only for
   modern Italian. On both historical cases the fixed score preferred an incorrect
   expansion: e.g. score −2.197 for the wrong candidate versus −2.402 for the exact letters.
   The language prior therefore does not reliably choose the correct decoding. It is
   a mean conditional four-gram score with a unigram-distribution penalty, not a calibrated
   model of ciphertext generation or output length. Historical outputs have 146.3% pooled
   CER because the wrong mappings add many characters; this is not a clipped accuracy score.
4. **Broader failure.** The variable-length control and Naibbe fail even under the
   post-hoc best-candidate diagnostic (roughly 79–87% character error per case). Their
   selected outputs expand even further. Thus the weakness is not selection alone: search
   has not recovered the broader control. This result invalidates transfer claims for
   this decoder under this budget; it does not establish that every codebook-free method
   must fail, nor that Voynich lacks meaning. The earlier 99.44% Naibbe letter accuracy
   depended on the supplied structural codebook.

The gate required CER <=1% and word error <=10% per passage. Only the two modern
substitution cases pass. No full passage is exact. All candidate searches completed
72,000 proposals without hitting the 120-second cap; total solving and segmentation
took {cipher['seconds']:.1f} CPU wall-clock seconds. Grading time is additional. The old and
new benchmarks differ in passages and assumptions; their difference is not a causal
estimate of the codebook's effect. No tuning or reranking followed grading.

The next mechanism to test, if pursued, is **decoding selection under language-prior
mismatch**: can a generative encoding/complexity criterion prefer the correct historical
candidate without access to its reference? First validate selection and broader search
on separate development controls, then freeze and use new held-out passages. This is
not another BPC sweep and has not been launched automatically.

Source: Michael A. Greshko (2025),
[The Naibbe cipher](https://doi.org/10.1080/01611194.2025.2566408),
[published code and license](https://github.com/greshko/naibbe-cipher).
[Pinned source/attribution](../decipherment-sources.json). The published encoder is used
only in challenge preparation; no structural tables are opened by the solver.

Audit: [protocol](PROTOCOL.md), [frozen code/prior hashes](freeze.json),
[all per-case and candidate scores](results.json). Candidate mappings, ciphertext,
predictions, and hidden answers remain in `artifacts/codebook-free/`. No Voynich final-test
text was used. Do not replace this negative run with a retuned score on the same passages.
'''
    (ROOT/'experiments/codebook-free/REPORT.md').write_text(ciphertext)
    combined='''# Decipherment-method benchmark: 21 September 2026

The deliverable is a validated recovery benchmark with positive controls and explicit
Voynich limits. Translation into English or Italian is not supported by current evidence.
Publication suitability is an aim; the current pilot is not a completed decipherment paper.

## What was done

- Deleted Runpod pod `6o8irlwqlzhsb1` and attached temporary volumes. The audit confirms
  deletion; network storage is 0 GB and balance was $8.89 at check. Local results remain.
- Formally closed broad prediction training. The monitor stays paused; no more BPC sweeps,
  longer training, or larger models unless a specific falsifiable mechanism justifies one.
- Froze and evaluated a lexicon segmenter on 24 fresh passages after non-Dante tuning.
- Ran a codebook-free decoder on 12 ciphertexts from four further fresh passages.
- Started the independent-evidence track with existing visual descriptions and controlled,
  held-out-folio tests. Preserved failed endpoints, controls, and inference limits.

## Why it was done

Earlier work improved prediction without validating meaning. These tests measure actual
content recovery and an external visual association, while exposing which supplied hints
and evaluation choices make an apparent success possible.

## 1. Segmentation improved; the historical problem remains

![Fresh segmentation](figures/segmentation.png)

Word error fell from 15.9% to 6.1% on modern Italian and from 60.9% to 39.9% on historical
Italian. Both exceed the declared 20% relative-improvement gate. Historical text fails
our <10% word-error requirement. The 24 fresh passages exclude all previous challenge
sentences; historical passages still come from Dante, so they are not independent authors.

[Method, uncertainty, sources, and gates](../segmentation/REPORT.md).

## 2. Removing the codebook reveals an unvalidated decoder

![Recovery under fewer hints](figures/codebook-free.png)

Modern substitution succeeds at the declared gate (0% letter error, 5.72% pooled word
error). Historical substitution exposes a selection failure: an exact letter candidate
exists, but the language-prior score chooses an incorrect expansion. Naibbe and the broader
variable-length positive control both fail, including the post-hoc best-candidate check.
CER can exceed 100% because the chosen mappings insert many extra characters.

This is a documented negative for this decoder, not a proof that every possible
unknown-cipher method fails. The earlier codebook-assisted Naibbe result does not
establish transfer to Voynich. Search and selection need validation before such a claim.

[Fixed protocol, candidate diagnostics, and detailed results](../codebook-free/REPORT.md).

## 3. Image evidence: implemented pilot, no established association

![External visual evidence](figures/image-association.png)

The Grove/Stolfi root-color descriptions yield 59 labels on six training folios.
Controls score 59.43% balanced accuracy; adding EVA text scores 60.88%. The 1.45-point
gain has p=0.348 under within-page permutations and a folio interval of −6.81 to +20.83
points. It fails the declared gate. The source is useful for starting the study, but
selective descriptions and annotators' access to text limit independence.

[Source audit, controls, uncertainty, and remaining annotation work](../association/REPORT.md).

## Decision and remaining work

```text
Keep the prediction track closed
Preserve these frozen failures and the successful modern control
If continuing, validate decoder selection/search on development controls
Freeze again and reserve entirely new passages before any confirmatory run
Acquire independent visual annotations and a new held-out image sample
Require external evidence before proposing Voynich meanings
```

The next concrete bottleneck is validation of **decoding selection under language-prior
mismatch**, together with a search method that passes variable-length positive controls.
Historical segmentation remains another explicit failure. The larger herbal-page study
needs annotations of images with their text hidden, explicit unknown/absent labels,
annotator agreement, and frozen folio groups. None of these gaps is solved by renting a
larger GPU. No new follow-up computation was launched after viewing the answers.

All new work used CPU, with no new cloud charges from computation or model downloads.
The one-week allocation was a maximum budget, not a required runtime. Final Voynich test
pages remain sealed. The report does not identify any Voynich word or claim a translation.

## Reproduction and provenance

- [Current plan](../METHOD_BENCHMARK_PLAN.md); [central research record](../../RESEARCH_LOG.md).
- [Cloud deletion audit](../cloud-cleanup.json); [verification](verification.json).
- [Commands and artifact requirements](../../README.md#current-benchmark-commands).
- Recheck local frozen inputs and passage exclusions with `python -m experiments.benchmark_verify`.
- Reports regenerate from saved aggregate results with `python -m experiments.benchmark_report`.
  Plots need matplotlib; reports perform no training or test scoring.
- Source hashes, frozen code, configuration grids, per-case metrics, and failed gates are
  stored alongside each report. Raw licensed sources and private references remain
  local. A fresh replication needs an isolated output archive; it is not byte-identical
  because challenge keys are random. Exact regrading needs the preserved local artifacts.
- Existing Qwen/GRU, linguistic-statistics, and codebook-assisted reports remain historical
  records. Their next-step suggestions are superseded by the closed prediction policy.
'''
    (OUT/'REPORT.md').write_text(combined)
    print('Wrote four reports and three PNG/SVG figures')


if __name__=='__main__':main()
