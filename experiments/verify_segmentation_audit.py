"""Regrade audit records and check provenance without rerunning cipher search."""
import json
import math
from pathlib import Path

from experiments.joint_development import encode, load_vendor, majority_key, role_units
from experiments.joint_recovery_v4 import verify as verify_round_four, PRIOR_PATH
from experiments.joint_development_v3 import segmenter
from experiments.segmentation_audit import streams, reference, metrics
from voynich.context_reparse import score
from voynich.data import digest
from voynich.description_length import CharacterPrior
from voynich.decipher import edit_distance
from voynich.segmentation import boundaries

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/segmentation-audit'


def verify():
    result=json.loads((OUT/'results.json').read_text());p=result['provenance']
    verify_round_four()
    for path,h in p['code_sha256'].items():
        if digest(ROOT/path)!=h: raise ValueError('Source drift: '+path)
    for path,h in [(PRIOR_PATH,p['prior_sha256']),
                   (ROOT/'experiments/joint-recovery-v4/freeze.json',p['freeze_sha256']),
                   (ROOT/'artifacts/standard-decipherment/segmenter.json',p['word_model_sha256'])]:
        if digest(path)!=h: raise ValueError('Input drift: '+str(path))
    if not p['development_only'] or p['voynich_text_used'] or result['final_test_scored']:
        raise ValueError('Unexpected evaluation scope')
    texts=streams();vendor=load_vendor();prior=CharacterPrior.load(PRIOR_PATH);ws=segmenter()
    if {r['source'] for r in result['rows']}!=set(texts): raise ValueError('Incomplete audit')
    for r in result['rows']:
        source=r['source'];record=json.loads((OUT/(source+'-predictions.json')).read_text())
        dense,plain,length=reference(texts[source]);tokens,truth=encode(vendor,dense,p['seed'])
        if (record['dense'],record['plaintext'],record['ciphertext'])!=(dense,plain,tokens): raise ValueError('Development record drift')
        if [tuple(t) for t in record['truth']]!=truth: raise ValueError('Encoder trace drift')
        perfect=record['perfect_word_prediction']
        if perfect.replace(' ','')!=dense[:length] or ws.segment(dense[:length])!=perfect:
            raise ValueError('Perfect-letter segmentation drift')
        gold,pred=boundaries(plain),boundaries(perfect)
        expected=dict(characters=length,words=len(plain.split()),word_errors=edit_distance(perfect.split(),plain.split()),
                      wer=edit_distance(perfect.split(),plain.split())/len(plain.split()),
                      boundary_f1=2*len(gold&pred)/max(1,len(gold)+len(pred)))
        if expected!=r['perfect_letters']: raise ValueError('Word audit drift')
        mapping=majority_key(role_units(record['baseline']['segmentation']),record['baseline']['recovered'])
        baseline_score=score(record['baseline']['segmentation'],mapping,prior)
        candidate_score=score(record['candidate']['segmentation'],mapping,prior)
        if not math.isclose(baseline_score,r['baseline_score'],abs_tol=1e-7) or not math.isclose(candidate_score,r['candidate_score'],abs_tol=1e-7):
            raise ValueError('Conditional score drift')
        accepted=candidate_score<baseline_score-1e-8
        if accepted!=r['candidate_accepted_by_score']: raise ValueError('Acceptance drift')
        chosen=record['candidate'] if accepted else record['baseline']
        for stage,prediction in [('baseline',record['baseline']),('candidate',chosen),('oracle',record['oracle'])]:
            if [''.join(parts) for parts in prediction['segmentation']]!=tokens: raise ValueError('Cipher reconstruction failed')
            grade=metrics(prediction['recovered'],plain,dense,ws,length)
            if any(grade[k]!=r[stage][k] for k in grade): raise ValueError('Grade drift: '+source+'/'+stage)
            agreement=sum(tuple(a)==tuple(b) for a,b in zip(prediction['segmentation'],truth))/len(truth)
            if agreement!=r[stage]['parse_agreement']: raise ValueError('Parse score drift')
        print('Verified source, ciphertext, scores and grades:',source,flush=True)
    means={stage:{m:sum(r[stage][m] for r in result['rows'])/3 for m in ('cer','wer')} for stage in ('baseline','candidate','oracle')}
    if means!=result['means']: raise ValueError('Summary drift')
    selected=(means['baseline']['cer']-means['candidate']['cer']>=.01
              and all(r['candidate']['cer']-r['baseline']['cer']<=.01 for r in result['rows'])
              and means['candidate']['wer']<=means['baseline']['wer']
              and not any(any(r['cap_hits'].values()) for r in result['rows']))
    if selected!=result['selected']: raise ValueError('Selection drift')
    return dict(verified=True,sources=3,selected=selected,search_rerun=False)


if __name__=='__main__': print(json.dumps(verify()))
