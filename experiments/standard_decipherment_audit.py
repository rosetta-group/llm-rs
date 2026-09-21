"""Verify the completed benchmark independently of its reporting code."""
import json
import platform
import subprocess
from pathlib import Path

from experiments.standard_decipherment import ROOT, OUT, STATE, read, write, verify
from voynich.data import digest


def audit():
    frozen=verify()
    challenge=read(STATE/'challenge.json')
    predictions=read(STATE/'predictions.json')
    results=read(OUT/'results.json')
    answers=read(STATE/'evaluator-only/answers.json')
    public=read(STATE/'public.json')
    expected={r['id'] for r in answers}
    for rows in (public,predictions['results'],results['cases']):
        if len(rows)!=16 or {r['id'] for r in rows}!=expected:
            raise ValueError('Case inventory mismatch')
    if any(set(r)!= {'id','ciphertext'} for r in public):
        raise ValueError('Public metadata leak')
    excluded={'modern':set(),'historical':set()}
    for r in read(ROOT/'artifacts/decipherment/evaluator-only/answers.json'):
        excluded['historical'].update(r['source_sentences'])
    for folder in ('segmentation','codebook-free'):
        for r in read(ROOT/f'artifacts/{folder}/evaluator-only/answers.json'):
            excluded[r['dataset']].update(r['source_ids'])
    passages={r['passage']:r for r in answers}
    if len(passages)!=4: raise ValueError('Wrong independent passage count')
    for p in passages.values():
        if set(p['source_ids']) & excluded[p['dataset']]: raise ValueError('Reused source sentence')
    for dataset in ('modern','historical'):
        identifiers=[set(p['source_ids']) for p in passages.values() if p['dataset']==dataset]
        if len(identifiers)!=2 or identifiers[0]&identifiers[1]: raise ValueError('Overlapping fresh passages')
    if not all(r['roundtrip'] for r in answers): raise ValueError('Roundtrip failure')
    for key,path in [('public_sha256',STATE/'public.json'),('answers_sha256',STATE/'evaluator-only/answers.json'),('freeze_sha256',OUT/'freeze.json')]:
        if digest(path)!=challenge[key]: raise ValueError('Challenge provenance mismatch')
    if digest(STATE/'predictions.json')!=results['predictions_sha256']: raise ValueError('Predictions changed after grading')
    if digest(STATE/'segmentation-predictions.json')!=results['segmentation_predictions_sha256']: raise ValueError('Segmentation predictions changed')
    frozen_at=frozen['at'];prepared_at=challenge['at'];predicted_at=predictions['at']
    if not frozen_at<prepared_at<predicted_at: raise ValueError('Stage ordering invalid')
    if read(STATE/'segmentation-predictions.json')['at']<predicted_at: raise ValueError('Plain letters exposed before cipher predictions')
    committed=subprocess.check_output(['git','show',challenge['freeze_commit']+':experiments/standard-decipherment/freeze.json'],cwd=ROOT)
    if json.loads(committed)!=frozen: raise ValueError('Challenge does not name frozen commit')
    expected_gate=all(r['methods']['combined']['gate'] for r in results['cases'] if r['family']=='naibbe')
    if results['naibbe_gate']!=expected_gate or results['voynich_run'] or results['final_test_scored']:
        raise ValueError('Gate or sealed-test violation')
    # Small metadata-only reproducibility record; reference passages remain evaluator-only.
    output=dict(status='passed',method_commit=challenge['freeze_commit'],archive_commit='28ceb16',
        fresh_cases=16,independent_passages=4,earlier_sentence_overlap=0,public_fields=['id','ciphertext'],
        freeze_before_prepare=True,predictions_before_segmentation_controls=True,
        final_test_scored=False,voynich_run=False,naibbe_gate=expected_gate,
        decoding_wall_seconds=predictions['seconds_this_invocation'],
        beam_completions=sum(b['status']=='complete' for r in predictions['results'] for b in r['beam_runs']),
        beam_capacity_skips=sum(b['status']=='capacity_exceeded' for r in predictions['results'] for b in r['beam_runs']),
        hmm_runs=sum('restarts' in r['em'] for r in predictions['results']),
        hmm_restarts_total=sum(r['em'].get('restarts',0) for r in predictions['results']),
        python=platform.python_version(),platform=platform.platform(),
        results_sha256=digest(OUT/'results.json'),freeze_sha256=digest(OUT/'freeze.json'),
        audit_code_sha256=digest(Path(__file__)),
        limits='Procedural blinding on one machine; oracle quality metrics computed only after prediction freeze.')
    write(OUT/'verification.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__': audit()
