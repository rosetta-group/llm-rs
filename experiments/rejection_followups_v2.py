"""Three bounded development follow-ups; never a fresh confirmation or Voynich run."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tarfile
import time

from experiments import rejection_transfer_v2 as previous
from voynich.data import digest
from voynich.description_length import CharacterPrior
from voynich.rejection import decide, transfer
from voynich.rejection_development import decide_transfer, frequency_copy, copying_diagnostics
from voynich.variable_units_bounded import refine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/rejection-followups'
STATE = ROOT / 'artifacts/rejection-followups'
LANGUAGES = previous.LANGUAGES
read, seal = previous.read, previous.seal


def prepare():
    previous.verify()
    if (STATE / 'control-plan.json').exists():
        raise FileExistsError('Controls already prepared')
    benchmark = read(OUT / 'benchmark.json')
    if not benchmark['all_winners_identical']:
        raise ValueError('Scoring equivalence failed')
    results = read(ROOT / 'experiments/rejection-transfer-v2/results.json')
    controls, inputs = [], {}
    with tarfile.open(ROOT / 'experiments/rejection-transfer-v2/evaluated-records.tar.gz') as archive:
        for row in results['outcomes']:
            if row['kind'] != 'positive':
                continue
            ident = 'frequency-copy-' + str(row['block'])
            diagnostics = {}
            for role_index, role in enumerate(('fit', 'transfer')):
                original = json.load(archive.extractfile(f"public/{row['id']}-{role}.json"))['ciphertext'].split()
                seed = 20260925 + 2 * row['block'] + role_index
                copied = frequency_copy(original, seed)
                if Counter(original) != Counter(copied):
                    raise ValueError('Copying changed the token multiset')
                path = STATE / 'public' / f'{ident}-{role}.json'
                seal(path, dict(id=ident, ciphertext=' '.join(copied)))
                inputs[str(path.relative_to(ROOT))] = digest(path)
                diagnostics[role] = dict(seed=seed, original=copying_diagnostics(original),
                                         copied=copying_diagnostics(copied), exact_multiset_match=True)
            controls.append(dict(id=ident, source_id=row['id'], source_language=row['language'],
                                  block=row['block'], diagnostics=diagnostics))
    seal(STATE / 'control-plan.json', dict(controls=controls, inputs=inputs,
           source_archive_sha256=digest(ROOT / 'experiments/rejection-transfer-v2/evaluated-records.tar.gz'),
           development_only=True, at=time.time()))
    print('Prepared',len(controls),'development copying pairs with exact token frequencies.')


def freeze():
    old = previous.verify()
    benchmark = read(OUT / 'benchmark.json')
    code = dict(old['code_and_records'])
    for name in ('experiments/rejection_followups.py', 'experiments/rejection_followups_v2.py',
                 'experiments/rejection-followups/CONTROL_RESOURCE_REPAIR.md', 'experiments/rejection_refiner_benchmark.py',
                 'experiments/rejection-followups/PROTOCOL.md', 'experiments/rejection-followups/BENCHMARK_PLAN.json',
                 'experiments/rejection-followups/benchmark.json', 'tests/test_rejection_followups.py',
                 'voynich/rejection_development.py', 'voynich/variable_units_bounded.py',
                 'experiments/rejection-transfer-v2/results.json'):
        code[name] = digest(ROOT / name)
    external = dict(old['external'])
    external.update(read(STATE / 'control-plan.json')['inputs'])
    external[str((STATE / 'control-plan.json').relative_to(ROOT))] = digest(STATE / 'control-plan.json')
    for path in (STATE / 'benchmark-inputs').glob('*.json'):
        external[str(path.relative_to(ROOT))] = digest(path)
    seal(OUT / 'freeze-v2.json', dict(code_and_records=code, external=external, prior_files=old['prior_files'],
                entropy=old['entropy'], settings=old['settings'], fit_cap=3600, refine_cap=1200,
                sweeps=200, max_evaluations=20_000_000, batch_size=512, workers=5, threads=2,
                max_worker_seconds=12*3600, backend=benchmark['selected_backend'], at=time.time(), development_only=True))
    print('Frozen development controls. Commit before fitting.')


def verify():
    f = read(OUT / 'freeze-v2.json')
    for path, expected in {**f['code_and_records'], **f['external']}.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen drift: ' + path)
    for path in [*f['code_and_records'], 'experiments/rejection-followups/freeze-v2.json']:
        data = subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != digest(ROOT/path):
            raise ValueError('Uncommitted freeze: '+path)
    return f


def fit_key(tokens, prior, frozen):
    s, cap = frozen['settings'], frozen['fit_cap']
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap*.25)
    first = previous.joint_em(tokens, prior, **em)
    second = previous.prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = previous.repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'],
                 passes=s['repair_passes'], usage_floor=s['repair_usage_floor'], **em)
    units = previous.role_units(repaired['segmentation'])
    ref = refine(units, prior, previous.majority_key(units,repaired['recovered']), (),
                 seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                 cap=min(frozen['refine_cap'],max(30.,cap-(time.monotonic()-started))),
                 sweeps=frozen['sweeps'], max_evaluations=frozen['max_evaluations'],
                 batch_size=frozen['batch_size'], backend=frozen['backend'])
    seconds = time.monotonic()-started
    hits = dict(first=first['cap_hit'],second=second['cap_hit'],repair=repaired['cap_hit'],
                refine=ref['cap_hit'],case=seconds>cap)
    return dict(recovered=ref['recovered'],mapping=ref['mapping'],cap_hits=hits,cap_hit=any(hits.values()),
                bits_per_letter=prior.bits(ref['recovered'])/len(ref['recovered']),seconds=seconds,
                refiner={k:v for k,v in ref.items() if k not in ('mapping','recovered')})


def worker(stage, ident, language):
    f = read(OUT/'freeze-v2.json')
    plan = read(STATE/'control-plan.json')
    provenance = dict(freeze_sha256=digest(OUT/'freeze-v2.json'),control_plan_sha256=digest(STATE/'control-plan.json'),
                      id=ident,language=language)
    path = STATE/stage/f'{ident}-{language}.json'
    if path.exists():
        record=read(path)
        if record['provenance']!=provenance:raise ValueError('Checkpoint provenance changed')
        return str(path)
    public=STATE/'public'/f'{ident}-{stage}.json'
    if digest(public)!=plan['inputs'][str(public.relative_to(ROOT))]:raise ValueError('Public input changed')
    tokens=read(public)['ciphertext'].split()
    prior_path=ROOT/f'artifacts/rejection-transfer-v2/priors/{language}.npz'
    if digest(prior_path)!=f['prior_files'][str(prior_path.relative_to(ROOT))]:raise ValueError('Prior changed')
    prior=CharacterPrior.load(prior_path)
    if stage=='fit':
        result=fit_key(tokens,prior,f)
    else:
        key_path=STATE/'fit'/f'{ident}-{language}.json'
        if digest(key_path)!=read(STATE/'sealed-keys'/f'{ident}.json')['key_files'][str(key_path.relative_to(ROOT))]:
            raise ValueError('Key changed after sealing')
        key=read(key_path)['result']['mapping']
        result=transfer(tokens,key,prior)
        result['key_sha256']=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    seal(path,dict(provenance=provenance,result=result,at=time.time()))
    return str(path)


def run():
    f=verify()
    if (OUT/'control-results.json').exists():raise FileExistsError('Development controls already ended')
    for name in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='2' if name in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS') else '1'
    outcomes=[];stop=None
    with ProcessPoolExecutor(max_workers=f['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for control in read(STATE/'control-plan.json')['controls']:
            ident=control['id']
            graded=STATE/'graded'/f'{ident}.json'
            if graded.exists():
                outcome=read(graded)
                if any(digest(ROOT/p)!=h for p,h in outcome['files'].items()):
                    raise ValueError('Completed development checkpoint changed')
                outcomes.append(outcome)
                continue
            used=sum(read(p)['result']['seconds'] for p in (STATE/'fit').glob('*.json'))
            if used+len(LANGUAGES)*f['fit_cap']>f['max_worker_seconds']:
                stop='inconclusive_worker_budget';break
            print(ident+': fitting five priors',flush=True)
            jobs=[pool.submit(worker,'fit',ident,l) for l in LANGUAGES]
            paths=[Path(j.result()) for j in jobs]
            keys=dict(key_files={str(p.relative_to(ROOT)):digest(p) for p in paths},freeze_sha256=digest(OUT/'freeze-v2.json'))
            key_path=STATE/'sealed-keys'/f'{ident}.json'
            if key_path.exists():
                if read(key_path)!=keys:raise ValueError('Key seal changed')
            else:seal(key_path,keys)
            print(ident+': keys sealed; transferring',flush=True)
            jobs=[pool.submit(worker,'transfer',ident,l) for l in LANGUAGES]
            paths += [Path(j.result()) for j in jobs]
            scores={}
            for language in LANGUAGES:
                a=read(STATE/'fit'/f'{ident}-{language}.json')['result']
                b=read(STATE/'transfer'/f'{ident}-{language}.json')['result']
                scores[language]=dict(fit_excess=a['bits_per_letter']-f['entropy'][language],
                     transfer_excess=None if b['bits_per_letter'] is None else b['bits_per_letter']-f['entropy'][language],
                     coverage=b['token_coverage'],cap_hit=a['cap_hit'],cap_hits=a['cap_hits'],seconds=a['seconds'],refiner=a['refiner'])
            outcome=dict(**control,scores=scores,original=decide(scores),candidate=decide_transfer(scores),
                         files={str(p.relative_to(ROOT)):digest(p) for p in paths})
            outcomes.append(outcome)
            target=STATE/'graded'/f'{ident}.json'
            if target.exists():
                if read(target)!=outcome:raise ValueError('Graded control changed')
            else:seal(target,outcome)
            print(json.dumps(dict(id=ident,original=outcome['original'],candidate=outcome['candidate'])),flush=True)
    seal(OUT/'control-results.json',dict(outcomes=outcomes,stop_reason=stop,planned=3,development_only=True,
         fit_worker_seconds=sum(read(p)['result']['seconds'] for p in (STATE/'fit').glob('*.json')),
         freeze_sha256=digest(OUT/'freeze-v2.json'),freeze_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def calibrate():
    verify()
    original=read(ROOT/'experiments/rejection-transfer-v2/results.json')
    originals={r['id']:r['scores'] for r in original['outcomes'] if 'scores' in r}
    cases=[]
    for row in original['outcomes']:
        candidates=[l for l in LANGUAGES if l!=row['language']] if row['kind']=='absent' else None
        cases.append(dict(id=row['id'],kind=row['kind'],expected=row['language'] if row['kind']=='positive' else None,
                          scores=originals[row['id']],candidates=candidates))
    for row in read(OUT/'control-results.json')['outcomes']:
        cases.append(dict(id=row['id'],kind='frequency_copy',expected=None,scores=row['scores'],candidates=None))
    rows=[]
    for case in cases:
        rows.append(dict(id=case['id'],kind=case['kind'],expected=case['expected'],
                         original=decide(case['scores'],case['candidates']),
                         candidate=decide_transfer(case['scores'],case['candidates'])))
    sensitivity=[]
    for step in range(21):
        ceiling=step/20
        counts={}
        for case in cases:
            result=decide_transfer(case['scores'],case['candidates'],ceiling)
            c=counts.setdefault(case['kind'],dict(evaluated=0,accepted=0,correct=0,inconclusive=0))
            c['evaluated']+=1;c['accepted']+=result['accepted'] is not None
            c['inconclusive']+=result['inconclusive']
            c['correct']+=not result['inconclusive'] and result['accepted']==case['expected']
        sensitivity.append(dict(transfer_ceiling=ceiling,counts=counts))
    seal(OUT/'calibration-results.json',dict(development_only=True,rows=rows,sensitivity=sensitivity,
          threshold_selected_on_fresh_test=False,independent_source_key_blocks=original['independent_positive_keys']))


def verify_records():
    f=verify();result=read(OUT/'control-results.json');replayed=0
    for row in result['outcomes']:
        for path,sha in row['files'].items():
            if digest(ROOT/path)!=sha:raise ValueError('Control result file changed')
        scores={}
        for language in LANGUAGES:
            key_path=STATE/'fit'/f"{row['id']}-{language}.json"
            if digest(key_path)!=read(STATE/'sealed-keys'/f"{row['id']}.json")['key_files'][str(key_path.relative_to(ROOT))]:
                raise ValueError('Sealed control key changed')
            fit=read(key_path)['result'];held=read(STATE/'transfer'/f"{row['id']}-{language}.json")['result']
            if held['key_sha256']!=hashlib.sha256(json.dumps(fit['mapping'],sort_keys=True).encode()).hexdigest():
                raise ValueError('Transfer key differs from sealed key')
            prior=CharacterPrior.load(ROOT/f'artifacts/rejection-transfer-v2/priors/{language}.npz')
            actual=transfer(read(STATE/'public'/f"{row['id']}-transfer.json")['ciphertext'].split(),fit['mapping'],prior)
            for k,v in actual.items():
                if isinstance(v,float):
                    if abs(v-held[k])>1e-10:raise ValueError('Control transfer score mismatch')
                elif v!=held[k]:raise ValueError('Control transfer mismatch')
            bits=prior.bits(fit['recovered'])/len(fit['recovered'])
            if abs(bits-fit['bits_per_letter'])>1e-10:raise ValueError('Control fit score mismatch')
            scores[language]=dict(fit_excess=bits-f['entropy'][language],transfer_excess=None if actual['bits_per_letter'] is None else actual['bits_per_letter']-f['entropy'][language],coverage=actual['token_coverage'],cap_hit=fit['cap_hit'])
            replayed+=1
        if decide(scores)!=row['original'] or decide_transfer(scores)!=row['candidate']:raise ValueError('Decision mismatch')
    return dict(replayed_keys=replayed,transfer_and_decisions_reproduce=True,development_only=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','freeze','verify','run','calibrate','verify_records'])
    command=p.parse_args().command
    result=globals()[command]()
    if result is not None:print(json.dumps(result if command=='verify_records' else {'verified':True},indent=2))
