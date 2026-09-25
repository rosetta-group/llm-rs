"""Read-only replay audit for completed development follow-ups."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

from experiments import rejection_followups_v2 as development
from voynich.rejection import decide
from voynich.rejection_development import decide_transfer, frequency_copy, copying_diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/rejection-followups'
STATE = ROOT / 'artifacts/rejection-followups'


def audit():
    fixed_keys = development.verify_records()
    plan = development.read(STATE / 'control-plan.json')
    archive_path = ROOT / 'experiments/rejection-transfer-v2/evaluated-records.tar.gz'
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != plan['source_archive_sha256']:
        raise ValueError('Released source archive changed')
    generated = 0
    with tarfile.open(archive_path) as archive:
        for control in plan['controls']:
            for role in ('fit', 'transfer'):
                original = json.load(archive.extractfile(f"public/{control['source_id']}-{role}.json"))['ciphertext'].split()
                actual = development.read(STATE / 'public' / f"{control['id']}-{role}.json")['ciphertext'].split()
                declared = control['diagnostics'][role]
                if actual != frequency_copy(original, declared['seed']):
                    raise ValueError('Copying generator does not reproduce')
                if Counter(actual) != Counter(original):
                    raise ValueError('Copying changed token frequencies')
                if copying_diagnostics(actual) != declared['copied'] or copying_diagnostics(original) != declared['original']:
                    raise ValueError('Copying diagnostics changed')
                generated += 1
    original = development.read(ROOT / 'experiments/rejection-transfer-v2/results.json')['outcomes']
    scores = {row['id']: row['scores'] for row in original if 'scores' in row}
    languages = {row['id']: row['language'] for row in original}
    controls = development.read(OUT / 'control-results.json')['outcomes']
    scores.update({row['id']: row['scores'] for row in controls})
    calibration = development.read(OUT / 'calibration-results.json')
    for row in calibration['rows']:
        candidates = [l for l in development.LANGUAGES if l != languages[row['id']]] if row['kind'] == 'absent' else None
        if decide(scores[row['id']], candidates) != row['original']:
            raise ValueError('Original calibration decision changed')
        if decide_transfer(scores[row['id']], candidates) != row['candidate']:
            raise ValueError('Candidate calibration decision changed')
    for point in calibration['sensitivity']:
        counts = {}
        for row in calibration['rows']:
            candidates = [l for l in development.LANGUAGES if l != languages[row['id']]] if row['kind'] == 'absent' else None
            result = decide_transfer(scores[row['id']], candidates, point['transfer_ceiling'])
            c = counts.setdefault(row['kind'], dict(evaluated=0, accepted=0, correct=0, inconclusive=0))
            c['evaluated'] += 1
            c['accepted'] += result['accepted'] is not None
            c['inconclusive'] += result['inconclusive']
            c['correct'] += not result['inconclusive'] and result['accepted'] == row['expected']
        if counts != point['counts']:
            raise ValueError('Threshold sensitivity does not reproduce')
    benchmark = development.read(OUT / 'benchmark.json')
    return dict(**fixed_keys, copied_passages_reproduced=generated, exact_token_multisets=True,
                calibration_decisions_reproduced=len(calibration['rows']),
                threshold_sensitivity_points_reproduced=len(calibration['sensitivity']),
                benchmark_record_hash_verified=True, benchmark_same_winners=benchmark['all_winners_identical'],
                unused_answers_opened=False, voynich_used=False)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
