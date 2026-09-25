"""Verify released rejection records without refitting keys or opening unused answers.

python -m experiments.verify_rejection_records rejection-transfer-v2
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

from experiments import rejection_transfer as original
from voynich.corpora import conllu_sentences
from voynich.decipher import edit_distance, normalize
from voynich.description_length import CharacterPrior
from voynich.rejection import decide, transfer

ROOT = Path(__file__).resolve().parents[1]


def verify(name):
    folder = ROOT / 'experiments' / name
    frozen = json.loads((folder / 'freeze.json').read_text())
    sources = json.loads((folder / 'sources.json').read_text())
    historical_manifest = ROOT / 'experiments/standard-decipherment/sources.json'
    if hashlib.sha256(historical_manifest.read_bytes()).hexdigest() != sources['historical_manifest_sha256']:
        raise ValueError('Historical source manifest changed since the exclusion audit')
    results = json.loads((folder / 'results.json').read_text())
    with tarfile.open(folder / 'evaluated-records.tar.gz', 'r:gz') as archive:
        data = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    records = {path: json.loads(value) for path, value in data.items()}
    answers = records['answers.json']
    if set(answers) != {r['id'] for r in results['outcomes']}:
        raise ValueError('Archive includes ungraded answers or omits graded answers')
    source_rows = {}
    for source in sources['files']:
        path = ROOT / source['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Challenge source drift')
        source_rows[source['language']] = {
            row['id']: normalize(' '.join(row['words'])) for row in conllu_sentences(path)
        }
    with redirect_stdout(io.StringIO()):
        vendor = original.load_vendor()
    checked_passages = set()
    for ident, gold in answers.items():
        for passage in gold['passages']:
            key = (gold['language'], tuple(passage['source_ids']))
            if key not in checked_passages:
                rebuilt = ' '.join(source_rows[gold['language']][i] for i in passage['source_ids'])
                if rebuilt != passage['plaintext'] or len(rebuilt.replace(' ', '')) != passage['letters']:
                    raise ValueError('Released plaintext differs from pinned source sentences')
                checked_passages.add(key)
        positive = [original.encrypt(vendor, p['plaintext'], gold['key'], seed)
                    for p, seed in zip(gold['passages'], gold['seeds'][:2])]
        if gold['kind'] == 'positive':
            expected = positive
        elif gold['kind'] == 'shuffle':
            expected = [list(tokens) for tokens in positive]
            for tokens, seed in zip(expected, gold['seeds'][2:4]):
                random.Random(seed).shuffle(tokens)
        else:
            expected = [original.copy_text(positive[0], len(tokens), seed)
                        for tokens, seed in zip(positive, gold['seeds'][4:])]
        for role, tokens in zip(('fit', 'transfer'), expected):
            if ' '.join(tokens) != records[f'public/{ident}-{role}.json']['ciphertext']:
                raise ValueError('Released challenge generation does not reproduce')
    priors = {}
    for path, sha in frozen['prior_files'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
            raise ValueError('Prior drift')
        priors[Path(path).stem] = CharacterPrior.load(ROOT / path)
    verified = []
    for ident, gold in answers.items():
        pred = records[f'predictions/{ident}.json']
        for path, expected in pred['files'].items():
            relative = '/'.join(Path(path).parts[-2:])
            if hashlib.sha256(data[relative]).hexdigest() != expected:
                raise ValueError('Archived prediction hash mismatch: ' + relative)
        held_public = records[f'public/{ident}-transfer.json']['ciphertext']
        rows = {}
        for language, prior in priors.items():
            fit_record = records[f'fit/{ident}-{language}.json']
            held_record = records[f'transfer/{ident}-{language}.json']
            fit, held = fit_record['result'], held_record['result']
            inputs = fit_record['provenance']['inputs']
            for role in ('fit', 'transfer'):
                public_path = f'public/{ident}-{role}.json'
                expected = inputs[f'artifacts/{name}/{public_path}']
                if hashlib.sha256(data[public_path]).hexdigest() != expected:
                    raise ValueError('Archived public input changed')
            key_path = f'artifacts/{name}/fit/{ident}-{language}.json'
            sealed_hash = records[f'sealed-keys/{ident}.json']['key_files'][key_path]
            if sealed_hash != hashlib.sha256(data[f'fit/{ident}-{language}.json']).hexdigest():
                raise ValueError('Learned key changed after sealing')
            if held['key_sha256'] != hashlib.sha256(json.dumps(fit['mapping'], sort_keys=True).encode()).hexdigest():
                raise ValueError('Transfer used a different key')
            replay = transfer(held_public.split(), fit['mapping'], prior)
            for key, value in replay.items():
                if isinstance(value, float):
                    if abs(value - held[key]) > 1e-10:
                        raise ValueError('Transfer score does not reproduce: ' + key)
                elif value != held[key]:
                    raise ValueError('Transfer does not reproduce: ' + key)
            fit_bits = prior.bits(fit['recovered']) / len(fit['recovered'])
            if abs(fit_bits - fit['bits_per_letter']) > 1e-10:
                raise ValueError('Fit score does not reproduce')
            rows[language] = dict(fit_excess=fit_bits - frozen['entropy'][language],
                                  transfer_excess=None if replay['bits_per_letter'] is None else replay['bits_per_letter'] - frozen['entropy'][language],
                                  coverage=replay['token_coverage'], cap_hit=any(fit['cap_hits'].values()))
        for outcome in (r for r in results['outcomes'] if r['id'] == ident):
            candidates = [l for l in priors if l != gold['language']] if outcome['kind'] == 'absent' else None
            if decide(rows, candidates) != outcome['decision']:
                raise ValueError('Decision does not reproduce')
            if outcome['kind'] == 'positive':
                language = gold['language']
                for stage, passage in zip(('fit', 'transfer'), gold['passages']):
                    recovered = records[f'{stage}/{ident}-{language}.json']['result']['recovered']
                    plain = passage['plaintext'].replace(' ', '')
                    actual = edit_distance(recovered, plain) / len(plain)
                    if actual != outcome['scores'][language][stage + '_cer']:
                        raise ValueError('True-language character error does not reproduce')
        verified.append(ident)
    return dict(round=name, released_inputs=len(verified), replayed_keys=len(verified) * len(priors),
                source_passages_reproduced=len(checked_passages), challenge_generation_reproduces=True,
                transfer_reproduces=True, decisions_reproduce=True, true_language_cer_reproduces=True,
                unused_answers_opened=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('round', choices=['rejection-transfer', 'rejection-transfer-v2'])
    print(json.dumps(verify(vars(parser.parse_args())['round']), indent=2))
