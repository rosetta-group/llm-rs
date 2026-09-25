"""Read-only model, generator, transfer and decision replay after the pilot."""
import contextlib
import hashlib
import io
import json
import random

from experiments.language_expansion import MODELS, OUT, ROOT, STATE, read, replay, verify
from experiments.joint_development import load_vendor
from experiments.rejection_transfer_v2 import encrypt, seal, exclusions, grams
from voynich.data import digest
from voynich.decipher import ALPHABET
from voynich.decipher import normalize
from voynich.description_length import CharacterPrior


def main():
    f = verify()
    partitions = read(STATE / 'partitions.json')
    from experiments.language_expansion_sources import documents, chunks
    raw_rows = {r['id']: r for language in documents().values() for d in language.values() for r in chunks(d)}
    source_rows_checked = 0
    for name in ('czech', 'occitan'):
        for role in ('train', 'calibration'):
            for row in partitions['priors'][name][role]:
                assert raw_rows[row['id']]['text'].replace(' ', '').startswith(row['text'])
                source_rows_checked += 1
        for passage in partitions['passages'][name]:
            assert ''.join(r['text'].replace(' ', '') for r in passage['rows'])[:5200] == passage['plaintext']
            for row in passage['rows']:
                assert row == raw_rows[row['id']]
                source_rows_checked += 1
    metadata_path = STATE/'raw/histcorp-metadata.json'
    entry = next(x for x in read(metadata_path)['files'] if x['key'] == 'czech-diakorp-txt.zip')
    publisher_md5 = hashlib.md5((STATE/'raw/diakorp.zip').read_bytes()).hexdigest()
    assert entry['checksum'] == 'md5:' + publisher_md5
    checked = []
    groups = set()
    for model in MODELS:
        data = partitions['priors'][model]
        texts = [r['text'] for r in data['train']]
        assert sum(map(len, texts)) == 400_000
        assert set(''.join(texts)) <= set(ALPHABET)
        p = CharacterPrior.fit(texts)
        sha = hashlib.sha256(b''.join(x.tobytes() for x in p.probabilities)).hexdigest()
        assert sha == f['prior_probability_sha256'][model]
        held = ''.join(r['text'] for r in data['calibration'])
        assert len(held) == 20_000
        assert abs(p.bits(held) / len(held) - f['entropy'][model]) < 1e-12
        groups.update(r['group'] for role in ('train','calibration') for r in data[role])
        checked.append(model)
    for pair in partitions['passages'].values():
        pair_groups = [{r['group'] for r in passage['rows']} for passage in pair]
        assert not pair_groups[0] & pair_groups[1]
        assert not (pair_groups[0] | pair_groups[1]) & groups
        assert all(len(p['plaintext']) == 5200 for p in pair)
    assert all(n == 0 for n in read(OUT / 'sources.json')['final_overlap_counts'].values())
    # Additional post-run check against previously released challenge text.
    _, prior_grams, exclusion_inputs = exclusions()
    earlier = ROOT / 'artifacts/rejection-transfer-v2/released-answers.json'
    exclusion_inputs[str(earlier.relative_to(ROOT))] = digest(earlier)
    for answer in read(earlier).values():
        for passage in answer['passages']:
            prior_grams.update(grams(normalize(passage['plaintext'])))
    previous = ROOT / 'artifacts/language-coverage/partitions.json'
    exclusion_inputs[str(previous.relative_to(ROOT))] = digest(previous)
    for pair in read(previous)['passages'].values():
        for passage in pair:
            prior_grams.update(grams(' '.join(r['text'] for r in passage['rows'])))
    released_overlap = {}
    for language, pair in partitions['passages'].items():
        for i, passage in enumerate(pair):
            released_overlap[f'{language}:{i}'] = len(grams(' '.join(r['text'] for r in passage['rows'])) & prior_grams)
    assert not any(released_overlap.values())
    # Report without selecting replacement passages.
    generated = 0
    plaintext_diagnostics = []
    with contextlib.redirect_stdout(io.StringIO()):
        vendor = load_vendor()
    for answer in read(STATE / 'evaluator-only/answers.json'):
        letters = list(ALPHABET)
        random.Random(answer['key_seed']).shuffle(letters)
        key = dict(zip(ALPHABET, letters))
        for stage, passage, seed in zip(('fit', 'transfer'), answer['passages'], answer['encoder_seeds']):
            tokens = encrypt(vendor, passage['plaintext'], key, seed)
            assert ' '.join(tokens) == read(STATE / 'public' / f"{answer['id']}-{stage}.json")['ciphertext']
            generated += 1
        # Post-run explanation only: distinguish model mismatch from recovery errors.
        model_scores = {}
        for model in MODELS:
            p = CharacterPrior.load(STATE / 'priors' / f'{model}.npz')
            bpc = [p.bits(passage['plaintext']) / 5200 for passage in answer['passages']]
            model_scores[model] = dict(plaintext_bits_per_letter=bpc,
                                      plaintext_excess=[v-f['entropy'][model] for v in bpc])
        plaintext_diagnostics.append(dict(id=answer['id'], language=answer['language'], scores=model_scores))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        replay()
    results = dict(publisher_diakorp_md5=publisher_md5, publisher_metadata_sha256=digest(metadata_path), source_rows_reextracted=source_rows_checked, models_rebuilt=checked, calibration_reproduces=True, source_groups_disjoint=True,
                   generator_passages_replayed=generated, **json.loads(output.getvalue()),
                   additional_released_overlap_counts=released_overlap, exclusion_inputs=exclusion_inputs,
                   freeze_sha256=digest(OUT / 'freeze.json'), results_sha256=digest(OUT / 'results.json'))
    seal(OUT / 'audit.json', results)
    seal(OUT / 'plaintext-diagnostics.json', dict(post_run_diagnostic_only=True,
         used_for_model_or_threshold_selection=False, rows=plaintext_diagnostics))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
