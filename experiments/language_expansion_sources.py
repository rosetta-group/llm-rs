"""Pinned Old Czech / Old Occitan sources for the eight-language pilot."""
import argparse
import hashlib
import re
import subprocess
import zipfile

from experiments.language_coverage_sources import (
    ROOT, normalize_source, chunks, exact_take, exclude, audit_passage_overlap,
)
from experiments import language_coverage_sources as old_sources
from experiments.rejection_transfer_v2 import read, seal, exclusions, grams
from voynich.data import digest

OUT = ROOT / 'experiments/language-expansion'
STATE = ROOT / 'artifacts/language-expansion'
RAW = STATE / 'raw'
CZECH = {
    'train': ['diakorp27-1440-1460', 'diakorp39-1350-1400', 'diakorp47-1400', 'diakorp50-1389-1401'],
    'calibration': ['diakorp46-1380-1400', 'diakorp72-1410'],
    'challenge': ['diakorp14-1440-1460', 'diakorp1-1492'],
}
OCCITAN = {
    'train': ['Français_1049', 'Français_13503', 'BmC-34', 'Français_25425'],
    'calibration': ['NAF_6195', 'Français_2232'],
    'challenge': ['NAF_11151', 'Harley_7403'],
}
TITLES = {
    'Français_1049': 'Robert of Sicily; Libre de vicis et de vertutz; Barlam et Josaphas',
    'Français_13503': 'Vida de santa Doucelina', 'BmC-34': 'Roman de Flamenca',
    'Français_25425': 'Chanson de la Croisade contre les Albigeois',
    'NAF_6195': 'Vida de sant Honorat (M)', 'Français_2232': 'Roman de Philomena (P)',
    'NAF_11151': 'Arbitral sentences; Mulomedicina; recepta del vi',
    'Harley_7403': 'Nicodemus; fifteen signs; Cross; dietetics; repentance; doctrinal',
}


def czech_document(name, text):
    metadata = dict(re.findall(r'^# ([^:]+):[ \t]*(.*)$', text, re.M))
    dates = [int(x) for x in re.findall(r'\d{4}', metadata['originDate'])]
    if not dates or min(dates) < 1300 or max(dates) > 1500:
        raise ValueError('Not a selected medieval Czech source: ' + name)
    body = '\n'.join(line for line in text.splitlines() if not line.startswith('#'))
    body = re.sub(r'\[\s*\.\.\.\s*\]', ' ', body)
    return dict(id='czech:' + name, group='czech:' + name,
                title=metadata['title'].strip(), period=metadata['originDate'].strip(), text=body)


def occitan_document(name, text):
    if name == 'Français_1049':
        header = 'Sur le trepas de robert de sicile comte\nde provence'
        if not text.startswith(header):
            raise ValueError('Unexpected COMETA editorial heading')
        text = text[len(header):]
    # Join only explicit line-end hyphenation; keep ordinary word boundaries.
    text = re.sub(r'(?<=\w)-[ \t]*\n[ \t]*(?=\w)', '', text)
    text = re.sub(r'\[\s*\.\.\.\s*\]', ' ', text)
    return dict(id='occitan:' + name, group='occitan:' + name,
                title=TITLES[name], period='medieval (COMETA; manuscript dates not homogenized)', text=text)


def documents():
    out = dict(czech={}, occitan={})
    with zipfile.ZipFile(RAW / 'diakorp.zip') as archive:
        for name in sum(CZECH.values(), []):
            path = 'czech/diakorp/txt/' + name + '.txt'
            out['czech'][name] = czech_document(name, archive.read(path).decode('utf-8'))
    for name in sum(OCCITAN.values(), []):
        out['occitan'][name] = occitan_document(name, (RAW / (name + '.txt')).read_text())
    return out


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for row in read(OUT / 'sources.json')['downloads']:
        path = ROOT / row['path']
        if not path.exists():
            subprocess.run(['curl', '--fail', '-sSL', '--retry', '2', '--max-time', '120', row['url'], '-o', str(path)], check=True)
        if digest(path) != row['sha256']:
            raise ValueError('Download changed: ' + str(path))


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'sources.json').exists():
        raise FileExistsError('Source audit already sealed')
    downloads = read(RAW / 'downloads.json')
    for row in downloads:
        assert digest(ROOT / row['path']) == row['sha256']
    docs = documents()
    prior_parts = read(old_sources.STATE / 'partitions.json')
    # Preserve every original language model's exact data and calibration.
    names = ['latin_broad', 'german_broad', 'old_french', 'english', 'italian', 'catalan']
    priors = {n: prior_parts['priors'][n] for n in names}
    old = old_sources.language_id.texts()
    references = [dict(id=f'{l}:{which}:{i}', document=f'{l}:{which}:{i}', group=l, text=t)
                  for l, splits in old.items() for which in (0, 1) for i, t in enumerate(splits[which])]
    # Whole previous historical collections: conservative reference for foreign quotations.
    old_docs = [*old_sources.latin_documents(), *old_sources.rem_documents(old_sources.RAW/'ReM-v2.1_tei.zip'),
                *old_sources.catalan_documents(old_sources.RAW/'LlibredelsFets_60000corrected.txt')]
    references += [r for d in old_docs for r in chunks(dict(d, id='previous:' + d['id']))]
    _, released_grams, exclusion_inputs = exclusions()
    extra = ROOT/'artifacts/rejection-transfer-v2/released-answers.json'
    exclusion_inputs[str(extra.relative_to(ROOT))] = digest(extra)
    for a in read(extra).values():
        for p in a['passages']:
            released_grams.update(grams(normalize_source(p['plaintext'])))
    for pair in prior_parts['passages'].values():
        for p in pair:
            released_grams.update(grams(' '.join(r['text'] for r in p['rows'])))
    exclusion_inputs[str((old_sources.STATE/'partitions.json').relative_to(ROOT))] = digest(old_sources.STATE/'partitions.json')
    pools, stats = {}, {}
    for language, roles in [('czech', CZECH), ('occitan', OCCITAN)]:
        pools[language] = {}
        stats[language] = {}
        seen = set()
        for role, names in roles.items():
            pools[language][role] = []
            for name in names:
                doc = docs[language][name]
                assert doc['group'] not in seen
                seen.add(doc['group'])
                rows = list(chunks(doc))
                kept, removed, tail = [], [], []
                for row in rows:
                    candidate = ' '.join(tail + row['text'].split())
                    if grams(candidate) & released_grams:
                        removed.append(row['id'])
                    else:
                        kept.append(row)
                        tail = candidate.split()[-7:]
                pools[language][role].extend(kept)
                stats[language][name] = dict(role=role, title=doc['title'], period=doc['period'],
                    letters=sum(len(r['text'].replace(' ', '')) for r in rows), removed_released=removed)
    for pool in pools.values():
        references += pool['train'] + pool['calibration']
    passages, removals = {}, {}
    for language, roles in [('czech', CZECH), ('occitan', OCCITAN)]:
        pool = pools[language]
        train = []
        for name in roles['train']:
            train += exact_take([r for r in pool['train'] if r['document'] == language+':'+name], 100000)
        calibration, removed = exclude(pool['calibration'], pool['train'])
        cal = []
        for name in roles['calibration']:
            cal += exact_take([r for r in calibration if r['document'] == language+':'+name], 10000)
        priors[language] = dict(train=train, calibration=cal)
        challenge, removed_challenge = exclude(pool['challenge'], references)
        pair = []
        for name in roles['challenge']:
            rows = [r for r in challenge if r['document'] == language+':'+name]
            if pair:
                rows, _ = exclude(rows, pair[0]['rows'])
            selected = exact_take(rows, 5200)
            ids = {r['id'] for r in selected}
            pair.append(dict(document=language+':'+name, plaintext=''.join(r['text'] for r in selected),
                             rows=[r for r in rows if r['id'] in ids]))
        passages[language] = pair
        removals[language] = dict(calibration=removed, challenge=removed_challenge)
    overlap = audit_passage_overlap(passages, references)
    for language, pair in passages.items():
        for i, passage in enumerate(pair):
            assert not grams(' '.join(r['text'] for r in passage['rows'])) & released_grams
    seal(STATE/'partitions.json', dict(priors=priors, passages=passages))
    seal(OUT/'sources.json', dict(downloads=downloads, statistics=stats, removed_overlap_chunks=removals,
        prior_letters=400000, calibration_letters=20000, final_overlap_counts=overlap,
        partitions_sha256=digest(STATE/'partitions.json'), exclusion_inputs=exclusion_inputs,
        roles=dict(czech=CZECH, occitan=OCCITAN),
        attribution=dict(czech='Kučera and Stluka (2011), DIAKORP v5; Pettersson and Megyesi (2018), HistCorp. CC BY-NC-SA 4.0.',
                         occitan='Marinus Wiedner (2025), COMETA v1, Zenodo 15300719. CC BY 4.0.'),
        limitations=['DIAKORP is transcribed historical language, not diplomatic spelling despite HistCorp header.',
                    'Czech accents collapse in the unchanged cipher alphabet; j→i, k→c, w→uu.',
                    'COMETA manually corrected HTR may retain transcription errors and untagged Latin quotations.',
                    'Eight-language new-source pilot; no fresh retention test of the six earlier languages.']))
    print({l: {r: sum(len(x['text'].replace(' ', '')) for x in p[r]) for r in p} for l,p in pools.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit','download'])
    globals()[parser.parse_args().command]()
