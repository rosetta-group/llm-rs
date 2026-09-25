"""Auditable historical-language source extraction and disjoint pilot splits."""
import argparse
from collections import Counter
import hashlib
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from experiments import language_id
from experiments.rejection_transfer_v2 import ROOT, read, seal, grams, fingerprint
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import normalize

OUT = ROOT / 'experiments/language-coverage'
STATE = ROOT / 'artifacts/language-coverage'
RAW = STATE / 'raw'
BUDGET = 400_000
NS = {'t': 'http://www.tei-c.org/ns/1.0'}


def download():
    """Restore pinned downloads using the committed manifest, never mutable HEAD."""
    manifest = read(OUT / 'sources.json')['downloads']
    RAW.mkdir(parents=True, exist_ok=True)
    for record in manifest:
        target = ROOT / record['path']
        if not target.exists():
            data = urllib.request.urlopen(record['url'], timeout=60).read()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('Upstream download changed')
            target.write_bytes(data)
        if digest(target) != record['sha256']:
            raise ValueError('Local download changed')
    if not (RAW / 'downloads.json').exists():
        seal(RAW / 'downloads.json', manifest)


def rem_documents(path):
    """Use publisher's normalized surface forms, never dictionary lemmas."""
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.xml'):
                continue
            root = ET.fromstring(archive.read(name))
            genre = root.find('.//t:classCode', NS)
            if genre is None or genre.text != 'P':
                continue
            ident = name.rsplit('/', 1)[-1][:-4]
            words = []
            for word in root.findall('.//t:body//t:w', NS):
                form = word.get('norm')
                if form is None:
                    raise ValueError('Missing normalized surface form: ' + ident)
                words.append(form)
            yield dict(id=ident, group=re.match(r'M\d+', ident)[0],
                       title=root.find('.//t:title', NS).text,
                       text=' '.join(words), representation='ReM norm attribute, not lemma')


def catalan_documents(path):
    """Folio sides stay together; strip annotation, title and folio markers."""
    docs, current = {}, None
    for line in path.read_text().splitlines()[1:]:
        fields = line.split('\t')
        if len(fields) < 3:
            if line.strip():
                raise ValueError('Malformed HisCat row')
            continue
        form, _, pos = fields[:3]
        if pos == 'FOL':
            match = re.fullmatch(r'FOL(\d+)[rv]', form)
            if not match:
                raise ValueError('Unexpected folio: ' + form)
            current = int(match[1])
            docs.setdefault(current, [])
        elif current is not None and pos != '§':
            docs[current].append(form)
    for number, words in sorted(docs.items()):
        yield dict(id=f'f{number:04}', group=f'f{number:04}',
                   title='Llibre dels Fets', text=' '.join(words))


def latin_documents():
    docs = {}
    for split in ('train', 'dev', 'test'):
        for row in conllu_sentences(RAW / f'la_llct-ud-{split}.conllu'):
            ident = re.search(r"document_id='([^']+)'", row['metadata']['reference'])[1]
            # Rebuild whole documents across UD splits before assigning roles.
            span = int(re.search(r"span='(\d+)'", row['metadata']['reference'])[1])
            docs.setdefault(ident, []).append((span, row['text']))
    for ident, pieces in sorted(docs.items()):
        # Two source spans contain multiple adjacent sentences; stable ordering
        # preserves their file order instead of sorting those sentences by text.
        yield dict(id=ident, group=ident, title='Tuscan charter ' + ident,
                   text=' '.join(t for _, t in sorted(pieces, key=lambda p: p[0])))


def normalize_source(text):
    # Explicit historical glyph expansions; shared across all new corpora.
    return normalize(text.replace('ſ', 's').replace('ß', 'ss').replace('æ', 'ae').replace('œ', 'oe').replace('ð', 'd'))


def chunks(document, words_per_chunk=80):
    words = normalize_source(document['text']).split()
    for i in range(0, len(words), words_per_chunk):
        text = ' '.join(words[i:i + words_per_chunk])
        if text:
            yield dict(id=f"{document['id']}:{i}", document=document['id'],
                       group=document['group'], text=text)


def exact_take(rows, budget):
    out, remaining = [], budget
    for row in rows:
        if not remaining:
            break
        dense = row['text'].replace(' ', '')[:remaining]
        out.append(dict(row, text=dense))
        remaining -= len(dense)
    if remaining:
        raise ValueError(f'Insufficient training/calibration text: missing {remaining} letters')
    return out


def exclude(rows, reference):
    exact = {fingerprint(r['text']) for r in reference}
    shingles = set().union(*(grams(r['text']) for r in reference))
    # Include windows crossing chunk boundaries inside each document.
    documents = {}
    for row in reference:
        documents.setdefault(row['document'], []).append(row['text'])
    for parts in documents.values():
        shingles.update(grams(' '.join(parts)))
    kept, removed, tail = [], [], []
    for row in rows:
        candidate = ' '.join(tail + row['text'].split())
        if fingerprint(row['text']) in exact or grams(candidate) & shingles:
            removed.append(row['id'])
        else:
            kept.append(row)
            tail = candidate.split()[-7:]
    return kept, removed


def audit_passage_overlap(passages, reference):
    """Final check includes artificial joins between retained challenge chunks."""
    by_document = {}
    for r in reference:
        by_document.setdefault(r['document'], []).append(r['text'])
    shingles = set()
    for parts in by_document.values():
        shingles.update(grams(' '.join(parts)))
    counts = {}
    for language, pair in passages.items():
        for i, passage in enumerate(pair):
            overlap = grams(' '.join(r['text'] for r in passage['rows'])) & shingles
            counts[f'{language}:{i}'] = len(overlap)
            if overlap:
                raise ValueError(f'Cross-boundary or cross-language overlap: {language}:{i}: {len(overlap)}')
    return counts


def pair_passages(rows):
    """Exactly 5,200 letters per passage, from distinct source documents."""
    grouped = {}
    for r in rows:
        grouped.setdefault(r['document'], []).append(r)
    out = []
    for doc, parts in grouped.items():
        if out and parts[0]['group'] in {r['group'] for r in out[0]['rows']}:
            continue
        if sum(len(r['text'].replace(' ', '')) for r in parts) < 5200:
            continue
        if out:
            parts, _ = exclude(parts, out[0]['rows'])
        if sum(len(r['text'].replace(' ', '')) for r in parts) < 5200:
            continue
        selected = exact_take(parts, 5200)
        out.append(dict(document=doc, plaintext=''.join(r['text'] for r in selected),
                        rows=[r for r in parts if r['id'] in {s['id'] for s in selected}]))
        if len(out) == 2:
            return out
    # Small documents (e.g. folios/charters) require grouped, non-overlapping aggregates.
    out, remaining = [], list(rows)
    for _ in range(2):
        selected = exact_take(remaining, 5200)
        ids = {r['id'] for r in selected}
        used_docs = {r['document'] for r in selected}
        used_groups = {r['group'] for r in selected}
        originals = [r for r in remaining if r['id'] in ids]
        out.append(dict(document=sorted(used_docs), plaintext=''.join(r['text'] for r in selected), rows=originals))
        remaining, _ = exclude([r for r in remaining if r['group'] not in used_groups], originals)
    return out


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'sources.json').exists():
        raise FileExistsError('Source audit already sealed')
    downloads = read(RAW / 'downloads.json')
    for row in downloads:
        if digest(ROOT / row['path']) != row['sha256']:
            raise ValueError('Download changed')
    sources = dict(latin=list(latin_documents()), german=list(rem_documents(RAW / 'ReM-v2.1_tei.zip')),
                   catalan=list(catalan_documents(RAW / 'LlibredelsFets_60000corrected.txt')))
    pools, stats = {}, {}
    for language, docs in sources.items():
        split = dict(train=[], calibration=[], challenge=[])
        for index, doc in enumerate(docs):
            if language == 'catalan':
                fraction = index / len(docs)
                role = 'train' if fraction < .80 else 'calibration' if fraction < .90 else 'challenge'
            else:
                bucket = int(hashlib.sha256(doc['group'].encode()).hexdigest()[:8], 16) % 10
                role = 'challenge' if bucket == 0 else 'calibration' if bucket == 1 else 'train'
            split[role].extend(chunks(doc))
        calibration, removed_cal = exclude(split['calibration'], split['train'])
        challenge, removed_test = exclude(split['challenge'], split['train'] + split['calibration'])
        pair = pair_passages(challenge)
        pools[language] = dict(train=split['train'], calibration=calibration, challenge=challenge, passages=pair)
        unsupported = Counter(c for d in docs for c in unicodedata.normalize('NFD', d['text'].lower())
                              if c.isalpha() and c not in 'abcdefghijklmnopqrstuvwxyzſßæœð')
        stats[language] = dict(documents=len(docs),
            letters={k: sum(len(r['text'].replace(' ', '')) for r in v) for k, v in split.items()},
            retained_calibration_letters=sum(len(r['text'].replace(' ', '')) for r in calibration),
            retained_challenge_letters=sum(len(r['text'].replace(' ', '')) for r in challenge),
            removed_calibration_chunks=removed_cal, removed_challenge_chunks=removed_test,
            challenge_documents=[p['document'] for p in pair], unsupported_alpha=dict(unsupported),
            titles={d['id']: d['title'] for d in docs})
    # Existing training/entropy partitions are reused; all models are refit at equal budget.
    old = language_id.texts()
    priors = {}
    def legacy_rows(language, which):
        return [dict(id=f'{language}:{which}:{i}', document=f'{language}:{which}:{i}', group=language, text=t)
                for i, t in enumerate(old[language][which])]
    for language in language_id.LANGUAGES:
        priors[language] = dict(train=exact_take(legacy_rows(language, 0), BUDGET),
                               calibration=exact_take(legacy_rows(language, 1), 20000))
    for language in ('latin', 'german'):
        priors[language + '_broad'] = dict(
            train=exact_take(legacy_rows(language, 0), BUDGET//2) + exact_take(pools[language]['train'], BUDGET//2),
            calibration=exact_take(legacy_rows(language, 1), 10000) + exact_take(pools[language]['calibration'], 10000))
    priors['catalan'] = dict(train=exact_take(pools['catalan']['train'], BUDGET),
                           calibration=exact_take(pools['catalan']['calibration'], 20000))
    # Check new challenge rows against every legacy training and calibration sentence too.
    references = [r for l in language_id.LANGUAGES for which in (0, 1) for r in legacy_rows(l, which)]
    references += [r for p in pools.values() for kind in ('train', 'calibration') for r in p[kind]]
    for language, pool in pools.items():
        eligible, removed = exclude(pool['challenge'], references)
        pool['passages'] = pair_passages(eligible)
        stats[language]['removed_global_overlap_chunks'] = removed
        stats[language]['challenge_documents'] = [p['document'] for p in pool['passages']]
    overlap_audit = audit_passage_overlap({l: p['passages'] for l, p in pools.items()}, references)
    seal(STATE / 'partitions.json', dict(priors=priors, passages={l: p['passages'] for l, p in pools.items()}))
    seal(OUT / 'sources.json', dict(downloads=downloads, statistics=stats, prior_letters=BUDGET, final_overlap_counts=overlap_audit,
         partitions_sha256=digest(STATE / 'partitions.json'), legacy_sources_sha256=digest(ROOT / 'experiments/language-sources.json'),
         historical_sources_sha256=digest(ROOT / 'experiments/standard-decipherment/sources.json'),
         new_sources=dict(latin=dict(revision='df63d06c5a7788b457f50bf37526146e9225c27d',license='CC BY-SA 4.0',period='774–897',genre='legal charters'),
                          german=dict(doi='10.5281/zenodo.13982324',license='CC BY-SA 4.0',period='1050–1350',genre='prose only'),
                          catalan=dict(doi='10.5281/zenodo.5615759',license='CC BY 4.0',period='13th century',genre='single chronicle'))))
    print({l: {k: v for k, v in s.items() if k not in ('titles','removed_calibration_chunks','removed_challenge_chunks')} for l,s in stats.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit', 'download'])
    globals()[parser.parse_args().command]()
