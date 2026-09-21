"""Pin original Italian verse that is not Dante: Petrarca's Canzoniere from Wikisource.

Mirrors experiments/historical_sources.py. Every fifth poem in the collection's order is
development; the rest is training. Dante is never fetched. The prior trained on this text
is meant for the historical-verse gap seen in experiments/joint-recovery/REPORT.md.

    python -m experiments.verse_sources prepare   # fetch, pin revisions and hashes, snapshot
    python -m experiments.verse_sources verify    # check hashes; return the poems
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup
from voynich.decipher import normalize

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'artifacts/historical-verse'
OUT = ROOT / 'experiments/verse-prior'
MANIFEST = OUT / 'sources.json'
INDEX = 'Canzoniere (Rerum vulgarium fragmenta)'
AGENT = {'User-Agent': 'VoynichMethodsBenchmark/1.0 (public-domain Italian research)'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(title, path):
    url = 'https://it.wikisource.org/wiki/' + urllib.parse.quote(title.replace(' ', '_'), safe='/()')
    if not path.exists():
        request = urllib.request.Request(url, headers=AGENT)
        for attempt in range(4):
            try:
                time.sleep(1)
                path.write_bytes(urllib.request.urlopen(request, timeout=45).read())
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 3:
                    raise
                print('Source rate limit; waiting 60 seconds', flush=True)
                time.sleep(60)
    return url, path.read_text()


def extract(html):
    soup = BeautifulSoup(html, 'html.parser')
    root = soup.select_one('#box_esterno')
    if root is None:
        raise ValueError('Missing literary text container')
    for node in root.select('script, style, table, sup, .ws-noexport, .noprint, .references, .mw-editsection, h1, h2, h3, h4, h5'):
        node.decompose()
    poem = root.select_one('div.poem')
    if poem is None:
        # One page (poem 362) carries the verse directly in the container; drop navigation first.
        for node in root.select('.ws-noexport, .noprint, .intest-normal, .intest-right, .intest-sezione, .barraCapitolo, .box_sottopagina, .AltraVersione'):
            node.decompose()
        poem = root
    lines = [' '.join(line.split()) for line in poem.get_text('\n').split('\n')]
    lines = [line for line in lines if line and not re.fullmatch(r'\d+', line)]
    if len(lines) < 4:
        raise ValueError('Too few verse lines')
    return lines


def poem_titles():
    _, html = download(INDEX, STATE / 'raw' / 'index.html')
    soup = BeautifulSoup(html, 'html.parser')
    prefix = '/wiki/' + urllib.parse.quote(INDEX.replace(' ', '_'), safe='()') + '/'
    titles = []
    for a in soup.select('#box_esterno a[href]'):
        href = a.get('href', '')
        if href.startswith(prefix) and 'redlink' not in href:
            title = urllib.parse.unquote(href[len('/wiki/'):]).replace('_', ' ')
            if title not in titles:
                titles.append(title)
    if len(titles) < 300:
        raise ValueError(f'Unexpected index size {len(titles)}')
    return titles


def prepare():
    if MANIFEST.exists():
        raise FileExistsError('Sources already pinned; use verify')
    (STATE / 'raw').mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    titles = poem_titles()
    rows = []
    for number, title in enumerate(titles, 1):
        path = STATE / 'raw' / f'canzoniere-{number:03}.html'
        url, html = download(title, path)
        revision = int(re.search(r'"wgRevisionId":(\d+)', html)[1])
        rows.append(dict(id=f'canzoniere-{number:03}', work='canzoniere', poem=number, title=title,
                         split='dev' if number % 5 == 0 else 'train', url=url, revision=revision,
                         permanent_url=f'https://it.wikisource.org/w/index.php?oldid={revision}',
                         raw_path=str(path.relative_to(ROOT)), raw_sha256=sha(path), lines=extract(html)))
        if number % 50 == 0:
            print(f'Pinned {number}/{len(titles)} poems', flush=True)
    texts = [dict(id=r['id'], work=r['work'], split=r['split'], lines=r['lines']) for r in rows]
    target = STATE / 'texts.json'
    target.write_text(json.dumps(texts, ensure_ascii=False, indent=2) + '\n')
    metadata = dict(sources=[{k: v for k, v in r.items() if k != 'lines'} for r in rows],
                    texts_path=str(target.relative_to(ROOT)), texts_sha256=sha(target),
                    attribution='Francesco Petrarca, Canzoniere (Rerum vulgarium fragmenta), 14th century; Wikisource contributors.',
                    license='Public-domain original; Wikisource transcription CC BY-SA, see each page and its history.',
                    split='Every fifth poem in collection order is development; all other poems training. Dante excluded.',
                    caveat='Edited transcription with modern punctuation and elision marks; line numbers stripped; not a diplomatic edition.',
                    words={split: sum(len(normalize(' '.join(r['lines'])).split()) for r in rows if r['split'] == split) for split in ('train', 'dev')})
    MANIFEST.write_text(json.dumps(metadata, indent=2) + '\n')
    snapshot()
    verify()


def snapshot():
    target = OUT / 'historical-verse.tar.gz'
    with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for path in sorted(STATE.rglob('*')):
                if not path.is_file():
                    continue
                info = archive.gettarinfo(str(path), arcname=str(path.relative_to(STATE)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                with path.open('rb') as source:
                    archive.addfile(info, source)


def restore():
    if STATE.exists():
        return verify()
    STATE.mkdir(parents=True)
    with tarfile.open(OUT / 'historical-verse.tar.gz', 'r:gz') as archive:
        archive.extractall(STATE, filter='data')
    return verify()


def verify():
    manifest = json.loads(MANIFEST.read_text())
    if sha(ROOT / manifest['texts_path']) != manifest['texts_sha256']:
        raise ValueError('Verse text drift')
    for row in manifest['sources']:
        if sha(ROOT / row['raw_path']) != row['raw_sha256']:
            raise ValueError('Verse raw source drift')
    rows = json.loads((ROOT / manifest['texts_path']).read_text())
    print({split: sum(len(normalize(' '.join(r['lines'])).split()) for r in rows if r['split'] == split) for split in ('train', 'dev')})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'restore'])
    globals()[parser.parse_args().command]()
