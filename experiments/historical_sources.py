"""Pin original Italian prose. Split whole tales before fitting any model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import tarfile
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup
from voynich.decipher import normalize

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'artifacts/historical-prose'
MANIFEST = ROOT / 'experiments/standard-decipherment/sources.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roman(n):
    result = ''
    for value, token in [(100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]:
        while n >= value:
            result += token
            n -= value
    return result


def extract(html):
    soup = BeautifulSoup(html, 'html.parser')
    root = soup.select_one('#box_esterno')
    if root is None:
        raise ValueError('Missing literary text container')
    for node in root.select('script, style, table, sup, .ws-noexport, .noprint, .references, .mw-editsection, h1, h2, h3, h4, h5, .centertext'):
        node.decompose()
    for br in root.select('br'):
        br.replace_with(' ')
    # Keep prose paragraphs only. Joining inline spans without added spaces preserves drop capitals.
    paragraphs = [' '.join(p.get_text().split()) for p in root.select('p')]
    paragraphs = [p for p in paragraphs if len(normalize(p).split()) >= 8]
    if not paragraphs:
        raise ValueError('No prose paragraphs extracted')
    return paragraphs


def fetch(item):
    work, number, title = item
    path = STATE / 'raw' / f'{work}-{number:03}.html'
    url = 'https://it.wikisource.org/wiki/' + urllib.parse.quote(title.replace(' ', '_'), safe='/')
    if not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'VoynichMethodsBenchmark/1.0 (public-domain Italian research)'})
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
    html = path.read_text()
    revision = int(re.search(r'"wgRevisionId":(\d+)', html)[1])
    paragraphs = extract(html)
    return dict(id=f'{work}-{number:03}', work=work, tale=number, split='dev' if number % 5 == 0 else 'train',
                url=url, revision=revision, permanent_url=f'https://it.wikisource.org/w/index.php?oldid={revision}',
                raw_path=str(path.relative_to(ROOT)), raw_sha256=sha(path), paragraphs=paragraphs)


def prepare():
    if MANIFEST.exists():
        raise FileExistsError('Sources already pinned; use verify')
    (STATE / 'raw').mkdir(parents=True, exist_ok=True)
    ordinals = ['prima', 'seconda', 'terza', 'quarta', 'quinta', 'sesta', 'settima', 'ottava', 'nona', 'decima']
    items = [('novellino', n, f'Novellino/{roman(n)}') for n in range(1, 101)]
    items += [('decameron', 10 * d + n + 1, f'Decameron/Giornata {day}/Novella {novel}')
              for d, day in enumerate(ordinals[:2]) for n, novel in enumerate(ordinals)]
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for row in pool.map(fetch, items):
            rows.append(row)
            if len(rows) % 20 == 0:
                print(f'Pinned {len(rows)}/120 tales', flush=True)
    texts = [dict(id=r['id'], work=r['work'], split=r['split'], paragraphs=r['paragraphs']) for r in rows]
    target = STATE / 'texts.json'
    target.write_text(json.dumps(texts, ensure_ascii=False, indent=2) + '\n')
    metadata = dict(sources=[{k:v for k,v in r.items() if k != 'paragraphs'} for r in rows],
                    texts_path=str(target.relative_to(ROOT)), texts_sha256=sha(target),
                    attribution='Anonymous, Novellino (13th century); Giovanni Boccaccio, Decameron (14th century); Wikisource contributors.',
                    license='Public-domain originals; Wikisource transcription CC BY-SA, see each page and its history.',
                    split='Every fifth whole tale is development; all other tales training. Dante excluded.',
                    caveat='Edited transcriptions, not manuscript diplomatic editions. Orthography and segmentation reflect editions; some pages are not fully proofread.')
    MANIFEST.write_text(json.dumps(metadata, indent=2) + '\n')
    snapshot()
    verify()


def snapshot():
    target = MANIFEST.parent / 'historical-prose.tar.gz'
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
    with tarfile.open(MANIFEST.parent / 'historical-prose.tar.gz', 'r:gz') as archive:
        archive.extractall(STATE, filter='data')
    return verify()


def verify():
    manifest = json.loads(MANIFEST.read_text())
    if sha(ROOT / manifest['texts_path']) != manifest['texts_sha256']:
        raise ValueError('Historical text drift')
    for row in manifest['sources']:
        if sha(ROOT / row['raw_path']) != row['raw_sha256']:
            raise ValueError('Historical raw source drift')
    rows = json.loads((ROOT / manifest['texts_path']).read_text())
    print({split: sum(len(normalize(' '.join(r['paragraphs'])).split()) for r in rows if r['split']==split)
           for split in ('train', 'dev')})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'restore'])
    globals()[parser.parse_args().command]()
