"""Small freeze/verify helpers for auditable, non-overwriting repair experiments."""

import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def freeze(out, files, sources):
    out.mkdir(parents=True, exist_ok=True)
    record = {"files": {str(p): digest(p) for p in sorted(set(map(str, files)))},
              "sources": {str(p): digest(p) for p in sorted(set(map(str, sources)))}}
    save(out/'freeze.json', record)
    print('Freeze written; commit code, protocol and freeze before running.', flush=True)


def verify(out):
    record = json.loads((out/'freeze.json').read_text())
    bad = [p for p, h in {**record['files'], **record['sources']}.items() if digest(p) != h]
    if bad:
        raise ValueError(f'Frozen inputs changed: {bad}')
    frozen = [*record['files'], str(out/'freeze.json')]
    subprocess.run(['git', 'ls-files', '--error-unmatch', *frozen], check=True, capture_output=True)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *frozen], check=True,
                   capture_output=True)
    return record
