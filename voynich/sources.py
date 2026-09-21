"""Fetch pinned public data; never execute downloaded code."""

import hashlib
import json
import urllib.request
from pathlib import Path

from .data import digest, write_json


def fetch_sources(manifest="experiments/sources.json", output="artifacts/sources"):
    sources = json.loads(Path(manifest).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if Path(name).name != name:
            raise ValueError("Source names must be filenames")
        path = output/name
        if path.exists():
            if digest(path) != source["sha256"]:
                raise ValueError(f"Local source changed: {path}")
            continue
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError(f"Downloaded source changed: {name}")
        path.write_bytes(data)
    return {"sources": len(sources), "output": str(output)}


def import_control(source, output, lines_per_page=29):
    """Chronological blocks with one unused block between each split."""
    if lines_per_page < 1:
        raise ValueError("lines_per_page must be positive")
    lines = [".".join(line.split()) for line in Path(source).read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.startswith("#")]
    if any(ord(char) >= 256 for line in lines for char in line):
        raise ValueError("Control must use transcription characters in range 0..255")
    blocks = ["\n".join(lines[i:i+lines_per_page]) for i in range(0, len(lines), lines_per_page)]
    n_train, n_val = int(.7*len(blocks)), max(2, int(.15*len(blocks)))
    if n_train + n_val + 2 >= len(blocks):
        raise ValueError("Control needs more blocks for train, validation, test, and gaps")
    guards = [n_train, n_train+n_val+1]
    docs = []
    for i, text in enumerate(blocks):
        if i in guards:
            continue
        split = "train" if i < n_train else "validation" if i < guards[1] else "test"
        docs.append(dict(page=f"block{i:04d}", folio=str(i), quire=str(i//4),
                         currier="unknown", section="control", hand="unknown", text=text,
                         split=split, loci=[]))
    manifest = dict(version=1, source_sha256=digest(source), group="synthetic_block",
                    assignments={d["page"]: d["split"] for d in docs}, unused_blocks=guards)
    audit = dict(source=str(source), source_sha256=digest(source), lines_per_page=lines_per_page,
                 normalization="Drop # headers and empty lines; whitespace becomes dots",
                 splits={s: sum(d["split"] == s for d in docs) for s in ("train", "validation", "test")},
                 limitation="One published sample; blocks are not independent generator realizations")
    output = Path(output)
    write_json(output/"documents.json", docs)
    write_json(output/"manifest.json", manifest)
    write_json(output/"audit.json", audit)
    return audit
