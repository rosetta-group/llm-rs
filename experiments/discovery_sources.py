"""Fetch or verify the pinned language and decipherment sources."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch(file, root=ROOT):
    path = root / file["path"]
    data = path.read_bytes() if path.exists() else urlopen(file["url"], timeout=120).read()
    if hashlib.sha256(data).hexdigest() != file["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    count = 0
    for name in ("language-sources", "decipherment-sources"):
        manifest = json.loads((ROOT / f"experiments/{name}.json").read_text())
        files = manifest.get("files", []) + [f for s in manifest.get("sources", []) for f in s["files"]]
        for file in files:
            fetch(file)
            count += 1
    print(f"Verified {count} pinned source files")


if __name__ == "__main__":
    main()
