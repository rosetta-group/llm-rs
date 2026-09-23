"""Word forms by language from TLHdig Beta 0.3 (Hittite tablet collections, CC BY 4.0).

Each ``<w>`` element carries a normalised transcription in ``trans``. Its language is the word's
own ``lg`` if present, else that of the latest ``<lb>`` line. Words whose transcription contains
capitals or digits include logograms (``ḪUR.SAGna``) and are skipped, as are partly broken words
marked with ``[`` or ``…``.
"""

import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path("artifacts/linear-a-sources/tlhdig/TLHbasisONLINE25_1_ZENODO_Beta_03")
CACHE = Path("artifacts/linear-a-sources/tlhdig/forms.json")
LANGUAGES = ("Hit", "Luw", "Pal", "Hur", "Hat", "Akk")

_TOKEN = re.compile(r'<lb\b[^>]*?\blg="([^"]*)"[^>]*/>|<w\b([^>]*)>')
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def extract(root=ROOT):
    counts = {language: collections.Counter() for language in LANGUAGES}
    for path in glob.glob(str(root / "**" / "*.xml"), recursive=True):
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        line_language = None
        for match in _TOKEN.finditer(text):
            if match.group(1) is not None:
                line_language = match.group(1)
                continue
            attrs = dict(_ATTR.findall(match.group(2)))
            language = attrs.get("lg") or line_language
            form = attrs.get("trans", "")
            if language not in counts or not form:
                continue
            if any(ch.isupper() or ch.isdigit() for ch in form) or re.search(r"[\[\]…?x~%]", form):
                continue
            counts[language][form] += 1
    return counts


def forms(refresh=False):
    if CACHE.exists() and not refresh:
        return {k: collections.Counter(v) for k, v in json.loads(CACHE.read_text()).items()}
    counts = extract()
    CACHE.write_text(json.dumps(counts, ensure_ascii=False))
    return counts
