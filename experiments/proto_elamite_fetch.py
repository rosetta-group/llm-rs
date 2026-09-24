"""Download Proto-Elamite and proto-cuneiform records from the CDLI search API.

    python -m experiments.proto_elamite_fetch

One page of 100 records per request, at most one request per second, following the API's
``Link: rel="next"`` header. Pages go to artifacts/proto-elamite-sources/<period>/page-NNN.json.
Existing pages are kept, so an interrupted run resumes. CDLI terms: re-use under academic practice
with credit to CDLI.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("artifacts/proto-elamite-sources")
PERIODS = {"proto-elamite": "Proto-Elamite", "uruk-iii": "Uruk III", "uruk-iv": "Uruk IV"}
HEADERS = {"Accept": "application/json", "User-Agent": "llm-rs research (non-profit; 1 req/s)"}


def fetch(url):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read()), response.headers.get_all("Link") or []
        except Exception as error:  # noqa: BLE001
            print("retry", url, error, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def next_link(links):
    for link in links:
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if match:
            return match.group(1)
    return None


def main():
    for folder, period in PERIODS.items():
        out = ROOT / folder
        out.mkdir(parents=True, exist_ok=True)
        url = f"https://cdli.earth/search?period={urllib.parse.quote_plus(period)}&limit=100&page=1"
        page = 1
        while url:
            path = out / f"page-{page:03d}.json"
            started = time.time()
            records, links = fetch(url)
            path.write_text(json.dumps(records, ensure_ascii=False))
            print(folder, page, len(records), flush=True)
            url = next_link(links) if records else None
            page += 1
            time.sleep(max(0.0, 1.0 - (time.time() - started)))


if __name__ == "__main__":
    main()
