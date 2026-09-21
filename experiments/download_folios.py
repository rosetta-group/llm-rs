"""Archive Yale's complete scan set and the Voynich.nu panel-to-scan index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/folios"
CATALOGUE = "https://voynich.nu/folios.html"
MANIFEST = "https://collections.library.yale.edu/manifests/2002046"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "curl/8.7.1"})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def catalogue():
    html = fetch(CATALOGUE)
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    for a in soup.select('a[href]'):
        img = a.find("img")
        if img is None or not re.match(r"q\d+/.*#f", a["href"]):
            continue
        # Filename disambiguates the two f101v panels sharing one HTML anchor.
        stem = Path(img["src"]).stem.replace("_th", "")
        folio = re.sub(r"^f0+(\d)", r"f\1", stem)
        entries.append({"folio_id": folio, "catalogue_url": urljoin(CATALOGUE, a["href"]),
                        "thumbnail_url": urljoin(CATALOGUE, img["src"])})
    urls = sorted({e["catalogue_url"].split("#")[0] for e in entries})
    def quire(url):
        raw = fetch(url)
        page = BeautifulSoup(raw, "html.parser")
        mapping, active = {}, None
        for tag in page.find_all(True):
            if tag.get("id", "").startswith("f") and tag.name == "th":
                active = tag["id"]
            if tag.name == "a" and "child_oid=" in tag.get("href", "") and active:
                oid = parse_qs(urlparse(tag["href"]).query)["child_oid"][0]
                mapping.setdefault(active, []).append(oid)
        return url, mapping, digest(raw)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pages = list(pool.map(quire, urls))
    mappings = {u: m for u, m, _ in pages}
    for entry in entries:
        url, anchor = entry["catalogue_url"].split("#")
        entry["yale_image_ids"] = list(dict.fromkeys(mappings[url].get(anchor, [])))
        if not entry["yale_image_ids"]:
            raise ValueError(f"Missing Yale link: {entry}")
    assert len({e["folio_id"] for e in entries}) == len(entries)
    write_json(DATA / "sources/catalogue-index.json", {
        "url": CATALOGUE, "html_sha256": digest(html),
        "quire_sources": [{"url": u, "html_sha256": h} for u, _, h in pages],
        "panels": entries})
    return entries


def download(long_edge=1800):
    source = DATA / "sources/yale-manifest.json"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(fetch(MANIFEST))
    manifest = json.loads(source.read_text())
    image_dir = DATA / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    old_path = DATA / "images.json"
    old = {r["image_id"]: r for r in json.loads(old_path.read_text())} if old_path.exists() else {}

    def one(item):
        index, canvas = item
        body = canvas["items"][0]["items"][0]["body"]
        service = body["service"][0]["@id"]
        oid = service.rsplit("/", 1)[1]
        url = f"{service}/full/!{long_edge},{long_edge}/0/default.jpg"
        path = image_dir / f"yale_{oid}.jpg"
        data = path.read_bytes() if path.exists() else fetch(url)
        if oid in old and (digest(data) != old[oid]["sha256"] or url != old[oid]["download_url"]):
            raise ValueError(f"Archived image changed or resolution differs: {oid}")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            if image.format != "JPEG" or max(image.size) != long_edge:
                raise ValueError(f"Invalid image response: {oid}")
        if not path.exists():
            temporary = path.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.replace(path)
        label = next(iter(canvas["label"].values()))[0]
        print(f"{index + 1:3}/{len(manifest['items'])} {label}", flush=True)
        return {"image_id": oid, "scan_index": index, "label": label,
                "path": str(path.relative_to(ROOT)), "download_url": url,
                "original_url": body["id"], "width": width, "height": height,
                "original_width": body["width"], "original_height": body["height"],
                "bytes": len(data), "sha256": digest(data)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, enumerate(manifest["items"])))
    write_json(old_path, records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue-only", action="store_true")
    args = parser.parse_args()
    if not (DATA / "sources/catalogue-index.json").exists():
        panels = catalogue()
        print(f"Indexed {len(panels)} catalogue panels", flush=True)
    if not args.catalogue_only:
        images = download()
        print(f"Archived {len(images)} images; {sum(r['bytes'] for r in images) / 1e6:.1f} MB")
