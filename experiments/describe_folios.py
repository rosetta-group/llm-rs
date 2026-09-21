"""Describe every catalogue panel; archive a versioned CSV/JSONL table and gallery."""
import csv
import hashlib
import html
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
from PIL import Image
from experiments.download_folios import DATA, ROOT
from voynich.folio_description import CONFIG, describe


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint():
    paths = ["voynich/folio_description.py", "experiments/describe_folios.py",
             "experiments/folio_requirements.txt", "data/folios/images.json",
             "data/folios/crops.json", "data/folios/sources/catalogue-index.json"]
    info = {"schema_version": 1, "config": CONFIG,
            "files": {p: sha((ROOT / p).read_bytes()) for p in paths},
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ["numpy", "Pillow", "opencv-python-headless"]},
            "python": platform.python_version(), "architecture": platform.machine()}
    return f"{CONFIG['method']}_{sha(canonical(info).encode())[:12]}", info


def panel_records():
    panels = json.loads((DATA / "sources/catalogue-index.json").read_text())["panels"]
    images = {r["image_id"]: r for r in json.loads((DATA / "images.json").read_text())}
    crops = {r["folio_id"]: r for r in json.loads((DATA / "crops.json").read_text())["crops"]}
    if len({p["folio_id"] for p in panels}) != len(panels):
        raise ValueError("Duplicate folio identifier")
    for record in images.values():
        if sha((ROOT / record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError(f"Image hash mismatch: {record['path']}")
    counts = {}
    for panel in panels:
        for oid in panel["yale_image_ids"]:
            counts[oid] = counts.get(oid, 0) + 1
    for panel in panels:
        crop = crops.get(panel["folio_id"])
        shared = any(counts[x] > 1 for x in panel["yale_image_ids"])
        if shared and (crop is None or not crop.get("visually_reviewed")):
            raise ValueError(f"Missing reviewed foldout crop: {panel['folio_id']}")
        oid = crop["image_id"] if crop else panel["yale_image_ids"][0]
        if oid not in panel["yale_image_ids"]:
            raise ValueError("Crop source is not linked from this catalogue panel")
        box = crop["bbox"] if crop else [0, 0, 1, 1]
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Invalid crop rectangle")
        yield panel, images[oid], box


def crop_image(image, box):
    x1, y1, x2, y2 = box
    return image.crop((round(x1 * image.width), round(y1 * image.height),
                       round(x2 * image.width), round(y2 * image.height)))


def gallery(rows, analysis_id):
    entries = []
    for r in rows:
        e = html.escape
        entries.append(f'<tr><td><a href="../../images/yale_{r["image_id"]}.jpg">'
                       f'<img loading="lazy" src="previews/{r["folio_id"]}.jpg" alt="{r["folio_id"]}"></a></td>'
                       f'<td><b>{e(r["folio_id"])}</b><br><small>{e(r["scan_label"])}</small></td>'
                       f'<td>{e(r["layout"])}</td><td>{e(r["description"])}'
                       f'<details><summary>Measurements and rule trace</summary><pre>{e(json.dumps(r, indent=2))}</pre></details></td></tr>')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voynich folio descriptions</title><style>
body{{font:16px system-ui;margin:32px;color:#202b35;background:#faf8f2}}h1{{margin-bottom:8px}}
input{{font:inherit;padding:10px;width:min(600px,90%)}}table{{border-collapse:collapse;width:100%;margin-top:24px}}
td,th{{text-align:left;padding:12px;border-bottom:1px solid #cfc9bb;vertical-align:top}}td img{{width:150px;height:180px;object-fit:contain}}
small,summary{{color:#52616c}}pre{{white-space:pre-wrap;font-size:12px;max-width:650px}}code{{overflow-wrap:anywhere}}
@media(max-width:700px){{body{{margin:12px}}td img{{width:75px;height:120px}}td,th{{padding:5px}}}}
</style><h1>Voynich folio descriptions</h1><p>{len(rows)} catalogue panels · <code>{analysis_id}</code></p>
<p>Pixel Layout v1 measures colour and layout. Patches are not objects; circles are not proof of a celestial subject.
Domain is unassigned. Click an image for its complete Yale scan.</p>
<p><a href="descriptions.csv">CSV table</a> · <a href="descriptions.jsonl">JSONL table</a> · <a href="analysis.json">Heuristic registry</a></p>
<label for="filter">Filter folio, layout or description</label><br><input id="filter" type="search" placeholder="e.g. f75r or circular">
<p id="count">{len(rows)} panels</p><table><thead><tr><th>Panel crop</th><th>Folio</th><th>Layout rule</th><th>Description</th></tr></thead>
<tbody>{''.join(entries)}</tbody></table><script>
document.querySelector('#filter').addEventListener('input',function(){{
 let count=0;const query=this.value.toLowerCase();
 document.querySelectorAll('tbody tr').forEach(row=>{{row.hidden=!row.textContent.toLowerCase().includes(query);if(!row.hidden)count++;}});
 document.querySelector('#count').textContent=count+' panels';
}});</script></html>'''


def run():
    analysis_id, metadata = fingerprint()
    target = DATA / "analyses" / analysis_id
    with tempfile.TemporaryDirectory(prefix="folio-analysis-") as temporary:
        stage = Path(temporary)
        (stage / "previews").mkdir()
        rows = []
        for panel, source, box in panel_records():
            with Image.open(ROOT / source["path"]) as image:
                cropped = crop_image(image.convert("RGB"), box)
            measured = describe(np.asarray(cropped))
            row = {"analysis_id": analysis_id, "folio_id": panel["folio_id"],
                   "image_id": source["image_id"], "scan_label": source["label"],
                   "image_path": source["path"], "image_sha256": source["sha256"],
                   "catalogue_url": panel["catalogue_url"], "crop_bbox_normalized": box,
                   "panel_width": cropped.width, "panel_height": cropped.height, **measured}
            if min(cropped.size) < 500:
                row["quality_flags"].append("small_panel_at_archive_resolution")
            rows.append(row)
            cropped.thumbnail((260, 320))
            cropped.save(stage / "previews" / (panel["folio_id"] + ".jpg"), quality=85)
        jsonl = "".join(canonical(row) + "\n" for row in rows)
        (stage / "descriptions.jsonl").write_text(jsonl)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        (stage / "descriptions.csv").write_text(buffer.getvalue(), newline="")
        (stage / "index.html").write_text(gallery(rows, analysis_id))
        metadata.update(analysis_id=analysis_id, row_count=len(rows),
                        outputs={str(p.relative_to(stage)): sha(p.read_bytes())
                                 for p in sorted(stage.rglob("*")) if p.is_file()})
        (stage / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if target.exists():
            old = {str(p.relative_to(target)): sha(p.read_bytes()) for p in target.rglob("*") if p.is_file()}
            new = {str(p.relative_to(stage)): sha(p.read_bytes()) for p in stage.rglob("*") if p.is_file()}
            if old != new:
                raise ValueError("Existing analysis differs. Do not overwrite a frozen analysis.")
            print(f"Verified identical rerun: {analysis_id}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, target)
            print(f"Wrote {len(rows)} descriptions: {target}")
    return target


if __name__ == "__main__":
    run()
