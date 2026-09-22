"""Turn reviewed observations into versioned tables, evidence gallery and review templates.

    .venv/bin/python -m experiments.describe_folio_objects

This is deterministic annotation processing, not an automatic vision model.
"""
import csv
import html
import io
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from PIL import Image, ImageDraw
from experiments.describe_folios import panel_records, crop_image, canonical, sha
from experiments.download_folios import ROOT, DATA
from voynich.folio_objects import describe, DOMAINS, PATTERNS

INPUT = DATA / 'object-pilot/observations.json'
CODE = ['voynich/folio_objects.py', 'experiments/describe_folio_objects.py',
        'data/folios/object-pilot/RUBRIC.md', 'data/folios/object-pilot/observations.json',
        'data/folios/images.json', 'data/folios/crops.json', 'data/folios/sources/catalogue-index.json',
        'experiments/describe_folios.py', 'experiments/download_folios.py',
        'experiments/folio_requirements.txt', 'experiments/folio_agreement.py']


def physical_group(folio):
    # Conservative clusters; do not claim separate panels/leaves of a foldout independent.
    if re.match(r'f(?:67|68)[rv]', folio): return 'foldout_67_68'
    if folio == 'fRos' or re.match(r'f(?:85|86)[rv]', folio): return 'foldout_rosettes'
    return 'folio_' + re.match(r'f(\d+)', folio)[1]


def gallery(rows, ident):
    e = html.escape
    cards = []
    for r in rows:
        object_rows = ''.join(f"<tr><td>{e(o['id'])}</td><td>{e(o['kind'])}</td><td>{o['count_min']}–{o['count_max'] if o['count_max'] is not None else '?'}</td><td>{o['certainty']}</td></tr>" for o in r['objects'])
        relation_rows = ''.join(f"<li>{e(x['subject'])} → <b>{e(x['predicate'])}</b> → {e(x['object'])} ({x['certainty']})</li>" for x in r['relations'])
        cards.append(f'''<article><h2>{e(r['folio_id'])}</h2><div class="pair"><a href="../../{e(r['image_path'].split('data/folios/')[1])}"><img src="previews/{r['folio_id']}.jpg" alt="{r['folio_id']} evidence regions"></a><div>
<p><b>Domains:</b> {e(', '.join(r['domains']))}</p><p><b>Relation patterns:</b> {e(', '.join(r['relation_patterns']) or 'none assigned')}</p>
<p>{e(r['notes'])}</p><table><tr><th>Region</th><th>Visible form</th><th>Count bound</th><th>Certainty</th></tr>{object_rows}</table><ul>{relation_rows}</ul>
<p class="muted">Tentative domains: {e(', '.join(r['tentative_domains']) or 'none')}. Tentative patterns: {e(', '.join(r['tentative_patterns']) or 'none')}.</p>
<details><summary>Full structured observation</summary><pre>{e(json.dumps(r, indent=2))}</pre></details></div></div></article>''')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voynich Object–Relation pilot</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;background:#faf8f2;color:#24303a}}h1{{margin-bottom:8px}}.notice{{padding:18px;background:#fff0ce;border-left:4px solid #a56a13}}input{{font:inherit;padding:10px;width:80%}}article{{border-top:1px solid #bbb;padding:20px 0}}.pair{{display:grid;grid-template-columns:42% 1fr;gap:24px}}img{{width:100%}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:6px;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap;font-size:12px}}.muted{{color:#59646b}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.pair{{display:block}}}}</style>
<h1>Object–Relation pilot</h1><p>24 panels · <code>{ident}</code></p>
<div class="notice"><b>Development annotations by one AI observer.</b> Objects and evidence boxes were supplied by visual inspection. The script derives domain tags and relational patterns. Independent human agreement and text masking are pending. No word meanings are inferred.</div>
<p><a href="descriptions.csv">Panel table</a> · <a href="descriptions.jsonl">Structured records</a> · <a href="analysis.json">Provenance</a> · <a href="../../object-pilot/RUBRIC.md">Rubric</a></p>
<p>Boxes identify evidence groups; they are not precise silhouettes. Counts are bounds and may overlap between groups.</p>
<label>Filter by folio, domain, object or relation <input type="search" id="filter" placeholder="e.g. linked_basins or f81r"></label><p id="count">24 panels</p>
{''.join(cards)}<script>document.querySelector('#filter').addEventListener('input',function(){{let n=0;document.querySelectorAll('article').forEach(x=>{{x.hidden=!x.textContent.toLowerCase().includes(this.value.toLowerCase());if(!x.hidden)n++;}});document.querySelector('#count').textContent=n+' panels';}});</script></html>'''


def run():
    inputs = {p: sha((ROOT / p).read_bytes()) for p in CODE}
    import PIL
    ident = 'object_relation_v1_' + sha(canonical(dict(files=inputs, pillow=PIL.__version__)).encode())[:12]
    target = DATA / 'analyses' / ident
    observation = json.loads(INPUT.read_text())
    if len(observation['panels']) != 24 or len({r['folio_id'] for r in observation['panels']}) != 24:
        raise ValueError('Expected 24 distinct development panels')
    sources = {p['folio_id']: (s, box) for p, s, box in panel_records()}
    with tempfile.TemporaryDirectory(prefix='folio-objects-') as temp:
        stage = Path(temp); (stage / 'previews').mkdir(); rows = []
        for r in observation['panels']:
            source, box = sources[r['folio_id']]
            row = dict(analysis_id=ident, **r, physical_group=physical_group(r['folio_id']),
                       image_id=source['image_id'], image_path=source['path'], image_sha256=source['sha256'],
                       crop_bbox_normalized=box, **describe(r))
            rows.append(row)
            with Image.open(ROOT / source['path']) as im: panel = crop_image(im.convert('RGB'), box)
            panel.thumbnail((900, 1000)); draw = ImageDraw.Draw(panel)
            for o in r['objects']:
                x, y, right, bottom = o['bbox']; x*=panel.width; right*=panel.width; y*=panel.height; bottom*=panel.height
                colour = '#1267be' if o['certainty'] == 'clear' else '#b74114'
                draw.rectangle((x,y,right,bottom),outline=colour,width=2)
                draw.text((x+3,y+3),o['id'],fill='white',stroke_width=2,stroke_fill=colour)
            panel.save(stage / 'previews' / (r['folio_id'] + '.jpg'), quality=88)
        (stage / 'descriptions.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
        table = io.StringIO(newline=''); writer = csv.DictWriter(table,fieldnames=list(rows[0])); writer.writeheader()
        for r in rows: writer.writerow({k: canonical(v) if isinstance(v,(list,dict,bool)) else v for k,v in r.items()})
        (stage / 'descriptions.csv').write_text(table.getvalue(),newline='')
        (stage / 'index.html').write_text(gallery(rows,ident))
        mapping = {f'panel-{i+1:03}':r['folio_id'] for i,r in enumerate(rows)}
        (stage / 'review-key.json').write_text(json.dumps(dict(status='coordinator_only_after_masking',mapping=mapping),indent=2)+'\n')
        for reviewer, seed in [('A',17),('B',29)]:
            ids=list(mapping); random.Random(seed).shuffle(ids)
            template = dict(analysis_id=ident,annotator_id=None,annotator_type=None,independent=False,
                            text_masked=False,mask_reviewed=False,mask_manifest_sha256=None,status='pending',
                            panels=[dict(panel_id=i,domains={k:None for k in DOMAINS},patterns={k:None for k in PATTERNS},
                                         unassessable_reason=None,notes='') for i in ids])
            (stage / f'reviewer-{reviewer}.json').write_text(json.dumps(template,indent=2)+'\n')
        meta = dict(analysis_id=ident,files=inputs,pillow=PIL.__version__,observer=observation['observer'],
                    annotation_status='single_ai_development_review',text_masked=False,independent_annotators=0,
                    agreement=None,blind_packet_ready=False,row_count=len(rows),
                    domains={d:sum(d in r['domains'] for r in rows) for d in DOMAINS},
                    patterns={p:sum(p in r['relation_patterns'] for r in rows) for p in PATTERNS},
                    outputs={str(p.relative_to(stage)):sha(p.read_bytes()) for p in sorted(stage.rglob('*')) if p.is_file()})
        (stage / 'analysis.json').write_text(json.dumps(meta,indent=2)+'\n')
        if target.exists():
            if {str(p.relative_to(target)):sha(p.read_bytes()) for p in target.rglob('*') if p.is_file()} != {str(p.relative_to(stage)):sha(p.read_bytes()) for p in stage.rglob('*') if p.is_file()}:
                raise ValueError('Frozen output drift')
            print('Verified identical run:',ident)
        else:
            shutil.copytree(stage,target); print('Wrote:',target)
    return target


if __name__ == '__main__': run()
