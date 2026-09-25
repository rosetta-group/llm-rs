"""Archive and summarize the completed eight-language extension."""
import gzip
import hashlib
import tarfile

from experiments.language_expansion import OUT, ROOT, STATE, read, MODELS, EXPANDED
from experiments.rejection_transfer_v2 import seal
from voynich.data import digest


def archive():
    completed = set(read(STATE/'run.json')['completed'])
    released = [dict(id=r['id'], language=r['language'],
        source_groups=sorted({x['group'] for p in r['passages'] for x in p['rows']}),
        source_ids=sorted({x['id'] for p in r['passages'] for x in p['rows']}))
        for r in read(STATE/'evaluator-only/answers.json') if r['id'] in completed]
    seal(OUT/'released-source-ids.json', dict(evaluated=released,
        policy='Exclude these complete works, manuscript contents and parallel versions from future fresh rounds; retain all previous exclusions.'))
    paths = [STATE/n for n in ['partitions.json','challenge.json','run.json','tests.log','run.log','evaluation.log','audit.log','ATTRIBUTION.md']]
    for directory in ('public','fit','transfer','sealed-keys','evaluator-only'):
        paths.extend(sorted((STATE/directory).glob('*.json')))
    paths += [STATE/'raw'/n for n in ['cometa-metadata.json','histcorp-metadata.json','diakorp-readme.txt','downloads.json']]
    manifest = {str(p.relative_to(STATE)): digest(p) for p in paths}
    target = OUT/'evaluated-records.tar.gz'
    with target.open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for path in paths:
                info = tar.gettarinfo(str(path),arcname=str(path.relative_to(STATE)))
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ''
                with path.open('rb') as stream:
                    tar.addfile(info,stream)
    with tarfile.open(target,'r:gz') as tar:
        restored = {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
    assert restored == manifest
    seal(OUT/'archive.json',dict(files=manifest,sha256=digest(target),verified=True,
        note='Includes normalized source derivatives under their original licenses; see ATTRIBUTION.md. Large raw inputs and binary models excluded.'))


def report():
    result, f, audit = [read(OUT/n) for n in ('results.json','freeze.json','audit.json')]
    diagnostics = {r['id']:r for r in read(OUT/'plaintext-diagnostics.json')['rows']}
    rows = result['outcomes']
    lines = ['# Old Czech and Old Occitan extension', '',
        'Pattern: **language coverage test**. Compare six and eight candidate languages on two fresh historical-source ciphertext pairs.', '',
        f"The eight-language system accepted {result['expanded_correct']}/2 correct languages; removing the true language gave {result['omitted_rejected']}/2 conclusive rejections.", '',
        '**Transfer:** decode a different work with the fitted key held fixed.',
        '**CER:** character edit distance divided by the 5,200 reference letters.',
        '**Excess:** decoded bits per letter minus that model’s calibration score.', '',
        '## What was done', '',
        '- Added Old Czech (DIAKORP/HistCorp) and Old Occitan (COMETA); six existing models reproduce exactly.',
        '- Trained each new model on 400,000 letters across four works; calibration uses 20,000 letters across two other works.',
        '- Compared Latin, German, Old French, English, Italian, Catalan, Old Czech and Old Occitan.',
        f"- Completed {len(rows)*len(MODELS)}/16 fits in {result['fit_worker_seconds']/3600:.3f} aggregate worker-hours; 194 tests passed.",
        f"- Re-extracted {audit['source_rows_reextracted']} source rows, rebuilt eight priors, replayed four encryptions and {audit['replayed_transfers']} transfers.", '',
        '## Why it was done', '',
        'Czech supplies the first Slavic candidate and historical medical material. Occitan tests a medieval Romance alternative close to the existing Catalan and Old French candidates.', '',
        '```text', 'Freeze sources, models and existing rule; commit', 'Encrypt two independent-key pairs',
        'Fit eight models; seal keys; decode different works', 'Compare six, eight and true-language-omitted candidate sets',
        'Open answers once; preserve all failures', '```', '',
        '## Results', '',
        '| True language | Six-language accepted | Eight-language fit / transfer winners | Eight-language accepted | True language omitted |',
        '|---|---|---|---|---|']
    def label(d):
        return d['accepted'] or ('inconclusive' if d['inconclusive'] else 'none')
    for r in rows:
        d=r['decisions'];e=d['expanded']
        lines.append(f"| {r['language']} | {label(d['baseline'])} | {e['fit']['winner']} / {e['transfer']['winner']} | {label(e)} | {label(d['omitted'])} |")
    lines += ['', f"The frozen feasibility target **{'passed' if result['feasibility_pass'] else 'failed'}**. Stop reason: `{result['stop_reason']}`.", '',
        '| True model | Fit CER | Transfer CER | Fit excess | Transfer excess | Coverage | Fit / transfer margin |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=EXPANDED[r['language']];s=r['scores'][m];e=r['character_error'][m];d=r['decisions']['expanded']
        lines.append(f"| {r['language']} | {e['fit_cer']:.2%} | {e['transfer_cer']:.2%} | {s['fit_excess']:.3f} | {s['transfer_excess']:.3f} | {s['coverage']:.2%} | {d['fit']['margin']:.3f} / {d['transfer']['margin']:.3f} |")
    lines += ['', 'Gate reasons (unchanged thresholds):', '']
    for r in rows:
        for name,d in r['decisions'].items():
            lines.append(f"- {r['language']} / {name}: {', '.join(d['reasons']) or 'all gates passed'}.")
    lines += ['', '## Post-run diagnostic', '',
        '| Source | Calibration bits/letter | True transfer bits/letter | True transfer excess | Recovered transfer bits/letter |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        m=EXPANDED[r['language']];truth=diagnostics[r['id']]['scores'][m]
        lines.append(f"| {r['language']} | {f['entropy'][m]:.3f} | {truth['plaintext_bits_per_letter'][1]:.3f} | {truth['plaintext_excess'][1]:.3f} | {r['scores'][m]['transfer_excess']+f['entropy'][m]:.3f} |")
    lines += ['', 'These true-plaintext scores were calculated after grading, never used to choose models or thresholds. '
        'A low true-plaintext excess with a high recovered-text excess points to key recovery; a high true-plaintext excess also exposes source/genre mismatch.', '',
        '## Source interpretation and limits', '',
        '1. **Czech is historical, but spelling is normalized.** '
        '[DIAKORP](https://wiki.korpus.cz/doku.php/en:cnk:diakorp) transcribes historical forms. '
        'The selected work dates range from 1350–1400 to 1492; the fixed alphabet additionally removes accents and merges j/i and k/c. '
        'Training includes Lékařství neznámého františkána; testing uses Hvězdářství krále Jana and the 1492 travel account. '
        'No modern Czech newspaper material is used.', '',
        '2. **Occitan supplies different works.** '
        '[COMETA](https://zenodo.org/records/15300719) provides medieval Provence/Languedoc transcriptions, manually corrected after handwriting recognition. '
        'The two challenge manuscripts are NAF 11151 and Harley 7403; their opening excerpts are not a dedicated medical test. '
        'Multiple copies of Honorat and Philomena are kept out of other roles. Verse, prose, spelling and untagged foreign quotations remain mixed.', '',
        '3. **Two cases do not validate eight languages generally.** Each new language has one independent key, with fit and transfer from different works. '
        'No fresh retention controls for the existing six languages were run. A correct winner is weaker than passing the fixed acceptance rule. '
        'No reserved Voynich text was used and no Voynich language was identified.', '',
        '4. **Source independence is filtered and finite.** No selected challenge passage has an eight-word overlap with the audited references or earlier released passages. '
        'Filtering can remove material and create non-contiguous excerpts; different works can still share genre or translated traditions. '
        'Procedural blinding on one computer is not an independent evaluation.', '',
        '## Records and reproduction', '',
        '- [Protocol](PROTOCOL.md), [sources](sources.json), [freeze](freeze.json), [results](results.json), [audit](audit.json), '
        '[diagnostics](plaintext-diagnostics.json), [archive hashes](archive.json), [released works](released-source-ids.json).',
        '- Freeze commit: `'+read(STATE/'challenge.json')['freeze_commit']+'`.',
        '- `python -m experiments.language_expansion_sources download` restores pinned new inputs.',
        '- `python -m experiments.language_expansion verify` checks the committed freeze and input hashes.',
        '- `python -m experiments.language_expansion replay` repeats all transfer predictions and decisions.',
        '- `python -m experiments.audit_language_expansion` rebuilds models and encryption; sealed audit outputs must not already exist.',
        '- The archive contains model partitions, evaluated ciphertext/answers/predictions, logs and attribution. '
        'Earlier frozen dependencies remain required; see [reproduction instructions](../../docs/REPRODUCE.md).', '',
        '## Attribution', '',
        'Czech: Karel Kučera and Martin Stluka (2011), DIAKORP v5; Eva Pettersson and Beáta Megyesi (2018), HistCorp. '
        '[Distribution and license](https://sprakbanken-clarin.lingfil.uu.se/histcorp/readme/czech-readme-diakorp): '
        '[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). '
        'Occitan: Marinus Wiedner (2025), COMETA v1, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). '
        'Normalized excerpts are adaptations; the original source licenses continue to apply. '
        'Earlier-language source licenses and authors remain in the previous manifests and the archived ATTRIBUTION.md.', '']
    interpretation = ['## What the result supports', '',
        '1. **Czech is a useful addition in this control.** It wins on both works and passes acceptance with 5.54% transfer character error. '
        'The earlier six-language set ranks Old French first but correctly rejects it.', '',
        '2. **Occitan remains a failed acceptance case.** It wins both rankings, but 15.19% transfer character error raises excess to 0.525, '
        'above the frozen 0.500 ceiling. The small 0.025 miss is still a failure; neither the threshold nor the passage was changed.', '',
        '3. **The next bottleneck is recovery on these examples.** Correct Occitan transfer plaintext scores at −0.139 excess, '
        'so recovery adds 0.664 bits per letter. Develop key recovery on these now-released cases, then require new works and keys for confirmation. '
        'The two omitted-language rejections are controls, not an estimated false-acceptance rate.', '']
    lines[lines.index('## Source interpretation and limits'):lines.index('## Source interpretation and limits')] = interpretation
    source_lines = ['## Selected works', '',
        '| Language | Work / manuscript | Role | Source date label | Selected letters |',
        '|---|---|---|---|---:|']
    for language, works in read(OUT/'sources.json')['statistics'].items():
        for name, metadata in works.items():
            budget = dict(train=100000, calibration=10000, challenge=5200)[metadata['role']]
            source_lines.append(f"| {language} | {metadata['title']} (`{name}`) | {metadata['role']} | {metadata['period']} | {budget:,} |")
    source_lines += ['', 'Each challenge work contributes one passage. Date labels come from corpus metadata; '
                     'COMETA manuscript dates are not claimed to be a single contemporaneous window.', '']
    index = lines.index('## Records and reproduction')
    lines[index:index] = source_lines
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    archive()
    report()
