"""Render the completed coverage pilot and archive small reproducibility records."""
import gzip
import hashlib
import tarfile

from experiments.language_coverage import OUT, ROOT, STATE, read
from experiments.rejection_transfer_v2 import seal
from voynich.data import digest


def archive():
    completed = set(read(STATE / 'run.json')['completed'])
    released = [dict(id=r['id'], language=r['language'],
                     source_groups=sorted({x['group'] for p in r['passages'] for x in p['rows']}),
                     source_ids=sorted({x['id'] for p in r['passages'] for x in p['rows']}))
                for r in read(STATE / 'evaluator-only/answers.json') if r['id'] in completed]
    seal(OUT / 'released-source-ids.json', dict(evaluated=released,
         policy='Future fresh rounds must exclude these complete source groups and previous released sources.'))
    names = ['partitions.json','challenge.json','run.json','tests.log','run.log']
    paths = [STATE / n for n in names]
    for directory in ('public','fit','transfer','sealed-keys','evaluator-only'):
        paths.extend(sorted((STATE / directory).glob('*.json')))
    paths.extend(STATE / 'raw' / name for name in ('README.md','LICENSE.txt','5615759.json','13982324.json','downloads.json'))
    manifest = {str(p.relative_to(STATE)): digest(p) for p in paths}
    target = OUT / 'evaluated-records.tar.gz'
    with target.open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for path in paths:
                info = tar.gettarinfo(str(path),arcname=str(path.relative_to(STATE)))
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ''
                with path.open('rb') as stream:
                    tar.addfile(info,stream)
    with tarfile.open(target,'r:gz') as tar:
        restored = {member.name: hashlib.sha256(tar.extractfile(member).read()).hexdigest() for member in tar.getmembers()}
    if restored != manifest:
        raise ValueError('Archive round-trip differs')
    seal(OUT/'archive.json',dict(files=manifest,sha256=digest(target),verified=True,
         note='Normalized training/calibration and evaluated text included; raw large corpora available by pinned URL. Priors rebuild from partitions; compare probability hashes.'))


def report():
    result = read(OUT/'results.json')
    sources = read(OUT/'sources.json')
    audit = read(OUT/'audit.json')
    frozen = read(OUT/'freeze.json')
    lines = [
        '# Historical language coverage pilot', '',
        'Pattern: **language coverage test**. Compare fixed candidate sets on ciphertext with known historical sources.', '',
        f"The expanded system accepted {result['expanded_correct']}/{len(result['outcomes'])} true languages; "
        f"omitting the true language produced {result['omitted_rejected']}/{len(result['outcomes'])} conclusive rejections.", '',
        '**Transfer:** decode a second passage with the first passage’s sealed key.',
        '**Excess:** decoded bits per letter minus that model’s separate calibration score.',
        '**CER:** character edit distance divided by the 5,200 plaintext letters; gaps count as errors.', '',
        '## What was done', '',
        '- Audited and pinned Latin charters, Middle High German prose and Old Catalan.',
        '- Compared five original languages with six expanded candidates on three fresh passage pairs.',
        '- Refit all eight required models on exactly 400,000 letters each; calibration used 20,000 letters each.',
        f"- Completed {len(result['outcomes'])*8}/24 fits in {result['fit_worker_seconds']/3600:.3f} aggregate fit-worker hours.",
        f"- Rebuilt {len(audit['models_rebuilt'])} priors and replayed {audit['generator_passages_replayed']} encryptions and {audit['replayed_transfers']} fixed-key transfers.", '',
        '## Why it was done', '',
        'The earlier five-language pilot had one passage per language and mixed modern and historical sources. '
        'This tests whether broader historical coverage improves identification without forcing an incorrect label when a language is unavailable.', '',
        '```text',
        'Freeze equal-budget models, sources and decision rule',
        'Encrypt three independent-key passage pairs',
        'Fit models on the first passage and seal each key',
        'Transfer fixed keys to the second passage',
        'Compare baseline, expanded and true-language-omitted decisions',
        '```', '',
        '## Results', '',
        '| True source | Baseline accepted | Expanded accepted | Expanded fit / transfer winners | Omitted accepted |',
        '|---|---|---|---|---|',
    ]
    label = lambda decision: decision['accepted'] or ('inconclusive' if decision['inconclusive'] else 'none')
    for row in result['outcomes']:
        d = row['decisions']
        lines.append(f"| {row['language']} | {label(d['baseline'])} | {label(d['expanded'])} | "
                     f"{d['expanded']['fit']['winner']} / {d['expanded']['transfer']['winner']} | {label(d['omitted'])} |")
    lines += ['', f"Feasibility target: **{'passed' if result['feasibility_pass'] else 'not passed'}**. "
              f"Stop reason: `{result['stop_reason']}`.", '',
              '| Source | Model | Fit excess | Transfer excess | Coverage | Fit CER | Transfer CER |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in result['outcomes']:
        for model, error in row['character_error'].items():
            score = row['scores'][model]
            held = score['transfer_excess']
            held_label = f'{held:.3f}' if held is not None else 'unscorable'
            lines.append(f"| {row['language']} | {model} | {score['fit_excess']:.3f} | "
                f"{held_label} | {score['coverage']:.2%} | {error['fit_cer']:.2%} | {error['transfer_cer']:.2%} |")
    lines += ['', 'Broad models also have different calibration entropies. The following raw scores help separate '
              'that change from better text prediction; a smaller excess alone is not proof of improved recovery.', '',
              '| Source | Model | Calibration bits/letter | Decoded transfer bits/letter | True transfer plaintext bits/letter |',
              '|---|---|---:|---:|---:|']
    diagnostics = {r['id']: r for r in read(OUT/'plaintext-diagnostics.json')['rows']}
    for row in result['outcomes']:
        for model in row['character_error']:
            score = row['scores'][model]
            raw = None if score['transfer_excess'] is None else score['transfer_excess'] + frozen['entropy'][model]
            raw_label = f'{raw:.3f}' if raw is not None else 'unscorable'
            truth = diagnostics[row['id']]['scores'][model]['plaintext_bits_per_letter'][1]
            lines.append(f"| {row['language']} | {model} | {frozen['entropy'][model]:.3f} | {raw_label} | {truth:.3f} |")
    lines += ['', 'True-plaintext scores are post-run diagnostics only. They did not select models, thresholds or challenge passages.']
    lines += ['', 'Decision reasons (unchanged thresholds):', '']
    for row in result['outcomes']:
        for name, decision in row['decisions'].items():
            lines.append(f"- {row['language']} / {name}: {', '.join(decision['reasons']) or 'all gates passed'}. "
                         f"Fit margin {decision['fit']['margin']:.3f}; transfer margin {decision['transfer']['margin']:.3f}.")
    lines += ['', '## Interpretation and next target', '',
        '1. **Coverage changes the answer.** The baseline ranks Old French first on both passages of all three cases, '
        'but rejects all three. The expanded system ranks the correct language first on both passages in all three. '
        'A winning language label alone would have been misleading under the original candidate set.', '',
        '2. **Recovery improves materially.** German transfer CER falls from 45.69% to 12.48%; Latin from 12.37% to 6.69%. '
        'These error reductions are independent of entropy normalization. Catalan transfer CER is 9.27%.', '',
        '3. **The acceptance target fails.** Only Latin passes. German transfer excess is 0.745, above 0.50. '
        'Catalan has transfer excess 0.860 and a fitting margin of 0.148, below 0.25. '
        'Coverage exceeds 97% for all true models, and no cap explains these failures. '
        'All three omitted-language cases reject; that is a paired pilot result, not a population false-acceptance estimate.', '',
        '4. **Work on key recovery next.** Post-run scores of the correct German and Catalan transfer plaintexts '
        'have excesses 0.190 and 0.150, both below 0.50. The recovered text adds about 0.56 and 0.71 bits/letter respectively. '
        'Use these now-released cases for recovery development, retain the thresholds, and require fresh source groups for confirmation. '
        'A second Catalan author and unseen medical prose remain specific coverage gaps. Adding more language names is lower priority.', '',
        'The interpretation above describes this executed pilot only; it is not a newly tuned decision rule.', '',
        '## Source audit and limits', '',
        '1. **Latin is broader, not medically validated.** '
        '[LLCT](https://universaldependencies.org/treebanks/la_llct/) contains Tuscan legal charters from AD 774–897. '
        'The broad prior mixes 200,000 Aquinas letters with 200,000 charter letters; this does not represent fifteenth-century medical Latin.', '',
        '2. **German uses historical surface forms.** '
        '[ReM 2.1](https://doi.org/10.5281/zenodo.13982324) supplies 191 prose documents from 1050–1350. '
        'Training includes medical and religious texts; the challenge is M113 (St. Trudperter Hohes Lied) and M172 (Prager Predigtentwürfe). '
        'Numeric work groups keep manuscript variants together. Dialect, genre and date remain mixed.', '',
        '3. **Catalan remains within one work.** '
        '[HisCat](https://doi.org/10.5281/zenodo.5615759) is the thirteenth-century Llibre dels Fets. '
        'Complete folios separate training, calibration and challenge, but author and genre do not change. '
        'This is why Catalan was selected over using a modern news corpus; Occitan remains untested.', '',
        '4. **Small, deliberately filtered pilot.** Three keys cannot establish sensitivity or false-acceptance rates. '
        'Overlap filtering removes formulaic chunks and makes some passages non-contiguous. '
        'The baseline also has 400,000 training letters, so compare it with the expanded system here, not directly with the older 606,976-letter pilot. '
        'The expanded system changes both models and candidate count; no old-language retention claim is justified.', '',
        '| New source | Training-pool letters | Calibration letters after filtering | Challenge letters after filtering |',
        '|---|---:|---:|---:|']
    for name, stats in sources['statistics'].items():
        lines.append(f"| {name} | {stats['letters']['train']:,} | {stats['retained_calibration_letters']:,} | {stats['retained_challenge_letters']:,} |")
    lines += ['', 'Final checks found zero eight-word overlaps for every selected challenge passage against all reference collections, '
        'including joins between retained chunks. Texts use a Latin-letter normalization with explicit historical glyph expansions. '
        'Neither this alphabet nor these successful/failed controls identify Voynich’s language or validate the cipher family.', '',
        f"An additional post-run audit found {sum(audit['additional_released_overlap_counts'].values())} eight-word matches "
        'against the earlier released challenge inventory. This check selected no replacement passages.', '',
        '## Reproduction and records', '',
        '- [Protocol](PROTOCOL.md), [source manifest](sources.json), [freeze](freeze.json), [results](results.json), '
        '[audit](audit.json), [post-run plaintext diagnostics](plaintext-diagnostics.json), [archive hashes](archive.json), '
        '[released source groups](released-source-ids.json).',
        '- `python -m experiments.language_coverage_sources download` restores hash-pinned raw corpora.',
        '- `python -m experiments.language_coverage verify` checks code, sources, models and committed freeze.',
        '- `python -m experiments.language_coverage replay` repeats all completed transfer scores and decisions.',
        '- `python -m experiments.audit_language_coverage` additionally rebuilds priors and reproduces encryption; '
        'its sealed output must not already exist.',
        '- `evaluated-records.tar.gz` contains normalized model inputs, evaluated plaintext/ciphertext, keys, predictions and logs. '
        'Raw ReM and LLCT downloads and binary priors remain in artifacts; probability hashes support rebuilding priors.',
        '- New-source attribution: Timo Korkiakangas, Flavio Massimiliano Cecchini and Marco Passarotti (LLCT); '
        'ReM 2.1 creators listed in archived Zenodo metadata; Afra Pujol i Campeny and Marieke Meelen (HisCat). '
        'LLCT/ReM text CC BY-SA 4.0; HisCat CC BY 4.0. Existing corpus licenses remain recorded in the prior source manifests.', '',
        '**No Voynich text or reserved manuscript test was used.**', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    archive()
    report()
