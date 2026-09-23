"""Rank Voynich word tokens that are likely transcription problems, training pages only.

Signals per token (no decoding, no meaning claim):
  near_hapax  - the form occurs once in its transcription and is one edit from a form seen >= FREQUENT times
  flagged     - the transcriber marked it uncertain (masked alternative reading '?')
  uncertain_space - an uncertain space (',') touches the token
  line_split_disagree - v101 and EVA word counts on certain spaces differ by >= 2 on this line
A natural-language control applies the same near-hapax filter to 25,000-token samples of
pinned corpora; the Voynich rate is compared with it before any token is called an error.
The two transcriptions use different glyph alphabets, so readings are not compared letter by letter.

    python -m experiments.voynich_suspects
"""
from collections import Counter
import csv
import json
from pathlib import Path
import re

from voynich.data import load_documents
from voynich.decipher import edit_distance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/voynich-suspects'
FREQUENT = 5
IMAGES = ROOT / 'data/folios/images.json'


def certain_counts(name):
    out = {}
    for doc in load_documents(ROOT / f'artifacts/data/{name}'):
        if doc['split'] != 'train': continue
        for locus in doc['loci']:
            out[locus['id']] = len([w for w in doc['text'][locus['start']:locus['end']].replace(',', '').split('.') if w])
    return out


CONTROLS = dict(latin='UD_Latin-ITTB/la_ittb-ud-train.conllu', italian='UD_Italian-ISDT/it_isdt-ud-train.conllu',
                old_french='UD_Old_French-PROFITEROLE/fro_profiterole-ud-train.conllu', german='UD_German-GSD/de_gsd-ud-train.conllu')


def near_hapax_rate(words):
    counts = Counter(words); by_length = {}
    for f, c in counts.items():
        if c >= FREQUENT: by_length.setdefault(len(f), []).append(f)
    near = sum(1 for w in words if counts[w] == 1 and neighbours(w, by_length.get(len(w) - 1, []) + by_length.get(len(w), []) + by_length.get(len(w) + 1, [])))
    return dict(tokens=len(words), hapax_share_of_types=sum(1 for c in counts.values() if c == 1) / len(counts), near_hapax_share_of_tokens=near / len(words))


def controls(n=25000):
    from voynich.corpora import conllu_sentences
    from voynich.decipher import normalize
    out = {}
    for lang, path in CONTROLS.items():
        words = []
        for sentence in conllu_sentences(ROOT / 'artifacts/language-sources' / path):
            words += normalize(' '.join(sentence['words'])).split()
            if len(words) >= n: break
        out[lang] = near_hapax_rate(words[:n])
    return out


def lines(name):
    """Training-page lines keyed by locus: list of (word, uncertain_space_touching)."""
    out = {}
    for doc in load_documents(ROOT / f'artifacts/data/{name}'):
        if doc['split'] != 'train': continue
        for locus in doc['loci']:
            text = doc['text'][locus['start']:locus['end']]
            pieces = re.split(r'([.,])', text)
            words, seps = pieces[0::2], pieces[1::2]
            row = []
            for i, w in enumerate(words):
                if not w: continue
                touching = (i > 0 and seps[i - 1] == ',') or (i < len(seps) and seps[i] == ',')
                row.append((w, touching))
            out[locus['id']] = dict(page=doc['page'], currier=doc['currier'], section=doc['section'], words=row)
    return out


def neighbours(form, frequent):
    """Frequent forms at edit distance exactly one (length filter first)."""
    return [f for f in frequent if abs(len(f) - len(form)) <= 1 and edit_distance(form, f) == 1]


def analyse(name, own, other, split_gap):
    counts = Counter(w for l in own.values() for w, _ in l['words'] if '?' not in w)
    frequent = [f for f, c in counts.items() if c >= FREQUENT]
    by_length = {}
    for f in frequent: by_length.setdefault(len(f), []).append(f)
    rows = []
    for locus, l in own.items():
        disagree = split_gap.get(locus, 0) >= 2
        for i, (w, touching) in enumerate(l['words']):
            flagged = '?' in w
            near = []
            if not flagged and counts[w] == 1:
                pool = by_length.get(len(w) - 1, []) + by_length.get(len(w), []) + by_length.get(len(w) + 1, [])
                near = sorted(neighbours(w, pool), key=lambda f: -counts[f])
            score = 2 * bool(near) + 2 * flagged + touching + disagree
            if score:
                rows.append(dict(transcription=name, locus=locus, page=l['page'], word_index=i, form=w, count=counts.get(w, 0),
                                 near_hapax=bool(near), nearest=near[0] if near else '', nearest_count=counts[near[0]] if near else 0,
                                 flagged=flagged, uncertain_space=touching, line_split_disagree=disagree,
                                 currier=l['currier'], section=l['section'], score=score))
    tokens = sum(len(l['words']) for l in own.values())
    hapax = sum(1 for c in counts.values() if c == 1)
    return rows, dict(tokens=tokens, types=len(counts), hapax_types=hapax, hapax_share_of_types=hapax / len(counts),
                      near_hapax_tokens=sum(r['near_hapax'] for r in rows), flagged_tokens=sum(r['flagged'] for r in rows),
                      uncertain_space_tokens=sum(r['uncertain_space'] for r in rows),
                      near_hapax_share_of_tokens=sum(r['near_hapax'] for r in rows) / tokens,
                      lines=len(own), disagreeing_lines=sum(1 for k in own if split_gap.get(k, 0) >= 2))


def main():
    gc, zl = lines('gc'), lines('zl')
    a, b = certain_counts('gc'), certain_counts('zl')
    split_gap = {k: abs(a[k] - b[k]) for k in set(a) & set(b)}
    images = {'f' + x['label']: x['path'] for x in json.loads(IMAGES.read_text()) if not x['label'].startswith('[')}
    all_rows, summary = [], {}
    for name, own, other in (('v101', gc, zl), ('eva', zl, gc)):
        rows, s = analyse(name, own, other, split_gap)
        for r in rows: r['image'] = images.get(r['page'], '')
        all_rows += rows; summary[name] = s
    all_rows.sort(key=lambda r: (-r['score'], r['transcription'], r['page'], r['locus'], r['word_index']))
    # Lines both transcriptions flag in some way: highest priority for visual review.
    flagged = {t: {r['locus'] for r in all_rows if r['transcription'] == t and (r['near_hapax'] or r['flagged'])} for t in ('v101', 'eva')}
    both = sorted(flagged['v101'] & flagged['eva'])
    # Visual-review list: EVA single-glyph substitutions of a frequent word, plain EVA letters only
    # (rare-glyph codes such as 'Ç' or 'Î' are real rare glyphs, reported separately), length >= 4.
    plain = re.compile(r'^[a-z]{4,}$')
    subs = [r for r in all_rows if r['transcription'] == 'eva' and r['near_hapax'] and len(r['form']) == len(r['nearest'])]
    pairs = Counter(tuple(sorted((a, b))) for r in subs for a, b in zip(r['form'], r['nearest']) if a != b and a.isascii() and b.isascii())
    review = [r for r in subs if plain.match(r['form']) and r['nearest_count'] >= 50]
    review.sort(key=lambda r: (-r['nearest_count'], r['transcription'], r['locus']))
    summary.update(eva_substitution_pairs=[dict(pair=''.join(k), count=v) for k, v in pairs.most_common(15)],
                   eva_substitutions=len(subs), eva_rare_glyph_near_hapax=sum(1 for r in all_rows if r['transcription'] == 'eva' and r['near_hapax'] and not r['form'].isascii()),
                   natural_language_controls=controls(), substitution_review_candidates=len(review),
                   lines_with_certain_space_gap_1=sum(v >= 1 for v in split_gap.values()), shared_loci=len(set(gc) & set(zl)), loci_flagged_by_both=len(both), frequent_threshold=FREQUENT,
                   pages=len({l['page'] for l in gc.values()}), training_pages_only=True, voynich_decoded=False)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'suspects.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    (OUT / 'review-loci.json').write_text(json.dumps(dict(loci_flagged_by_both=both,
        lines={k: dict(v101=' '.join(w for w, _ in gc[k]['words']), eva=' '.join(w for w, _ in zl[k]['words']),
                       page=gc[k]['page'], image=images.get(gc[k]['page'], '')) for k in both}), indent=2, ensure_ascii=False) + '\n')
    with (OUT / 'review-top50.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(review[:50])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
