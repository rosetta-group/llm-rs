"""Descriptive source-audit helpers; no name recognition or relation inference."""
from collections import Counter


def validate(inventory, corpus):
    """Require complete logical-row coverage and preserve original sign occurrences."""
    faces = inventory['faces']
    rows = inventory['rows']
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate audit row')
    for doc, face in faces.items():
        source = corpus[doc]
        selected = [r for r in rows if r['document'] == doc]
        lines = source['unicode_text'].splitlines()
        if sorted(r['logical_row'] for r in selected) != list(range(1, len(lines) + 1)):
            raise ValueError(f'Logical-row coverage: {doc}')
        covered = []
        for row in selected:
            if row['object'] != face['object']:
                raise ValueError(f'Wrong physical object: {row["id"]}')
            if row['raw'] != lines[row['logical_row'] - 1]:
                raise ValueError(f'Changed original glyphs: {row["id"]}')
            if row['source_signs'] != [s for s in source['signs'] if s['n'] in row['sign_n']]:
                raise ValueError(f'Changed original signs: {row["id"]}')
            if row['source_words'] != [source['words'][i] for i in row['word_indices']]:
                raise ValueError(f'Changed original words: {row["id"]}')
            if row['kind'] == 'logogram_count':
                if row['word_indices'] or any(s['role'] != 'logogram' for s in row['source_signs']):
                    raise ValueError(f'Logogram promoted to word: {row["id"]}')
            if row['kind'] == 'lexical_count' and len(row['source_words']) != 1:
                raise ValueError(f'Expected one graphic word: {row["id"]}')
            covered.extend(row['sign_n'])
        covered.extend(face['erased_sign_n'])
        if sorted(covered) != [s['n'] for s in source['signs']]:
            raise ValueError(f'Sign coverage or duplicated sign: {doc}')
    if any(r['document'] not in faces for r in rows):
        raise ValueError('Unknown audit face')


def summary(inventory):
    rows = inventory['rows']
    return {
        'scope': 'Supplied development annotations; no automatic semantic classification',
        'physical_objects': len({r['object'] for r in rows}),
        'faces': len(inventory['faces']),
        'logical_rows': len(rows),
        'kinds': dict(Counter(r['kind'] for r in rows)),
        'by_face': {doc: dict(Counter(r['kind'] for r in rows if r['document'] == doc))
                    for doc in inventory['faces']},
        'inferred_kinship_edges': 0,
        'independent_expert_review': False,
    }


def literal_occurrences(corpus, queries):
    """Literal source-word hits, including conflict flags; never merge proposed aliases."""
    out = {}
    for word in queries:
        hits = []
        for doc, record in sorted(corpus.items()):
            for index, found in enumerate(record.get('words') or []):
                if found == word:
                    hits.append({
                        'document': doc, 'object': record.get('parent_object') or doc,
                        'word_index': index, 'words': record['words'],
                        'record_conflicts': record.get('conflicts') or [],
                        'record_has_uncertain_signs': any(s.get('certain') is not True
                                                        for s in record.get('signs', [])),
                        'record_has_sign_layer': bool(record.get('signs')),
                    })
        out[word] = {'occurrences': len(hits), 'objects': len({h['object'] for h in hits}),
                     'hits': hits}
    return out


def markdown_inventory(inventory):
    lines = ['# HT 85/117 source inventory', '',
             'Manual development annotations; all personal identity, gender and kinship remain unknown.',
             'Physical lines come from GORILA I. Logical rows index the pinned corpus, including its divider.',
             'Spellings with asterisks are sign labels, not recovered pronunciations.', '',
             '| Face | Physical line | Logical row | Working reading | Count | Role | Note |',
             '|---|---|---:|---|---:|---|---|']
    for r in inventory['rows']:
        quantity = '' if r['quantity'] is None else str(r['quantity'])
        lines.append(f'| {r["document"]} | {r["physical_line"]} | {r["logical_row"]} | '
                     f'{r["working_reading"]} | {quantity} | {r["kind"]} | {r["note"]} |')
    return '\n'.join(lines) + '\n'
