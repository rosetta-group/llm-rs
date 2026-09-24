"""Relational-fragment development control. Recognition never exports parentage edges."""
from collections import Counter, defaultdict
import math
import random
import re
import unicodedata

CLASSES = ('FAMILY', 'OTHER')
ARMS = ('frame', 'marker', 'frame_marker')
EXACT = ('i-*65', 'i-*65-qe', 'i-jo', 'i-jo-qe', 'i-je-we',
         'tu-ka-te-qe', 'tu-ka-te-re', 'tu-ka-ta-si')
TAILS = ('-u-jo', '-ko-wo', '-*65')


def tokens(text):
    return re.split(r'[\s,/⌞⌟]+', re.sub(r'⟦.*?⟧', ' ', text, flags=re.S))


def clean_candidate(token):
    """For candidate retrieval ONLY; never accepted as a strict reading."""
    return ''.join(ch for ch in unicodedata.normalize('NFD', token)
                   if not unicodedata.combining(ch) and ch not in '[]⌜⌝⌞⌟')


def inventory(records):
    hits = []
    for obj, item in sorted(records.items()):
        text = item.get('content') or ''
        unerased = re.sub(r'⟦.*?⟧', lambda m: re.sub(r'[^\n]', ' ', m.group()), text, flags=re.S)
        for line_index, (raw, visible) in enumerate(zip(text.splitlines(), unerased.splitlines()), 1):
            for token in tokens(visible):
                candidate = clean_candidate(token)
                category = ('exact' if token in EXACT else
                            'attached_tail' if any(token.endswith(t) and len(token) > len(t)
                                                   for t in TAILS) else None)
                intact = bool(re.fullmatch(r'(?:[a-z0-9]+|\*\d+)(?:-(?:[a-z0-9]+|\*\d+))*', token))
                if category and intact:
                    status = 'intact'
                elif candidate in EXACT or any(candidate.endswith(t) and len(candidate) > len(t) for t in TAILS):
                    category, status = 'candidate', 'uncertain_text'
                else:
                    continue
                hits.append({'object': obj, 'document': item['heading_short'], 'line_index': line_index,
                             'token': token, 'category': category, 'status': status, 'raw': raw})
    return {'hits': hits, 'note': 'Candidate inventory only; no automatic kinship labels or lemma counts.'}


def public_view(cases, records):
    """Validate exact component spans; supply only ordered onomastic/non-name identities."""
    rows, vocabulary = [], {}
    for case in cases:
        if case['status'] != 'scored':
            continue
        sequence = []
        locations = []
        for part in case['parts']:
            pieces = []
            for segment in part['segments']:
                raw = records[case['object']]['content'].splitlines()[segment['line_index'] - 1]
                start, end = segment['span']
                locations.append((segment['line_index'], start, end))
                if raw != segment['raw'] or raw[start:end] != segment['text']:
                    raise ValueError('Changed source span')
                pieces.append(segment['text'])
            word = '-'.join(pieces)
            if word != part['text']:
                raise ValueError('Invalid cross-line join')
            if part['kind'] not in ('name', 'word'):
                raise ValueError('Invalid supplied span kind')
            qe = word.endswith('-qe')
            root = word[:-3] if qe else word
            if part['kind'] == 'name':
                sequence.append('N')
            else:
                if not re.fullmatch(r'(?:[a-z0-9]+|\*\d+)(?:-(?:[a-z0-9]+|\*\d+))*', root):
                    raise ValueError('Uncertain relation word cannot enter scoring')
                vocabulary.setdefault(root, f'W{len(vocabulary)+1:03d}')
                sequence.append(vocabulary[root])
            if qe:
                sequence.append('Q')
        if locations != sorted(locations) or len(set(locations)) != len(locations):
            raise ValueError('Fragments must preserve source order')
        rows.append({'id': case['id'], 'object': case['object'], 'sequence': sequence})
    return rows, vocabulary


def signature(row, arm):
    seq = row['sequence']
    if arm == 'frame':
        return tuple('W' if x.startswith('W') else x for x in seq)
    if arm == 'marker':
        return tuple(sorted({x for x in seq if x.startswith('W')}))
    if arm == 'frame_marker':
        return tuple(seq)
    raise ValueError(arm)


def predict(train, labels, row, arm):
    groups = defaultdict(Counter)
    for r in train:
        if signature(r, arm) == signature(row, arm):
            groups[r['object']][labels[r['id']]] += 1
    votes = Counter({label: 0.0 for label in CLASSES})
    for counts in groups.values():
        for label, n in counts.items():
            votes[label] += n / sum(counts.values())
    n = len(groups)
    best = max(CLASSES, key=lambda label: votes[label])
    agreement = votes[best] / n if n else 0
    tie = math.isclose(votes[CLASSES[0]], votes[CLASSES[1]], abs_tol=1e-12)
    reason = 'unseen' if n == 0 else 'too_few_objects' if n < 2 else 'conflict' if tie or agreement < .9 else 'predicted'
    return {'prediction': best if reason == 'predicted' else None, 'reason': reason,
            'agreement': agreement, 'votes': dict(votes), 'support_objects': sorted(groups)}


def weights(rows, labels):
    counts = Counter((labels[r['id']], r['object']) for r in rows)
    n_objects = Counter(label for label, obj in counts)
    if set(n_objects) != set(CLASSES):
        raise ValueError('Both classes required')
    return {r['id']: .5 / n_objects[labels[r['id']]] / counts[labels[r['id']], r['object']]
            for r in rows}


def metrics(rows, labels, predictions):
    w = weights(rows, labels)
    called = [r['id'] for r in rows if predictions[r['id']]['prediction'] is not None]
    correct = [key for key in called if predictions[key]['prediction'] == labels[key]]
    cov = sum(w[key] for key in called)
    rec = sum(w[key] for key in correct)
    family = sum(w[key] for key in correct if labels[key] == 'FAMILY') * 2
    false = sum(w[key] for key in called if labels[key] == 'OTHER' and predictions[key]['prediction'] == 'FAMILY') * 2
    return {'balanced_recall': rec, 'coverage': cov, 'conditional_accuracy': rec / cov if cov else None,
            'family_recall': family, 'false_family_rate': false,
            'raw': {'cases': len(rows), 'correct': len(correct), 'wrong': len(called)-len(correct),
                    'abstained': len(rows)-len(called)}}


def evaluate(rows, labels, arm):
    predictions = {}
    for obj in sorted({r['object'] for r in rows}):
        train = [r for r in rows if r['object'] != obj]
        for row in rows:
            if row['object'] == obj:
                predictions[row['id']] = predict(train, labels, row, arm)
    return {'metrics': metrics(rows, labels, predictions), 'predictions': predictions}


def ceiling(rows, labels, arm):
    w = weights(rows, labels)
    buckets = defaultdict(lambda: {'weights': Counter(), 'cases': defaultdict(list)})
    for row in rows:
        b = buckets[signature(row, arm)]
        b['weights'][labels[row['id']]] += w[row['id']]
        b['cases'][labels[row['id']]].append(row['id'])
    return {'balanced_recall_upper_bound': sum(max(b['weights'].values()) for b in buckets.values()),
            'buckets': [{'signature': list(sig), 'class_weights': dict(b['weights']), 'cases': dict(b['cases'])}
                        for sig, b in buckets.items()]}


def performance_pass(m):
    return (m['balanced_recall'] >= .9 and m['coverage'] >= .9 and
            m['conditional_accuracy'] is not None and m['conditional_accuracy'] >= .95 and
            m['family_recall'] >= .9 and m['false_family_rate'] <= .05)


def representation(rows, labels):
    counts = {label: len({r['object'] for r in rows if labels[r['id']] == label}) for label in CLASSES}
    return {'class_objects': counts, 'evaluable': min(counts.values()) >= 5}


def run(rows, labels, cases):
    if len({r['id'] for r in rows}) != len(rows) or set(labels) != {r['id'] for r in rows}:
        raise ValueError('Case key mismatch')
    if set(labels.values()) != set(CLASSES):
        raise ValueError('Invalid gold classes')
    preflight = representation(rows, labels)
    subclasses = {kind: len({c['object'] for c in cases if c['status'] == 'scored' and c['kind'] == kind})
                  for kind in ('service', 'occupation', 'pair')}
    preflight['negative_subclass_objects'] = subclasses
    preflight['objects'] = len({r['object'] for r in rows})
    preflight['evaluable'] &= (preflight['objects'] >= 10 and subclasses['service'] >= 3 and
                               subclasses['occupation'] >= 3 and subclasses['pair'] >= 2)
    arms = {arm: {**evaluate(rows, labels, arm), 'ceiling': ceiling(rows, labels, arm)} for arm in ARMS}
    rng = random.Random(20260924)
    negatives = []
    for n in range(199):
        swaps = {obj for obj in sorted({r['object'] for r in rows}) if rng.random() < .5}
        other = {'FAMILY':'OTHER', 'OTHER':'FAMILY'}
        shuffled = {r['id']: other[labels[r['id']]] if r['object'] in swaps else labels[r['id']] for r in rows}
        valid = representation(rows, shuffled)['evaluable']
        m = evaluate(rows, shuffled, 'frame_marker')['metrics'] if valid else None
        negatives.append({'iteration': n, 'swapped_objects': sorted(swaps), 'evaluable': valid,
                          'metrics': m, 'pass': performance_pass(m) if valid else None})
    valid = [n for n in negatives if n['evaluable']]
    rate = sum(n['pass'] for n in valid) / len(valid) if valid else None
    success = performance_pass(arms['frame_marker']['metrics']) and rate is not None and rate <= .05
    return {'preflight': preflight, 'arms': arms, 'negatives': negatives,
            'gate': {'status': 'not_evaluable' if not preflight['evaluable'] else 'pass' if success else 'fail',
                     'negative_evaluable': len(valid), 'negative_passes': sum(n['pass'] for n in valid),
                     'negative_pass_rate': rate},
            'source_direction_status': dict(Counter(c['direction_status'] for c in cases
                                                    if c['status'] == 'scored' and c['label'] == 'FAMILY')),
            'predicted_edges': [], 'linear_a_scored': False}
