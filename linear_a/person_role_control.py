"""Object-held-out structural control; deliberately no lexical features."""
from collections import Counter, defaultdict
import random
import math
import re

CLASSES = ('PERSON', 'DESIGNATION')
FEATURES = {
    'role': ('role',),
    'role_quantity': ('role', 'quantity'),
    'layout': ('role', 'quantity', 'position', 'repeated'),
}


def literal_words(text):
    """Keep editorial uncertainty; never turn a restored/doubtful word into an intact one."""
    text = re.sub(r'⟦.*?⟧', ' ', text, flags=re.S)
    return re.split(r'[\s,\/]+', text)


def public_case(case, text):
    target = case['target']
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', target):
        raise ValueError('Target must be an intact literal word')
    if case['raw_line'] not in text.splitlines():
        raise ValueError('Source line changed')
    start, end = case['entry_span']
    entry = case['raw_line'][start:end]
    if not entry or entry != case['entry_text']:
        raise ValueError('Entry span changed')
    words = literal_words(entry)
    if words.count(target) != 1:
        raise ValueError('Entry must identify exactly one target')
    # Entry words are a supplied annotation: preceding doubtful words still occupy a slot.
    if case['entry_words'][case['target_index']] != target:
        raise ValueError('Wrong target index')
    for word in case['entry_words']:
        if word not in words:
            raise ValueError('Annotated word absent from entry')
    offsets = [words.index(word) for word in case['entry_words']]
    if offsets != sorted(set(offsets)):
        raise ValueError('Annotated words not in source order')
    if case['role'] not in ('heading', 'entry', 'footer'):
        raise ValueError('Invalid role')
    if case['quantity'] not in ('none', 'one', 'many', 'unknown'):
        raise ValueError('Invalid quantity')
    return {'id': case['id'], 'object': case['object'], 'role': case['role'],
            'quantity': case['quantity'],
            'position': 'initial' if case['target_index'] == 0 else 'noninitial',
            'repeated': literal_words(text).count(target) > 1}


def signature(row, feature_set):
    return tuple(row[key] for key in FEATURES[feature_set])


def predict(training, labels, row, feature_set, forced=False):
    """One vote per object/signature, distributed among that object's labelled cases."""
    by_object = defaultdict(Counter)
    sig = signature(row, feature_set)
    for other in training:
        if signature(other, feature_set) == sig:
            by_object[other['object']][labels[other['id']]] += 1
    votes = Counter({label: 0.0 for label in CLASSES})
    for counts in by_object.values():
        for label, count in counts.items():
            votes[label] += count / sum(counts.values())
    support = len(by_object)
    best = max(CLASSES, key=lambda label: votes[label])
    confidence = votes[best] / support if support else 0.0
    if not support:
        prediction, reason = None, 'unseen_signature'
    elif math.isclose(votes[CLASSES[0]], votes[CLASSES[1]], rel_tol=0, abs_tol=1e-12):
        prediction, reason = None, 'tie'
    elif not forced and support < 2:
        prediction, reason = None, 'too_few_objects'
    elif not forced and confidence < 0.9:
        prediction, reason = None, 'conflicting_objects'
    else:
        prediction, reason = best, 'predicted'
    return {'prediction': prediction, 'reason': reason, 'confidence': confidence,
            'support_objects': sorted(by_object), 'votes': dict(votes)}


def weights(rows, labels):
    """Equal class weight, then objects within class, then cases on that object."""
    counts = Counter((labels[row['id']], row['object']) for row in rows)
    n_objects = Counter(label for label, _ in counts)
    if set(n_objects) != set(CLASSES):
        raise ValueError('Both gold classes required')
    return {row['id']: 0.5 / n_objects[labels[row['id']]] /
            counts[labels[row['id']], row['object']] for row in rows}


def metrics(rows, labels, predictions):
    weight = weights(rows, labels)
    called = [row['id'] for row in rows if predictions[row['id']]['prediction'] is not None]
    correct = [key for key in called if predictions[key]['prediction'] == labels[key]]
    coverage = sum(weight[key] for key in called)
    recall = sum(weight[key] for key in correct)
    return {'balanced_recall': recall, 'coverage': coverage,
            'conditional_accuracy': recall / coverage if coverage else None,
            'raw': {'cases': len(rows), 'called': len(called), 'correct': len(correct),
                    'wrong': len(called) - len(correct), 'abstained': len(rows) - len(called)}}


def evaluate(rows, labels, feature_set, forced=False):
    predictions = {}
    for obj in sorted({row['object'] for row in rows}):
        train = [row for row in rows if row['object'] != obj]
        for row in rows:
            if row['object'] == obj:
                predictions[row['id']] = predict(train, labels, row, feature_set, forced)
    return {'metrics': metrics(rows, labels, predictions), 'predictions': predictions}


def ceiling(rows, labels, feature_set):
    weight = weights(rows, labels)
    buckets = defaultdict(lambda: {'weights': Counter(), 'cases': defaultdict(list)})
    for row in rows:
        bucket = buckets[signature(row, feature_set)]
        label = labels[row['id']]
        bucket['weights'][label] += weight[row['id']]
        bucket['cases'][label].append(row['id'])
    return {'balanced_recall_upper_bound': sum(max(b['weights'].values()) for b in buckets.values()),
            'signatures': [dict(signature=dict(zip(FEATURES[feature_set], sig)),
                                class_weights=dict(b['weights']), cases=dict(b['cases']))
                           for sig, b in buckets.items()],
            'note': 'Optimistic in-sample ceiling for deterministic use of exactly these features.'}


def preflight(rows, labels):
    counts = {}
    for label in CLASSES:
        selected = [r for r in rows if labels[r['id']] == label]
        counts[label] = {'objects': len({r['object'] for r in selected}),
                         **{role: len({r['object'] for r in selected if r['role'] == role})
                            for role in ('heading', 'entry')}}
    objects = len({r['object'] for r in rows})
    ok = objects >= 10 and all(c['objects'] >= 5 and c['heading'] >= 2 and c['entry'] >= 2
                               for c in counts.values())
    return {'evaluable': ok, 'objects': objects, 'classes': counts}


def performance_pass(m):
    return (m['balanced_recall'] >= 0.9 and m['coverage'] >= 0.9 and
            m['conditional_accuracy'] is not None and m['conditional_accuracy'] >= 0.95)


def swap_object_labels(rows, labels, rng):
    swapped = {obj for obj in sorted({r['object'] for r in rows}) if rng.random() < 0.5}
    other = dict(zip(CLASSES, reversed(CLASSES)))
    return {r['id']: other[labels[r['id']]] if r['object'] in swapped else labels[r['id']]
            for r in rows}, sorted(swapped)


def run(rows, labels):
    if len({r['id'] for r in rows}) != len(rows) or set(labels) != {r['id'] for r in rows}:
        raise ValueError('Case IDs must be unique and labels must match exactly')
    if set(labels.values()) != set(CLASSES):
        raise ValueError('Invalid labels')
    check = preflight(rows, labels)
    outputs = {name: {'strict': evaluate(rows, labels, name),
                      'forced': evaluate(rows, labels, name, forced=True),
                      'ceiling': ceiling(rows, labels, name)} for name in FEATURES}
    rng = random.Random(20260924)
    negatives = []
    for iteration in range(199):
        shuffled, swapped = swap_object_labels(rows, labels, rng)
        valid = preflight(rows, shuffled)['evaluable']
        # Whole-object swaps can theoretically leave only one class. Explicitly skip that case.
        m = evaluate(rows, shuffled, 'layout')['metrics'] if valid else None
        negatives.append({'iteration': iteration, 'swapped_objects': swapped, 'evaluable': valid,
                          'metrics': m, 'pass': performance_pass(m) if valid else None})
    evaluable = [n for n in negatives if n['evaluable']]
    negative_rate = sum(n['pass'] for n in evaluable) / len(evaluable) if evaluable else None
    negative_ok = negative_rate is not None and negative_rate <= 0.05
    actual_ok = performance_pass(outputs['layout']['strict']['metrics'])
    status = 'not_evaluable' if not check['evaluable'] else 'pass' if actual_ok and negative_ok else 'fail'
    return {'preflight': check, 'feature_sets': outputs, 'negative_runs': negatives,
            'gate': {'status': status, 'primary_performance_pass': actual_ok,
                     'negative_control_pass': negative_ok, 'negative_evaluable': len(evaluable),
                     'negative_pass_count': sum(n['pass'] for n in evaluable),
                     'negative_pass_rate': negative_rate},
            'linear_a_scored': False, 'evaluation_kind': 'curated development challenge'}
