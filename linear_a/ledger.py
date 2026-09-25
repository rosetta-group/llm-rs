"""Conservative accounting-program pilot: opaque words and integer quantity vectors."""

import collections
import hashlib
import re

from linear_a import contexts
from linear_a.arithmetic import numeral_value


OPS = ('sum_before', 'sum_after', 'balance_before')
B_WORD = re.compile(r'[a-z]+[0-9]*(?:-(?:[a-z]+[0-9]*|\*[0-9]+))*')
B_COMMODITY = re.compile(r'(?:[A-Z][A-Z0-9+*:±-]*|\*[0-9]+[A-Z]*)(?::[mf])?')
B_UNITS = {'T', 'S', 'V', 'Z', 'M', 'N', 'P', 'Q', 'L', 'ZE', 'MO'}


def opaque(text, prefix):
    return prefix + hashlib.sha256(text.encode()).hexdigest()[:20]


def b_line(line):
    """Parse a whole line or reject it; never erase restoration/uncertainty signs."""
    text = re.sub(r'^\s*\.\S+\s*', '', line).strip()
    if not text or re.fullmatch(r'vac(?:at)?\.?', text):
        return {'kind': 'barrier', 'reason': 'blank'}
    tokens = [t for t in re.split(r'[\s,/]+', text) if t]
    words, quantities, commodity = [], {}, None
    for t in tokens:
        if B_WORD.fullmatch(t):
            words.append(t)
        elif t in B_UNITS:
            return {'kind': 'barrier', 'reason': 'measurement_unit'}
        elif B_COMMODITY.fullmatch(t):
            if commodity:
                return {'kind': 'barrier', 'reason': 'multiple_unquantified_commodities'}
            commodity = t
        elif t.isascii() and t.isdigit():
            key = commodity or 'UNSPECIFIED'
            if key in quantities:
                return {'kind': 'barrier', 'reason': 'repeated_quantity_key'}
            quantities[key] = int(t)
            commodity = None
        else:
            return {'kind': 'barrier', 'reason': 'damage_or_unsupported_notation'}
    if commodity:
        return {'kind': 'barrier', 'reason': 'commodity_without_quantity'}
    if not quantities:
        return {'kind': 'barrier', 'reason': 'text_heading'}
    return {'kind': 'row', 'words': sorted(set(words)), 'quantity': quantities}


def a_line(line, table):
    """Keep sign IDs, including unknown sounds. Fractions are barriers, never zero."""
    words, quantities, run, commodity = [], {}, [], None
    number = []

    def flush_word():
        if run:
            words.append('/'.join(run))
            run.clear()

    def flush_number():
        nonlocal commodity
        if not number:
            return True
        key = commodity or 'UNSPECIFIED'
        if key in quantities:
            return False
        quantities[key] = sum(number)
        number.clear()
        commodity = None
        return True

    for ch in line:
        if contexts._is_fraction(ch):
            return {'kind': 'barrier', 'reason': 'fraction'}
        if contexts._is_numeral(ch):
            flush_word()
            number.append(numeral_value(ch))
            continue
        if not flush_number():
            return {'kind': 'barrier', 'reason': 'repeated_quantity_key'}
        t = contexts._sign_type(ch)
        if t:
            if table.get(t, (None, 0))[1] >= .8:
                flush_word()
                if commodity:
                    return {'kind': 'barrier', 'reason': 'multiple_unquantified_commodities'}
                commodity = t
            else:
                run.append(t)
        elif ch.isspace() or ch in '\U00010100\U00010101|':
            flush_word()
        else:
            return {'kind': 'barrier', 'reason': 'damage_or_unsupported_notation'}
    flush_word()
    if not flush_number():
        return {'kind': 'barrier', 'reason': 'repeated_quantity_key'}
    if commodity:
        return {'kind': 'barrier', 'reason': 'commodity_without_quantity'}
    if not quantities:
        return {'kind': 'barrier', 'reason': 'text_heading'}
    return {'kind': 'row', 'words': sorted(set(words)), 'quantity': quantities}


def blocks(lines):
    """Blank, damaged, fractional and text-only lines bound contiguous numeric blocks."""
    block = []
    for row in [*lines, {'kind': 'barrier'}]:
        if row['kind'] == 'row':
            block.append(row)
        else:
            if len(block) >= 3:
                yield block
            block = []


def add(vectors):
    out = collections.Counter()
    for v in vectors:
        out.update(v)
    return dict(out)


def prediction(block, index, operation):
    """Never read the target quantity. No subsets, fitted constants or tolerances."""
    before = [r['quantity'] for r in block[:index]]
    after = [r['quantity'] for r in block[index+1:]]
    if operation == 'sum_before' and len(before) >= 2:
        return add(before)
    if operation == 'sum_after' and len(after) >= 2:
        return add(after)
    if operation == 'balance_before' and len(before) >= 2:
        first, rest = before[0], add(before[1:])
        if set(rest) - set(first):
            return None
        out = {k: v-rest.get(k, 0) for k, v in first.items()}
        return out if all(v >= 0 for v in out.values()) else None
    return None


def examples(records):
    out = []
    for record in records:
        for b in blocks(record['lines']):
            for i, row in enumerate(b):
                candidates = {op: p for op in OPS if (p := prediction(b, i, op)) is not None}
                if row['words'] and candidates:
                    out.append({'document': record['document'], 'group': record['group'],
                        'row': row['line_number'], 'words': row['words'], 'truth': row['quantity'],
                        'candidates': candidates, 'context_rows': [r['line_number'] for r in b],
                        'fold': record['fold']})
    return out


def fit(rows, minimum=3, accuracy=.8):
    """Learn each word's operation from >=3 distinct objects with >=80% exact fits.

    Each object has equal weight for a word/operation. All of its eligible occurrences
    must agree for that object to count as a hit. Prediction ties abstain if values differ.
    """
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for word in r['words']:
            for op, value in r['candidates'].items():
                stats[word, op][r['group']].append(value == r['truth'])
    chosen = collections.defaultdict(list)
    for (word, op), groups in sorted(stats.items()):
        hits = sum(all(v) for v in groups.values())
        if len(groups) >= minimum and hits / len(groups) >= accuracy:
            chosen[word].append({'operation': op, 'objects': len(groups), 'hits': hits})
    return dict(chosen)


def predict(row, model):
    matched = [(w, rule['operation'], row['candidates'][rule['operation']])
               for w in row['words'] for rule in model.get(w, [])
               if rule['operation'] in row['candidates']]
    values = {tuple(sorted(p.items())) for _, _, p in matched}
    return {'prediction': dict(next(iter(values))) if len(values) == 1 else None,
            'rules': [(w, op) for w, op, _ in matched]}


def cross_validate(rows):
    models, outputs = {}, []
    for fold in range(5):
        model = fit([r for r in rows if r['fold'] != fold])
        models[str(fold)] = model
        outputs.extend({**r, **predict(r, model)} for r in rows if r['fold'] == fold)
    return models, outputs


def shuffled_records(records, rng):
    """Permute complete numeric vectors within each block, preserving words and layout."""
    result = []
    for record in records:
        lines = [dict(row) for row in record['lines']]
        for block in blocks(lines):
            values = [r['quantity'] for r in block]
            for row, i in zip(block, rng.permutation(len(block))):
                row['quantity'] = values[int(i)]
        result.append({**record, 'lines': lines})
    return result
