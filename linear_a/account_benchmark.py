"""Exact arithmetic on manually source-annotated accounts, not a language learner."""
from fractions import Fraction

# Relative measures only. Ventris & Chadwick (1956), p. 55.
UNITS = {
    'count': {'ONE': Fraction(1)},
    'weight': {'L': Fraction(1), 'M': Fraction(1, 30), 'N': Fraction(1, 120)},
    'liquid': {'BASE': Fraction(1), 'S': Fraction(1, 3), 'V': Fraction(1, 18), 'Z': Fraction(1, 72)},
    'dry': {'BASE': Fraction(1), 'T': Fraction(1, 10), 'V': Fraction(1, 60), 'Z': Fraction(1, 240)},
}


def quantity(row, strict=True):
    units = UNITS[row['dimension']]
    if not row['components']:
        return None
    value = Fraction(0)
    for component in row['components']:
        n = component['n']
        if n is None or (strict and component['certainty'] != 'certain'):
            return None
        if type(n) is not int or n < 0:
            raise ValueError('Quantities must be nonnegative integers')
        value += n * units[component['unit']]
    return value


def sum_items(account, strict=True):
    """Predict without accessing the target quantity or functional gold label."""
    rows = account['items']
    if not rows or (strict and not account['complete']):
        return None
    dimensions = {(r['dimension'], r['commodity']) for r in rows}
    if len(dimensions) != 1:
        return None
    values = [quantity(r, strict) for r in rows]
    return None if None in values else sum(values, Fraction(0))


def inspect(account):
    target = account['target']
    key = (target['dimension'], target['commodity'])
    compatible = all((r['dimension'], r['commodity']) == key for r in account['items'])
    result = {'id': account['id'], 'object': account['object']}
    for strict, mode in [(True, 'strict'), (False, 'transcribed')]:
        predicted = sum_items(account, strict) if compatible else None
        observed = quantity(target, strict)
        status = ('incompatible' if not compatible else 'not_evaluable' if predicted is None or observed is None
                  else 'balanced' if predicted == observed else 'mismatch')
        result[mode] = {'status': status, 'sum': None if predicted is None else str(predicted),
                        'target': None if observed is None else str(observed),
                        'residual_target_minus_sum': None if predicted is None or observed is None else str(observed-predicted)}
    return result
