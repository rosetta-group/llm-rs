"""Conditional dry-unit arithmetic, without promoting editorial readings to facts."""

SCALE_Z = {'BASE': 240, 'T': 24, 'V': 4, 'Z': 1}


def component_z(components):
    """Value of supplied components, never a restoration of missing components."""
    if set(components) - SCALE_Z.keys():
        raise ValueError('Unknown or non-dry unit')
    if any(type(n) is not int or n < 0 for n in components.values()):
        raise ValueError('Components must be nonnegative integers')
    return sum(SCALE_Z[unit] * n for unit, n in components.items())


def conversion_audit(rows):
    results = []
    for row in rows:
        value = component_z(row['components'])
        results.append({**row, 'conditional_component_z': value,
                        'published_minus_component_z': row['published_z'] - value,
                        'arithmetic_agrees': row['published_z'] == value,
                        'is_reconstructed_total': False})
    return results


def allocation_audit(account):
    """How much supplied quantity changes buckets if the sign changes function."""
    shift = sum(component_z(a['components']) for a in account['allocations'])
    exact = all(a['complete'] and a['quantity_secure'] for a in account['allocations'])
    blockers = [key for key in ('body_complete', 'total_secure', 'total_scope_secure')
                if not account[key]]
    if not exact:
        blockers.append('disputed_allocations_incomplete_or_uncertain')
    return {'object': account['object'], 'conditional_visible_shift_z': shift,
            'shift_exact': exact, 'blockers': blockers,
            'balance_comparison_eligible': not blockers,
            'preferred_reading': None}


def run(quantities, cases):
    if quantities['unit_scale_z'] != SCALE_Z:
        raise ValueError('Unit scale changed')
    conversions = conversion_audit(quantities['conversions'])
    accounts = [allocation_audit(a) for a in quantities['accounts']]
    return {'study': 'Theban *65/FAR source and quantity audit',
            'selected_objects': len(cases),
            'conversions': conversions,
            'conversion_discrepancies': sum(not c['arithmetic_agrees'] for c in conversions),
            'accounts': accounts,
            'eligible_accounts': sum(a['balance_comparison_eligible'] for a in accounts),
            'semantic_predictions': [], 'directed_edges': [],
            'linear_a_scored': False}
