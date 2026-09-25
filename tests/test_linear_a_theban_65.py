import copy
import unittest

from linear_a import theban_65 as t


def account():
    return {'object': 'synthetic', 'body_complete': True, 'total_secure': True,
            'total_scope_secure': True, 'allocations': [
                {'components': {'V': 1}, 'complete': True, 'quantity_secure': True}]}


class ThebanQuantityTests(unittest.TestCase):
    def test_dry_unit_equivalence(self):
        self.assertEqual(t.component_z({'BASE': 1}), t.component_z({'T': 10}))
        self.assertEqual(t.component_z({'T': 1}), t.component_z({'V': 6}))
        self.assertEqual(t.component_z({'V': 1}), t.component_z({'Z': 4}))

    def test_no_liquid_or_negative_components(self):
        for components in ({'S': 1}, {'V': -1}, {'Z': 1.5}, {'T': True}):
            with self.assertRaises(ValueError):
                t.component_z(components)

    def test_discrepancy_is_not_silently_repaired(self):
        rows = [{'components': {'T': 2, 'V': 1}, 'published_z': 99,
                 'complete_secure_total': False}]
        before = copy.deepcopy(rows)
        result = t.conversion_audit(rows)[0]
        self.assertEqual(result['conditional_component_z'], 52)
        self.assertEqual(result['published_minus_component_z'], 47)
        self.assertFalse(result['arithmetic_agrees'])
        self.assertFalse(result['is_reconstructed_total'])
        self.assertEqual(rows, before)

    def test_matching_components_do_not_reconstruct_total(self):
        result = t.conversion_audit([{'components': {'V': 2}, 'published_z': 8,
                                    'complete_secure_total': False}])[0]
        self.assertTrue(result['arithmetic_agrees'])
        self.assertFalse(result['is_reconstructed_total'])

    def test_each_account_blocker_prevents_comparison(self):
        for field in ('body_complete', 'total_secure', 'total_scope_secure'):
            data = account()
            data[field] = False
            result = t.allocation_audit(data)
            self.assertFalse(result['balance_comparison_eligible'])
            self.assertIn(field, result['blockers'])

    def test_missing_quantity_is_not_exact_zero(self):
        data = account()
        data['allocations'].append({'components': {}, 'complete': False,
                                    'quantity_secure': False})
        result = t.allocation_audit(data)
        self.assertEqual(result['conditional_visible_shift_z'], 4)
        self.assertFalse(result['shift_exact'])
        self.assertFalse(result['balance_comparison_eligible'])

    def test_doubtful_digit_is_conditional(self):
        data = account()
        data['allocations'][0]['quantity_secure'] = False
        result = t.allocation_audit(data)
        self.assertFalse(result['shift_exact'])
        self.assertIsNone(result['preferred_reading'])

    def test_complete_account_does_not_by_itself_label_sign(self):
        result = t.allocation_audit(account())
        self.assertTrue(result['balance_comparison_eligible'])
        self.assertTrue(result['shift_exact'])
        self.assertIsNone(result['preferred_reading'])


if __name__ == '__main__':
    unittest.main()
