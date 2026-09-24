import collections
import copy
import unittest

import numpy as np

from linear_a import ledger
from experiments.linear_a_ledger import make_record, preflight


class LedgerTests(unittest.TestCase):
    def test_b_units_damage_and_distinct_commodities(self):
        self.assertEqual(ledger.b_line('.1 a-ta VIR 2 OVIS:m 3')['quantity'], {'VIR': 2, 'OVIS:m': 3})
        for s in ('.1 a-ta GRA 1 T 3', '.1 a-ta VIR 2[', '.1 a-ta VIR 2̣',
                  '.1 a-ta [ VIR 2 ]', '.1 a-ta VIR OVIS:m 2', '.1 a-ta VIR 2 VIR 3'):
            self.assertEqual(ledger.b_line(s)['kind'], 'barrier', s)
        self.assertEqual(ledger.b_line('.B o OVIS:m 3')['words'], ['o'])

    def test_a_fraction_gap_unknown_sign(self):
        table = {'AB81': ('ku', 0), 'AB02': ('ro', 0), 'AB56': ('*56', 0)}
        s = '\U00010642\U00010601\U00010110\U00010109'
        r = ledger.a_line(s, table)
        self.assertEqual(r['quantity'], {'UNSPECIFIED': 13})
        for ending in ('\U00010746', '\U0001076b'):
            self.assertEqual(ledger.a_line(s+ending, table)['kind'], 'barrier')
        self.assertEqual(ledger.a_line('\U00010630\U00010108', table)['words'], ['AB56'])

    def test_target_quantity_never_used(self):
        b = [{'quantity': {'G': v}} for v in (2, 3, 5)]
        self.assertEqual(ledger.prediction(b, 2, 'sum_before'), {'G': 5})
        b[2]['quantity'] = {'G': 99999}
        self.assertEqual(ledger.prediction(b, 2, 'sum_before'), {'G': 5})
        self.assertEqual(ledger.prediction(b, 0, 'sum_before'), None)

    def test_operations_cannot_mix_commodities_or_fit_negative_balance(self):
        b = [{'quantity': {'G': 2}}, {'quantity': {'O': 3}}, {'quantity': {'G': 5}}]
        self.assertEqual(ledger.prediction(b, 2, 'sum_before'), {'G': 2, 'O': 3})
        self.assertIsNone(ledger.prediction(b, 2, 'balance_before'))
        b[1]['quantity'] = {'G': 3}
        self.assertIsNone(ledger.prediction(b, 2, 'balance_before'))
        b[0]['quantity'] = {'G': 10}
        self.assertEqual(ledger.prediction(b, 2, 'balance_before'), {'G': 7})

    def records(self):
        records = []
        for i in range(30):
            values = (17+i, 41+i*3, 58+i*4)
            rows = [{'kind': 'row', 'words': [f'person-{i}-{j}' if j<2 else 'total'],
                     'quantity': {'G': v}} for j,v in enumerate(values)]
            records.append(make_record(str(i), 'OBJ'+str(i), rows, {}))
        return records

    def test_blind_cross_object_recovery(self):
        records = self.records()
        examples = ledger.examples(records)
        models, out = ledger.cross_validate(examples)
        total = ledger.opaque('total', 'W')
        positives = [r for r in out if total in r['words']]
        self.assertEqual(len(positives), 30)
        self.assertTrue(all(r['prediction']==r['truth'] for r in positives))
        self.assertTrue(all(set(m)=={total} for m in models.values()))

    def test_damage_bounds_blocks_and_no_joins_invented(self):
        record = self.records()[0]
        record['lines'].insert(1, {'kind': 'barrier', 'reason': 'damage'})
        self.assertEqual(ledger.examples([record]), [])

    def test_quantity_shuffle_preserves_records_and_inputs(self):
        records = self.records()
        original = copy.deepcopy(records)
        changed = ledger.shuffled_records(records, np.random.default_rng(42))
        self.assertEqual(records, original)
        key = ledger.opaque('G', 'C')
        for a, b in zip(records, changed):
            self.assertEqual([r['words'] for r in a['lines']], [r['words'] for r in b['lines']])
            self.assertEqual(collections.Counter(r['quantity'][key] for r in a['lines']),
                             collections.Counter(r['quantity'][key] for r in b['lines']))
        _, out = ledger.cross_validate(ledger.examples(changed))
        self.assertFalse(any(r['prediction'] is not None for r in out))

    def test_repeated_faces_cannot_supply_three_training_objects(self):
        rows = ledger.examples(self.records()[:1])
        self.assertEqual(ledger.fit(rows*10), {})

    def test_disagreeing_rules_abstain(self):
        r = {'words': ['W'], 'candidates': {'sum_before': {'G': 5}, 'sum_after': {'G': 6}}}
        model = {'W': [{'operation': 'sum_before'}, {'operation': 'sum_after'}]}
        self.assertIsNone(ledger.predict(r, model)['prediction'])

    def test_preflight_rejects_impossible_control_before_fitting(self):
        records = self.records()
        small = preflight({'linear_b_control': records[:6], 'linear_a': records})
        self.assertFalse(small['passed'])
        self.assertEqual(len(small['reasons']), 2)
        self.assertTrue(preflight({'linear_b_control': records, 'linear_a': records[:12]})['passed'])


if __name__ == '__main__':
    unittest.main()
