"""Leakage, independence and identifiability checks on synthetic examples only."""
import random
import unittest
from linear_a import person_role_control as c


def row(key, obj, **features):
    return {'id': key, 'object': obj, 'role': 'entry', 'quantity': 'one',
            'position': 'initial', 'repeated': False, **features}


class PersonRoleControlTests(unittest.TestCase):
    def test_all_faces_of_held_out_object_are_removed(self):
        rows = [row('front', 'tablet'), row('back', 'tablet'),
                row('other', 'other_tablet', role='heading')]
        labels = {'front': 'PERSON', 'back': 'PERSON', 'other': 'DESIGNATION'}
        result = c.evaluate(rows, labels, 'layout')
        for key in ('front', 'back'):
            self.assertEqual(result['predictions'][key]['support_objects'], [])
            self.assertIsNone(result['predictions'][key]['prediction'])

    def test_roster_length_cannot_supply_independent_objects(self):
        train = [row(str(i), 'long_roster') for i in range(100)]
        labels = {r['id']: 'PERSON' for r in train}
        result = c.predict(train, labels, row('test', 'held_out'), 'layout')
        self.assertEqual(result['reason'], 'too_few_objects')
        self.assertEqual(result['votes']['PERSON'], 1)
        train.append(row('d', 'second_object'))
        labels['d'] = 'DESIGNATION'
        result = c.predict(train, labels, row('test', 'held_out'), 'layout')
        self.assertEqual(result['reason'], 'tie')
        self.assertEqual(result['votes'], {'PERSON': 1, 'DESIGNATION': 1})

    def test_conflict_abstention_and_forced_diagnostic(self):
        rows = [row(str(i), f'o{i}') for i in range(4)]
        labels = {str(i): 'PERSON' if i < 3 else 'DESIGNATION' for i in range(4)}
        test = row('test', 'held_out')
        self.assertEqual(c.predict(rows, labels, test, 'layout')['reason'], 'conflicting_objects')
        self.assertEqual(c.predict(rows, labels, test, 'layout', True)['prediction'], 'PERSON')
        labels['3'] = 'PERSON'
        self.assertEqual(c.predict(rows, labels, test, 'layout')['prediction'], 'PERSON')
        self.assertEqual(c.predict(rows, labels, row('test', 'held_out', role='footer'),
                                   'layout')['reason'], 'unseen_signature')

    def test_equal_class_then_object_weight(self):
        rows = [row(str(i), 'long') for i in range(10)] + [row('p', 'short'), row('d', 'designation')]
        labels = {r['id']: 'PERSON' for r in rows}; labels['d'] = 'DESIGNATION'
        w = c.weights(rows, labels)
        self.assertAlmostEqual(sum(w[str(i)] for i in range(10)), 0.25)
        self.assertEqual(w['p'], 0.25)
        self.assertEqual(w['d'], 0.5)
        predictions = {r['id']: {'prediction': None} for r in rows}
        predictions['d']['prediction'] = 'DESIGNATION'
        m = c.metrics(rows, labels, predictions)
        self.assertEqual(m['coverage'], 0.5)
        self.assertEqual(m['balanced_recall'], 0.5)
        self.assertEqual(m['conditional_accuracy'], 1)
        self.assertFalse(c.performance_pass(m))

    def test_mixed_signature_limits_any_deterministic_rule(self):
        rows = [row('p', 'p'), row('d', 'd')]
        labels = {'p': 'PERSON', 'd': 'DESIGNATION'}
        self.assertEqual(c.ceiling(rows, labels, 'layout')['balanced_recall_upper_bound'], 0.5)
        rows[1]['position'] = 'noninitial'
        self.assertEqual(c.ceiling(rows, labels, 'layout')['balanced_recall_upper_bound'], 1)

    def test_opaque_spelling_renaming_preserves_features_and_predictions(self):
        def public(word):
            raw = f'{word} po-me VIR 1'
            case = dict(id='test', object='held_out', target=word, raw_line=raw,
                        entry_span=[0, len(raw)], entry_text=raw,
                        entry_words=[word, 'po-me'], target_index=0, role='entry', quantity='one')
            return c.public_case(case, raw + '\n' + word + ' VIR 1')
        a, b = public('ko-pe-re-u'), public('u-re-pe-ko')
        self.assertEqual(a, b)
        training = [row('x', 'x', repeated=True), row('y', 'y', repeated=True)]
        labels = {'x': 'PERSON', 'y': 'PERSON'}
        self.assertEqual(c.predict(training, labels, a, 'layout'),
                         c.predict(training, labels, b, 'layout'))
        self.assertEqual(set(a), {'id','object','role','quantity','position','repeated'})

    def test_erasure_damage_and_substrings_do_not_create_repetitions(self):
        words = c.literal_words('e-qe-ta ⟦e-qe-ta\ne-qe-ta⟧ e-qe-ta-e e-qe-[ta] e-qe-ṭạ')
        self.assertEqual(words.count('e-qe-ta'), 1)

    def test_negative_swaps_preserve_within_object_label_relationships(self):
        rows = [row('a', 'one'), row('b', 'one'), row('c', 'two'), row('d', 'two')]
        labels = {'a': 'PERSON', 'b': 'DESIGNATION', 'c': 'PERSON', 'd': 'PERSON'}
        for seed in range(10):
            swapped, objects = c.swap_object_labels(rows, labels, random.Random(seed))
            self.assertNotEqual(swapped['a'], swapped['b'])
            self.assertEqual(swapped['c'], swapped['d'])
            for r in rows:
                self.assertEqual(swapped[r['id']] != labels[r['id']], r['object'] in objects)


if __name__ == '__main__':
    unittest.main()
