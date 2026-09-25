import copy
import json
import unittest
from pathlib import Path
from linear_a.corpus import load
from linear_a.person_slots import validate, summary, literal_occurrences


class PersonSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(Path('experiments/linear-a-person-slots/inventory.json').read_text())
        cls.corpus = load()

    def test_inventory_covers_words_signs_erasures_and_rows(self):
        validate(self.inventory, self.corpus)
        broken = copy.deepcopy(self.inventory)
        broken['faces']['HT 85b']['erased_sign_n'] = []
        with self.assertRaisesRegex(ValueError, 'Sign coverage'):
            validate(broken, self.corpus)

    def test_faces_not_independent_objects_and_logograms_not_names(self):
        result = summary(self.inventory)
        self.assertEqual((result['physical_objects'], result['faces']), (2, 4))
        self.assertEqual(result['by_face']['HT 85b']['lexical_count'], 8)
        self.assertEqual(result['by_face']['HT 85b']['logogram_count'], 3)
        broken = copy.deepcopy(self.inventory)
        logo = next(r for r in broken['rows'] if r['kind'] == 'logogram_count')
        logo['kind'] = 'lexical_count'
        with self.assertRaisesRegex(ValueError, 'Expected one graphic word'):
            validate(broken, self.corpus)

    def test_working_reading_does_not_rewrite_conflicting_source(self):
        row = next(r for r in self.inventory['rows'] if r['id'] == 'HT117a:10')
        self.assertEqual(row['working_reading'], 'te-ja-re')
        self.assertEqual(row['source_words'], ['te-*56-re'])
        self.assertEqual(row['source_signs'][1]['type'], 'AB56')
        self.assertEqual((row['physical_line'], row['logical_row']), ('5', 10))
        broken = copy.deepcopy(self.inventory)
        next(r for r in broken['rows'] if r['id'] == row['id'])['source_signs'][1]['type'] = 'AB57'
        with self.assertRaisesRegex(ValueError, 'Changed original signs'):
            validate(broken, self.corpus)

    def test_literal_queries_preserve_duplicates_conflicts_and_object_units(self):
        c = {'Xa': {'parent_object': 'X', 'words': ['te-ke', 'te-ki', 'te-ke'],
                    'conflicts': [{'reason': 'reading'}]},
             'Xb': {'parent_object': 'X', 'words': ['te-ke']}}
        hits = literal_occurrences(c, ['te-ke', 'te-ki', 'te-ja-re'])
        self.assertEqual((hits['te-ke']['occurrences'], hits['te-ke']['objects']), (3, 1))
        self.assertEqual(hits['te-ki']['occurrences'], 1)
        self.assertEqual(hits['te-ja-re']['occurrences'], 0)
        self.assertTrue(hits['te-ke']['hits'][0]['record_conflicts'])

    def test_missing_or_duplicate_logical_row_rejected(self):
        for duplicate in (False, True):
            broken = copy.deepcopy(self.inventory)
            row = broken['rows'].pop()
            if duplicate:
                broken['rows'].append(broken['rows'][-1])
            with self.assertRaises(ValueError):
                validate(broken, self.corpus)
