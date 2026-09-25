import unittest
from unittest.mock import patch

from experiments import rejection_transfer_v2 as experiment


class FreshAfterResourceStopTests(unittest.TestCase):
    def test_source_id_excludes_previously_graded_even_without_text_match(self):
        rows = [dict(id='graded', words=['a' * 5200]), dict(id='fresh1', words=['b' * 5200]),
                dict(id='fresh2', words=['c' * 5200])]
        passages, excluded = experiment.pack_passages(rows, set(), set(), count=2, excluded_ids={'graded'})
        self.assertEqual([p['source_ids'] for p in passages], [['fresh1'], ['fresh2']])
        self.assertEqual(excluded, {'previously_graded_source_id': 1})

    def test_released_ids_are_scoped_to_their_language(self):
        released = {'one': dict(language='english', passages=[{'source_ids': ['a', 'b']}, {'source_ids': ['c']}]),
                    'two': dict(language='latin', passages=[{'source_ids': ['z']}])}
        with patch.object(experiment, 'read', return_value=released):
            self.assertEqual(experiment.used_source_ids('english'), {'a', 'b', 'c'})
            self.assertEqual(experiment.used_source_ids('latin'), {'z'})
            self.assertEqual(experiment.used_source_ids('italian'), set())


if __name__ == '__main__':
    unittest.main()
