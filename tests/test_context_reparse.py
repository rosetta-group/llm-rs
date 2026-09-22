import itertools
import unittest

from voynich.context_reparse import options, reparse, score
from voynich.description_length import CharacterPrior


class ContextReparseTests(unittest.TestCase):
    def test_matches_exhaustive_paths_and_length_code(self):
        prior = CharacterPrior.fit(['abbaaba' * 20], order=3, alphabet='ab')
        mapping = {'u:xy': 'a', 'p:x': 'a', 's:y': 'b', 'u:z': 'b', 'u:q': 'a'}
        tokens = ['xy', 'z', 'xy', 'q']
        paths = [tuple(o[0] for o in path) for path in itertools.product(*(options(t, mapping) for t in tokens))]
        expected = min(score(path, mapping, prior) for path in paths)
        result = reparse(tokens, mapping, prior, width=1024)
        self.assertAlmostEqual(result['bits'], expected)
        self.assertAlmostEqual(result['bits'], score(result['segmentation'], mapping, prior))
        self.assertEqual([''.join(p) for p in result['segmentation']], tokens)

    def test_empty_and_unknown(self):
        prior = CharacterPrior.fit(['ab'], order=1, alphabet='ab')
        self.assertEqual(reparse([], {}, prior)['bits'], prior.bits(''))
        with self.assertRaises(ValueError):
            reparse(['x'], {}, prior)
        with self.assertRaises(ValueError):
            reparse(['x'], {'u:x': 'ab'}, prior)

    def test_role_homophones_are_charged(self):
        prior = CharacterPrior.fit(['ab' * 10], order=2, alphabet='ab')
        mapping = {'u:x': 'a', 'u:z': 'a', 's:y': 'b', 'p:q': 'a'}
        self.assertAlmostEqual(score([('x',), ('z',)], mapping, prior), prior.bits('aa') + 2)
