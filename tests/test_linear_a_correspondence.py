import collections
import unittest

import numpy as np

from linear_a import correspondence as c
from linear_a.probes_v2 import monte_carlo


class CorrespondenceTests(unittest.TestCase):
    def test_literal_sign_distinctions_and_damage(self):
        self.assertNotEqual(c.literal('a-qa-ro'), c.literal('a-ka-ro'))
        self.assertNotEqual(c.literal('ta2-ta-re'), c.literal('ta-ta-re'))
        for raw in (']a-ta-ro', 'a-ta-ṛọ', '[a-ta-ro]', 'a-ta-ro?', 'a-*56-ro'):
            self.assertIsNone(c.literal(raw))
        self.assertEqual(c.literal('qa-qa-ro'), ('qa', 'qa', 'ro'))

    def test_b_editorial_spans_and_whole_tokens(self):
        rows = list(c.b_tokens({'content': '.1 a-ta-ro [ a-ti-ro ] di-de-ṛọ ka-sa-ro pa-ja-ṛọ'}))
        self.assertEqual([r['raw'] for r in rows], ['a-ta-ro', 'ka-sa-ro'])

    def test_b_broken_edge_does_not_taint_next_line(self):
        rows = list(c.b_tokens({'content': '.A ko-ma-we-to OVIS:m 19̣[\n.B a-ti-ro , / e-ko-so , o OVIS:m 71'}))
        self.assertEqual([r['raw'] for r in rows], ['ko-ma-we-to', 'a-ti-ro', 'e-ko-so'])

    def a_record(self):
        return {'word_source': 'sigla', 'words': ['a-ta-re'],
                'unicode_text': '\U00010607\U00010633\U00010619\U00010107',
                'signs': [{'n': i+1, 'type': t, 'reading': r, 'role': 'syllabogram', 'certain': True}
                          for i, (t, r) in enumerate(zip(('AB08', 'AB59', 'AB27'), ('a', 'ta', 're')))]}

    def test_a_layers_certainty_and_gap(self):
        record = self.a_record()
        self.assertEqual(len(list(c.a_tokens({'X': record}))), 1)
        record['unicode_text'] = '\U0001076b' + record['unicode_text']
        self.assertEqual(list(c.a_tokens({'X': record})), [])
        record = self.a_record()
        record['signs'][1]['certain'] = False
        self.assertEqual(list(c.a_tokens({'X': record})), [])
        record = self.a_record()
        record['words'] = ['a-*56-re']
        self.assertEqual(list(c.a_tokens({'X': record})), [])
        record = self.a_record()
        record['conflicts'] = [{'ours': 'a-ta-re', 'theirs': 'a-ti-re'}]
        self.assertEqual(list(c.a_tokens({'X': record})), [])
        record['conflicts'] = [{'ours': 'other-word', 'theirs': 'another-word'}]
        self.assertEqual(len(list(c.a_tokens({'X': record}))), 1)

    def test_permutation_preserves_conditioned_inventory(self):
        words = [tuple(w.split('-')) for w in ('a-ta-re', 'a-ti-ru', 'di-de-ro', 'ka-sa-ma', 'pa-ja-na')]
        for mode in ('length', 'length_onset'):
            buckets = c.groups(words, mode)
            shuffled = c.permute_endings(words, buckets, np.random.default_rng(42))
            self.assertEqual([w[:-1] for w in words], [w[:-1] for w in shuffled])
            for ids in buckets:
                self.assertEqual(collections.Counter(words[i][-1] for i in ids),
                                 collections.Counter(shuffled[i][-1] for i in ids))
            if mode == 'length_onset':
                self.assertEqual([c.onset(w[-1]) for w in words], [c.onset(w[-1]) for w in shuffled])

    def test_duplicates_and_ending_variants_get_one_vote(self):
        self.assertEqual(c.score([('a', 'ta', 're'), ('a', 'ta', 'ru'), ('a', 'ta', 'ru')],
                                 [('a', 'ta', 'ro')]), 1)

    def test_optimized_null_matches_full_permutation(self):
        a = sorted([('a', 'ta', 're'), ('a', 'ti', 'ru'), ('di', 'de', 'ro'),
                    ('a', 'ta', 'ra'), ('a', 'ti', 'ma'), ('ka', 'sa', 're')])
        b = [('a', 'ta', 'ro'), ('a', 'ti', 'ro')]
        for mode in ('length', 'length_onset'):
            rng = np.random.default_rng(731)
            expected = [c.score(c.permute_endings(a, c.groups(a, mode), rng), b) for _ in range(40)]
            actual, _ = c.null_scores(a, b, mode, 40, 731)
            self.assertEqual(actual, expected)

    def test_known_signal_and_tied_null(self):
        # All r endings: the onset-conditioned test must still detect deliberately planted
        # re/ru association and give p=1 when every ending is exchangeably re.
        b = [(f's{i}', 'ta', 'ro') for i in range(10)]
        a = [(f's{i}', 'ta', 're' if i < 10 else 'ro') for i in range(100)]
        for mode in ('length', 'length_onset'):
            values, _ = c.null_scores(a, b, mode, 999, 41)
            self.assertLess(monte_carlo(c.score(a, b), values, .05/7)['p'], .05/7)
            ties, _ = c.null_scores([w[:-1]+('re',) for w in a], b, mode, 199, 42)
            self.assertEqual(monte_carlo(10, ties, .05/7)['p'], 1.)


if __name__ == '__main__':
    unittest.main()
