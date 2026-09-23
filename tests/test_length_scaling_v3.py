import unittest
from voynich import lexicon_repair, lexicon_repair_v2
from experiments.length_scaling_v3 import choose


class GlueTests(unittest.TestCase):
    def test_self_inclusion_lowers_ratio_of_a_whole_parsed_concatenation(self):
        # 'ab' is a concatenation of pieces 'a' and 'b' (letters x, z); EM parsed 3 of its 4 uses whole.
        seg = [('a', 'b')] + [('ab',)] * 3 + [('a', 'c'), ('d', 'b'), ('a', 'b')] + [('q',)]
        tokens = ['ab'] * 5 + ['ac', 'db', 'q']
        rec = 'xz' + 'y' * 3 + 'xw' + 'vz' + 'xz' + 'u'
        pieces = {'a', 'b', 'c', 'd', 'ab', 'q'}
        old = lexicon_repair.concatenation_ratios(tokens, pieces, seg, rec)['ab']
        new = lexicon_repair_v2.concatenation_ratios(tokens, pieces, seg, rec)['ab']
        self.assertLess(new, old)

    def test_selection(self):
        base = {10400: .032, 20800: .0325}
        self.assertIsNone(choose(dict(glue={10400: .031, 20800: .030}, reparse={10400: .04, 20800: .02}), base))
        self.assertEqual(choose(dict(glue={10400: .031, 20800: .026}, reparse={10400: .033, 20800: .025}), base), 'reparse')


if __name__ == '__main__':
    unittest.main()
