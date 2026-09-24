import math
import unittest
from voynich.segmentation import fit
from voynich.unknown_words import Spelling
from voynich.unknown_words_v4 import MixtureSegmenter


class MixtureTests(unittest.TestCase):
    def test_mixture_cost_matches_direct_computation(self):
        model = fit(['la casa'], [])
        a, b = Spelling(['casa', 'cosa', 'la'], 3), Spelling(['erono', 'fusse', 'giano'], 5)
        seg = MixtureSegmenter(model, [a, b], [.5, .5], .05, alpha=1., bigram=.5)
        direct = -math.log(.05) - math.log(.5 * math.exp(-a.cost('erono')) + .5 * math.exp(-b.cost('erono')))
        self.assertAlmostEqual(seg.unknown_costs('xerono', 1)[6], direct)
        self.assertEqual(seg.segment('lacasa'), 'la casa')

    def test_weights_are_checked(self):
        model = fit(['la'], []); a = Spelling(['la'], 3)
        with self.assertRaises(ValueError): MixtureSegmenter(model, [a, a], [.7, .7], .05)
        with self.assertRaises(ValueError): MixtureSegmenter(model, [a, a], [1., 0.], .05)


if __name__ == '__main__':
    unittest.main()
