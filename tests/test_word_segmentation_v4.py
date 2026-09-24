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

    def test_sacchetti_extractor_drops_italic_argument(self):
        from experiments.word_segmentation_v4_fresh import extract_sacchetti, roman
        body = ' '.join(['parola'] * 10)
        html = f'<div id="box_esterno"><p><i>Lo re Federigo di Cicilia e trafitto con una storia</i></p><p>{body} <i>detto</i></p></div>'
        rows, rubrics = extract_sacchetti(html, 2)
        self.assertEqual([r['text'] for r in rows], [body + ' detto']); self.assertEqual(len(rubrics), 1)
        self.assertEqual((roman(4), roman(39), roman(40)), ('IV', 'XXXIX', 'XL'))


if __name__ == '__main__':
    unittest.main()
