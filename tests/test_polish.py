import unittest

from voynich.description_length import CharacterPrior
from voynich.lexical_polish import LexicalCost, polish, windows
from voynich.segmentation import Segmenter, fit


class LexicalPolishTests(unittest.TestCase):
    def setUp(self):
        texts = ['la casa bella', 'la bella casa', 'casa la la bella'] * 20
        self.segmenter = Segmenter(fit(texts, ['la', 'casa', 'bella']), alpha=1., bigram=.5, unknown=3.)
        self.lexical = LexicalCost(self.segmenter)

    def test_cost_prefers_lexicon_words_and_matches_segmenter_choice(self):
        self.assertLess(self.lexical.cost('lacasabella'), self.lexical.cost('lacasabellx'))
        self.assertLess(self.lexical.cost('casa'), self.lexical.cost('qzxv'))
        self.assertEqual(self.segmenter.segment('lacasabella'), 'la casa bella')

    def test_windows_merge_overlaps(self):
        self.assertEqual(windows([5, 8, 40], 100, 3), [[2, 12], [37, 44]])
        self.assertEqual(windows([0], 5, 10), [[0, 5]])

    def test_polish_repairs_a_single_wrong_unit(self):
        prior = CharacterPrior.fit(['lacasabella lacasabellacasa'] * 10, order=3, alphabet='abcels')
        plain = 'lacasabellalacasabellacasa'
        units = [f'u:{c}{i % 2}' for i, c in enumerate(plain)]  # two cipher units per letter
        truth = {u: c for u, c in zip(units, plain)}
        wrong = dict(truth); wrong['u:a0'] = 'e'  # every other 'a' becomes 'e'
        result = polish(units, prior, wrong, self.lexical, weight=1., radius=6, shortlist=5, sweeps=2, cap=60)
        self.assertEqual(result['recovered'], plain)
        self.assertEqual(len(result['changes']), 1)
        self.assertEqual(result['changes'][0]['unit'], 'u:a0')


if __name__ == '__main__':
    unittest.main()
