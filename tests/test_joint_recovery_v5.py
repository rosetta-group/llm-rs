import unittest
from voynich.lexical_polish import LexicalCost
from voynich.lexical_polish_v3 import SpellingLexicalCost
from voynich.segmentation import Segmenter, fit
from voynich.unknown_words import Spelling, SpellingSegmenter


class RoundFiveTests(unittest.TestCase):
    def setUp(self):
        self.model = fit(['la casa e la cosa'] * 3, [])
        self.spelling = Spelling(self.model['lexicon'] + ['melano', 'milano'], 3)

    def test_known_text_costs_match_the_frozen_cost(self):
        v3 = SpellingSegmenter(self.model, self.spelling, .02, alpha=1., bigram=.5)
        old = Segmenter(self.model, alpha=1., bigram=.5)
        self.assertAlmostEqual(SpellingLexicalCost(v3).cost('lacasaelacosa'), LexicalCost(old).cost('lacasaelacosa'))

    def test_unknown_words_are_cheaper_than_the_flat_cost(self):
        v3 = SpellingSegmenter(self.model, self.spelling, .02, alpha=1., bigram=.5)
        old = Segmenter(self.model, alpha=1., bigram=.5)
        self.assertLess(SpellingLexicalCost(v3).cost('lamelano'), LexicalCost(old).cost('lamelano'))
        self.assertEqual(SpellingLexicalCost(v3).cost(''), 0.)


if __name__ == '__main__':
    unittest.main()
