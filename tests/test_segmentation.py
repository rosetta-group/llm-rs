import unittest

from voynich.segmentation import Segmenter, fit, boundaries
from experiments.segmentation import metrics


class SegmentationTests(unittest.TestCase):
    def test_lexicon_adds_unseen_inflections(self):
        model = fit(["i gatti dormono"] * 30, ["gattini"])
        self.assertEqual(Segmenter(model, alpha=1).segment("igattinidormono"), "i gattini dormono")

    def test_unknown_forms_do_not_lose_characters(self):
        model = fit(["la casa"], ["la", "casa"])
        result = Segmenter(model).segment("laxqztcasax")
        self.assertEqual(result.replace(" ", ""), "laxqztcasax")
        self.assertEqual(Segmenter(model).segment(""), "")
        with self.assertRaises(ValueError): Segmenter(model).segment("la casa")

    def test_word_transitions_disambiguate_equal_unigram_paths(self):
        model = fit(["ab c", "a bc"] * 10 + ["x a bc"] * 30, [])
        self.assertEqual(Segmenter(model, bigram=.9).segment("xabc"), "x a bc")

    def test_boundaries_and_word_error_are_distinct(self):
        self.assertEqual(boundaries("ab c de"), {2, 3})
        result = metrics("abc de", "ab c de")
        self.assertEqual((result['tp'], result['fn'], result['fp']), (1, 1, 0))
        with self.assertRaises(ValueError): metrics("ab x", "ab c")
