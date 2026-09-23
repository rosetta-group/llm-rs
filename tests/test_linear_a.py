import unittest

import numpy as np

from linear_a import arithmetic, contexts
from linear_a.matching import BigramNull, Matcher
from linear_a.spelling import parse_syllabic, render, spell


class LinearATest(unittest.TestCase):
    def test_spelling_follows_linear_b_conventions(self):
        self.assertEqual(render(spell("Knōssós")), "ko-no-so")
        self.assertEqual(render(spell("Phaistós")), "pa-i-to")
        self.assertEqual(render(spell("khrusós")), "ku-ru-so")
        self.assertEqual(render(spell("xénwia")), "ke-se-wi-ja")
        self.assertEqual(render(spell("nfr")), "n?-p?-r?")


    def test_parse_syllabic_rejects_unknown_signs(self):
        self.assertEqual(render(parse_syllabic("qe-ra2-u")), "ke-ra-u")
        self.assertIsNone(parse_syllabic("*79-su"))
        self.assertIsNone(parse_syllabic("A306-tu"))


    def test_matcher_distance_and_unknown_vowel(self):
        lexicon = {parse_syllabic("ma-ra-to"): ["x"], (("n", "?"), ("p", "?")): ["nfr"]}
        matcher = Matcher({"L": lexicon})
        self.assertEqual(matcher.distances([parse_syllabic("ma-ra-to")], "L")[0], 0.0)
        # One vowel change in three syllables: 0.5 / 3.
        self.assertAlmostEqual(matcher.distances([parse_syllabic("ma-ra-tu")], "L")[0], 0.5 / 3)
        self.assertEqual(matcher.distances([parse_syllabic("ne-pa")], "L")[0], 0.0)


    def test_bigram_null_keeps_length(self):
        words = [parse_syllabic(w) for w in ["ku-ro", "ki-ro", "pa-i-to", "a-di-ki-te"]]
        null = BigramNull(words, np.random.default_rng(0))
        self.assertTrue(all(len(null.sample(n)) == n for n in [2, 3, 4, 2, 3]))


    def test_ku_ro_sums_lines_since_last_total(self):
        text = ("\U0001063e\U00010609\U00010626\U00010633\U0001064d\n"
                "\U00010619\U0001060d\U0001010b\n\U00010603\U00010639\U00010114\U0001010c\n"
                "\U00010642\U00010601\U00010111\U00010107")
        rows = arithmetic.check_totals({"HT 13": {"unicode_text": text}})
        self.assertEqual(rows, [{"document": "HT 13", "stated": 21, "sum": 61, "difference": -40}])

    def test_linear_b_line_labels(self):
        items = contexts.linear_b_line_items(".2        di-ka-ta-jo  /  di-we    OLE    S   1")
        labelled = contexts.label_line_items(items, first_line=False)
        self.assertEqual(labelled, [("di-ka-ta-jo", "other"), ("di-we", "entry")])
        items = contexts.linear_b_line_items(".1   de-u-ki-jo-jo   'me-no'")
        self.assertEqual(contexts.label_line_items(items, first_line=True),
                         [("de-u-ki-jo-jo", "header")])

    def test_logogram_label_and_majority(self):
        items = [("word", "a"), ("logogram", "VIN"), ("word", "b"), ("logogram", "VIR"),
                 ("number", None)]
        self.assertEqual(contexts.label_line_items(items, first_line=False),
                         [("a", "logogram"), ("b", "entry")])
        self.assertEqual(contexts.type_labels([("x", "entry"), ("x", "other"), ("x", "other")]),
                         {"x": "other"})


if __name__ == "__main__":
    unittest.main()
