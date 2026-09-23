import unittest
import math
from experiments.word_segmentation_v3 import is_rubric, extract_book_v3, classify, choose
from voynich.segmentation import fit
from voynich.unknown_words import Spelling, SpellingSegmenter, join_elisions, unknown_rate

BODY = ' '.join(['parola'] * 10)


class WordSegmentationV3Tests(unittest.TestCase):
    def test_rubric_rule(self):
        self.assertTrue(is_rubric('IX \nCome Italo e Dardano vennero\n'))
        self.assertFalse(is_rubric('Il re venne.\nPoi parti.'))
        self.assertFalse(is_rubric('IX'))

    def test_extractor_drops_only_rubrics(self):
        html = f'<div id="box_esterno"><p>II\nCome si parti il mondo in tre parti ora</p><p>{BODY}</p></div>'
        rows, removed = extract_book_v3(html)
        self.assertEqual([r['text'] for r in rows], [BODY]); self.assertEqual(len(removed), 1)

    def test_attribution(self):
        lexicon = {'la', 'casa', 'ca', 'sa', 'dellanima'}
        d = classify('la ca sa dellanima', 'la casa della nima', lexicon)
        self.assertEqual(d['extra_spaces'], {'known_word': 1})
        self.assertEqual(d['missing_spaces'], {'known_token': 1})
        self.assertEqual(d['words'], {'correct': 1, 'split': 1, 'merged': 2})
        d = classify('la ca sa', 'la casa', {'la', 'ca', 'sa'})
        self.assertEqual(d['extra_spaces'], {'missing_form': 1}); self.assertEqual(d['oov_tokens_split'], 1)

    def test_spelling_is_a_distribution_over_short_words(self):
        spelling = Spelling(['la', 'casa', 'cosa', 'melano'], 3)
        letters = 'abcdefghilmnopqrstuvxyz$'
        self.assertAlmostEqual(sum(spelling.probability('^c', c) for c in letters), 1)
        self.assertLess(spelling.cost('casa'), spelling.cost('xqzv'))
        self.assertEqual(spelling.digest(), Spelling(['melano', 'cosa', 'casa', 'la'], 3).digest())

    def test_incremental_unknown_costs_match_direct_costs(self):
        model = fit(['la casa'], []); spelling = Spelling(model['lexicon'] + ['melano'], 3)
        segmenter = SpellingSegmenter(model, spelling, .05, alpha=1., bigram=.5)
        costs = segmenter.unknown_costs('xmelano', 1)
        self.assertAlmostEqual(costs[7], -math.log(.05) + spelling.cost('melano'))

    def test_known_words_still_segment_and_rate_is_checked(self):
        model = fit(['me la no casa'] * 3, [])
        spelling = Spelling(model['lexicon'] + ['melano', 'milano', 'molano'] * 1, 3)
        self.assertEqual(SpellingSegmenter(model, spelling, .5, alpha=1., bigram=.5).segment('lacasa'), 'la casa')
        with self.assertRaises(ValueError): SpellingSegmenter(model, spelling, 0, alpha=1.)

    def test_elision_join_and_rate(self):
        self.assertEqual(join_elisions('l innocenti e l cor dell anima'), 'linnocenti e l cor dellanima')
        self.assertEqual(unknown_rate(['a', 'a', 'b', 'c'], {'c'}), .25)

    def test_selection_rule(self):
        row = lambda c, h, v, m: dict(candidate=c, grades={s: dict(wer=x) for s, x in zip(('historical', 'verse', 'modern'), (h, v, m))})
        base = row(None, .15, .28, .07)
        self.assertIsNone(choose([base, row(dict(order=3, elision=False), .14, .26, .07)]))
        self.assertIsNone(choose([base, row(dict(order=3, elision=False), .10, .20, .09)]))
        a, b = dict(order=5, elision=False), dict(order=3, elision=True)
        self.assertEqual(choose([base, row(a, .10, .20, .07), row(b, .10, .20, .07)]), b)


if __name__ == '__main__':
    unittest.main()
