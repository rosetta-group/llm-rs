import unittest
from experiments.word_segmentation_v3 import is_rubric, extract_book_v3, classify

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


if __name__ == '__main__':
    unittest.main()
