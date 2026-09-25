"""Source-boundary regressions that could invalidate historical comparisons."""
import unittest
from experiments.language_expansion_sources import czech_document, occitan_document, CZECH, OCCITAN
from experiments.language_expansion import BASELINE, EXPANDED, MODELS, decisions
from experiments.language_coverage_sources import normalize_source


class LanguageExpansionTests(unittest.TestCase):
    def test_czech_metadata_never_enters_training(self):
        doc = czech_document('sample', '# title: Test title\n# originDate: 1440--1460\n# words: 3\nStarý text [...] zde.')
        self.assertNotIn('title', doc['text'])
        self.assertNotIn('1440', doc['text'])
        self.assertEqual(normalize_source(doc['text']), 'stary text zde')

    def test_czech_rejects_out_of_period_text(self):
        for date in ['1500--1520', '1890', '', '1200']:
            with self.subTest(date=date), self.assertRaises(ValueError):
                czech_document('wrong', f'# title: Wrong\n# originDate: {date}\nText')

    def test_occitan_hyphenation_and_editorial_heading(self):
        doc = occitan_document('Français_1049', 'Sur le trepas de robert de sicile comte\nde provence\nbenau-\nrada sancta [...]')
        self.assertEqual(normalize_source(doc['text']), 'benaurada sancta')
        with self.assertRaises(ValueError):
            occitan_document('Français_1049', 'Changed heading')

    def test_splits_keep_works_separate_and_exclude_parallel_manuscripts(self):
        for roles in (CZECH, OCCITAN):
            names = sum(roles.values(), [])
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(list(map(len, roles.values())), [4, 2, 2])
        self.assertFalse({'Français_13509', 'Additional_10323', 'Additional_21218'} & set(sum(OCCITAN.values(), [])))
        self.assertEqual(len(BASELINE), 6)
        self.assertEqual(len(EXPANDED), 8)
        self.assertEqual(len(MODELS), 8)
        self.assertEqual(BASELINE['latin'], 'latin_broad')
        self.assertEqual(BASELINE['german'], 'german_broad')

    def test_new_candidates_omitted_and_retained_as_whole_languages(self):
        for language in ('czech', 'occitan'):
            scores = {m: dict(fit_excess=2., transfer_excess=2., coverage=1., cap_hit=False) for m in MODELS}
            scores[language].update(fit_excess=.1, transfer_excess=.1)
            out = decisions(scores, language)
            self.assertEqual(out['expanded']['accepted'], language)
            self.assertIsNone(out['baseline']['accepted'])
            self.assertIsNone(out['omitted']['accepted'])
            self.assertNotEqual(out['omitted']['fit']['winner'], language)


if __name__ == '__main__':
    unittest.main()
