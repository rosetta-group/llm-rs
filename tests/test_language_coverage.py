import tempfile
from pathlib import Path
import unittest
import zipfile

from experiments.language_coverage import MODELS, decisions
from experiments.language_coverage_sources import (
    catalan_documents, exact_take, exclude, normalize_source, pair_passages, rem_documents,
)


def row(ident, text, group=None):
    return dict(id=ident, document=ident.split(':')[0], group=group or ident.split(':')[0], text=text)


class LanguageCoverageTests(unittest.TestCase):
    def test_catalan_removes_annotations_and_groups_folio_sides(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'text.txt'
            path.write_text('form\tlemma\tPOS\tmorph\nTitle\t\tN\t\nFOL1r\t\tFOL\t\n'
                            'Retrau\twronglemma\tVB\t\n§\t\t§\t\nFOL1v\t\tFOL\t\n'
                            'seyor\t\tN\t\nFOL2r\t\tFOL\t\nJacme\t\tNPR\t\n')
            docs = list(catalan_documents(path))
            self.assertEqual([d['text'] for d in docs], ['Retrau seyor', 'Jacme'])
            self.assertEqual([d['group'] for d in docs], ['f0001', 'f0002'])

    def test_rem_selects_prose_surface_and_groups_variants(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'text.zip'
            with zipfile.ZipFile(path, 'w') as z:
                for ident, genre in [('M113', 'P'), ('M113Y', 'P'), ('M200', 'V')]:
                    z.writestr(ident+'.xml', f'<TEI xmlns="http://www.tei-c.org/ns/1.0"><title>Example</title>'
                               f'<classCode>{genre}</classCode><text><body><w norm="sinen" lemma="sin">ſinen</w>'
                               '</body></text></TEI>')
            docs = list(rem_documents(path))
            self.assertEqual([d['text'] for d in docs], ['sinen', 'sinen'])
            self.assertEqual([d['group'] for d in docs], ['M113', 'M113'])

    def test_historical_letters_are_explicitly_expanded(self):
        self.assertEqual(normalize_source('ſæ ßœ ð j k w à'), 'sae ssoe d i c uu a')

    def test_budget_is_exact_and_shortfall_fails(self):
        rows = [row('a', 'ab cd'), row('b', 'ef ghi')]
        self.assertEqual([r['text'] for r in exact_take(rows, 6)], ['abcd', 'ef'])
        with self.assertRaises(ValueError):
            exact_take(rows, 10)

    def test_leakage_across_reference_chunk_boundary(self):
        refs = [row('a:0', 'one two three four'), row('a:4', 'five six seven eight')]
        kept, removed = exclude([row('b', 'one two three four five six seven eight')], refs)
        self.assertFalse(kept)
        self.assertEqual(removed, ['b'])

    def test_pair_rejects_same_work_variant(self):
        rows = [row('M1', 'a'*5300, 'M1'), row('M1Y', 'b'*5300, 'M1'), row('M2', 'c'*5300, 'M2')]
        pair = pair_passages(rows)
        self.assertEqual([p['document'] for p in pair], ['M1', 'M2'])
        self.assertEqual([len(p['plaintext']) for p in pair], [5200, 5200])

    def test_leakage_created_by_joining_query_chunks(self):
        refs = [row('a', 'one two three four five six seven eight')]
        kept, removed = exclude([row('b:0', 'one two three four'), row('b:4', 'five six seven eight')], refs)
        self.assertEqual([r['id'] for r in kept], ['b:0'])
        self.assertEqual(removed, ['b:4'])

    def test_matched_systems_replace_models_and_omit_whole_language(self):
        scores = {m: dict(fit_excess=2., transfer_excess=2., coverage=1., cap_hit=False) for m in MODELS}
        scores['latin_broad'].update(fit_excess=.2, transfer_excess=.2)
        out = decisions(scores, 'latin')
        self.assertEqual(out['expanded']['accepted'], 'latin')
        self.assertIsNone(out['baseline']['accepted'])
        self.assertIsNone(out['omitted']['accepted'])
        self.assertNotEqual(out['omitted']['fit']['winner'], 'latin')

    def test_cap_is_inconclusive_not_successful_rejection(self):
        scores = {m: dict(fit_excess=2., transfer_excess=2., coverage=1., cap_hit=False) for m in MODELS}
        scores['catalan'].update(fit_excess=.2, transfer_excess=.2)
        scores['italian']['cap_hit'] = True
        for result in decisions(scores, 'catalan').values():
            self.assertTrue(result['inconclusive'])
            self.assertIsNone(result['accepted'])


if __name__ == '__main__':
    unittest.main()
