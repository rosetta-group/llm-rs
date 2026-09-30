import unittest

from experiments.voynich_language_screen import BLOCKS, clean_tokens, outcome, passage


class CleaningTests(unittest.TestCase):
    def test_separators_and_unknown_glyphs(self):
        text = 'fachys.ykal,ar\nqo?dy.shol\x1cdaiin\x1d.o!k'
        self.assertEqual(clean_tokens(text), ['fachys', 'ykal', 'ar', 'shol', 'daiin'])

    def test_passage_slices_groups_in_order_and_reports_sources(self):
        documents = [dict(page='f1r', section='herbal', currier='A', text='a.c.d', split='train'),
                     dict(page='f2r', section='herbal', currier='B', text='x.y', split='train'),
                     dict(page='f3r', section='herbal', currier='A', text='e.f', split='validation')]
        tokens, sources = passage(documents, [('herbal', 'A', 1, 4)])
        self.assertEqual(tokens, ['c', 'd', 'e'])
        self.assertEqual(sources[0]['pages'], ['f1r', 'f3r'])
        with self.assertRaises(ValueError):
            passage(documents, [('herbal', 'A', 0, 9)])

    def test_blocks_do_not_overlap_within_a_group(self):
        spans = {}
        for block in BLOCKS:
            for role in ('fit', 'transfer'):
                for section, currier, start, stop in block[role]:
                    spans.setdefault((section, currier), []).append((start, stop if stop is not None else float('inf')))
        for group, ranges in spans.items():
            ranges.sort()
            for (_, a_stop), (b_start, _) in zip(ranges, ranges[1:]):
                self.assertLessEqual(a_stop, b_start, group)

    def test_outcome_classes(self):
        base = dict(accepted=None, reasons=[], inconclusive=False)
        self.assertEqual(outcome(dict(base, inconclusive=True)), 'inconclusive')
        self.assertEqual(outcome(dict(base, accepted='latin')), 'accepted')
        self.assertEqual(outcome(dict(base, reasons=['coverage', 'transfer_excess'])), 'unreadable')
        self.assertEqual(outcome(dict(base, reasons=['transfer_excess'])), 'rejected')


if __name__ == '__main__':
    unittest.main()
