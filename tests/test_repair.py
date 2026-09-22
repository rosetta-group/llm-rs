import unittest

from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em
from voynich.lexicon_repair import concatenation_ratios, repair, unparsed_complements


class LexiconRepairTests(unittest.TestCase):
    def test_concatenation_ratio_separates_bigram_strings_from_one_letter_pieces(self):
        # 40 tokens 'AB' decoded as the split (A, B) -> 'xz'; the string 'AB' also sits in the lexicon as a whole piece.
        # Its whole-token count (40) equals what its halves produce as a bigram, so the ratio is about 1.
        # 'CD' occurs 10 times as a one-letter piece 'q'; its halves are prefix/suffix pieces used once each,
        # so the expected concatenation count is tiny and the ratio is large.
        tokens = ['AB'] * 40 + ['CD'] * 10 + ['CB', 'AD']
        segmentation = [('A', 'B')] * 40 + [('CD',)] * 10 + [('C', 'B'), ('A', 'D')]
        recovered = 'xz' * 40 + 'q' * 10 + 'xz' + 'xz'
        pieces = {'A', 'B', 'C', 'D', 'AB', 'CD'}
        ratios = concatenation_ratios(tokens, pieces, segmentation, recovered)
        # 42 'xz' bigrams; A carries 41 of the 42 prefix-x occurrences, B 41 of the 42 suffix-z ones
        self.assertAlmostEqual(ratios['AB'], 40 / (42 * (41 / 42) * (41 / 42)), places=6)
        self.assertGreater(ratios['CD'], 100)
        self.assertNotIn('CB', ratios)  # not a lexicon piece

    def test_unparsed_complements_only_come_from_tokens_without_a_parse(self):
        pieces = {'A', 'B', 'AB'}
        tokens = ['AB', 'A', 'AZ', 'AZ', 'QB']
        self.assertEqual(unparsed_complements(tokens, pieces, minimum=2), {'Z'})
        self.assertEqual(unparsed_complements(tokens, pieces, minimum=1), {'Z', 'Q'})
        self.assertEqual(unparsed_complements(['AB', 'A'], pieces, minimum=1), set())

    def test_repair_admits_the_missing_piece_and_records_each_pass(self):
        prior = CharacterPrior.fit(['abbaababaabbab' * 3] * 5, order=3, alphabet='ab')
        tokens = ['XY'] * 10 + ['X'] * 5 + ['Y'] * 5 + ['XZ'] * 2
        em = dict(restarts=1, iterations=5, seed=0, cap=60)
        first = joint_em(tokens, prior, minimum=3, **em)  # 'Z' occurs twice, below the threshold
        self.assertNotIn('Z', first['candidate_pieces'])
        repaired = repair(tokens, prior, first, theta=5., complement_minimum=2, passes=2, **em)
        self.assertIn('Z', repaired['candidate_pieces'])
        self.assertEqual(len(repaired['repair']['record']), 2)
        self.assertEqual(len(repaired['segmentation']), len(tokens))
        self.assertGreaterEqual(repaired['repair']['record'][0]['complements_admitted'], 1)


if __name__ == '__main__':
    unittest.main()
