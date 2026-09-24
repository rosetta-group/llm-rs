import math
import unittest
from voynich.description_length import CharacterPrior
from voynich.piece_admission import admit, _cond_bits


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.prior = CharacterPrior.fit(['la casa e la cosa'] * 50)
        self.lookup = {c: i for i, c in enumerate(self.prior.alphabet)}

    def test_conditional_bits_add_up(self):
        whole = _cond_bits(self.prior, self.lookup, '', 'lacasa')
        split = _cond_bits(self.prior, self.lookup, '', 'lac') + _cond_bits(self.prior, self.lookup, 'lac', 'asa')
        self.assertAlmostEqual(whole, split)

    def test_admits_a_rare_whole_piece_that_fits_context(self):
        # Letters l a c a s a; the token 'Z' (twice) is read as the known split p:Q + s:R -> 'x','x',
        # which the prior dislikes; a new whole piece u:Z -> 'c' fits.
        mapping = {'u:L': 'l', 'u:A': 'a', 'u:S': 's', 'p:Q': 'x', 's:R': 'x'}
        tokens = ['L', 'A', 'QR', 'A', 'S', 'A'] * 2
        segmentation = [('L',), ('A',), ('Q', 'R'), ('A',), ('S',), ('A',)] * 2
        new, record = admit(tokens, segmentation, mapping, self.prior, max_count=5, threshold=10.)
        self.assertEqual(new.get('u:QR'), 'c')
        self.assertIn('u:QR', record['admitted_units'])
        none, rec = admit(tokens, segmentation, mapping, self.prior, threshold=math.inf)
        self.assertEqual(none, mapping); self.assertEqual(rec['admitted'], 0)


if __name__ == '__main__':
    unittest.main()
