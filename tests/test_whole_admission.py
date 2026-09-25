import unittest

import numpy as np

from voynich.description_length import CharacterPrior
from voynich.whole_admission import admit_wholes, apply_wholes, leave_one_out, whole_savings


class WholeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.prior = CharacterPrior.fit(['la casa e la cosa'] * 50)
        # Token 'QR' is read as the split p:Q + s:R -> 'xx'; the whole reading 'c' fits.
        self.mapping = {'u:L': 'l', 'u:A': 'a', 'u:S': 's', 'p:Q': 'x', 's:R': 'x'}

    def case(self, repeats):
        tokens = ['L', 'A', 'QR', 'A', 'S', 'A'] * repeats
        segmentation = [('L',), ('A',), ('Q', 'R'), ('A',), ('S',), ('A',)] * repeats
        return tokens, segmentation

    def test_admits_a_repeated_whole_piece_with_its_letter(self):
        tokens, segmentation = self.case(4)
        admitted, record = admit_wholes(tokens, segmentation, self.mapping, self.prior)
        self.assertEqual(admitted, {'QR': 'c'})
        self.assertEqual(record['tested'], 1)
        self.assertEqual(apply_wholes(tokens, segmentation, admitted)[2], ('QR',))

    def test_singletons_are_never_admitted(self):
        tokens, segmentation = self.case(1)
        admitted, record = admit_wholes(tokens, segmentation, self.mapping, self.prior)
        self.assertEqual(admitted, {})
        self.assertEqual(record['tested'], 0)

    def test_known_whole_entries_are_not_candidates(self):
        tokens, segmentation = self.case(3)
        self.assertEqual(whole_savings(tokens, segmentation, dict(self.mapping, **{'u:QR': 'c'}), self.prior), {})

    def test_leave_one_out_uses_other_occurrences_only(self):
        saving = np.array([[5., 0.], [0., 5.]])
        self.assertEqual(leave_one_out(saving), 0.)
        self.assertEqual(leave_one_out(np.array([[5., 0.], [4., 0.]])), 9.)


if __name__ == '__main__':
    unittest.main()
