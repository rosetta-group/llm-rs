import unittest

from voynich.description_length import CharacterPrior
from voynich.half_admission import admit_halves, half_savings


class HalfAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.prior = CharacterPrior.fit(['la casa e la casa'] * 50)
        # 'QR' hides 'ca' as p:Q + s:R, but s:R is unknown, so it is read as the whole u:QR -> 'x'.
        self.mapping = {'u:L': 'l', 'u:A': 'a', 'u:S': 's', 'p:Q': 'c', 'u:QR': 'x'}

    def case(self, repeats):
        tokens = ['L', 'A', 'QR', 'S', 'A'] * repeats
        segmentation = [('L',), ('A',), ('QR',), ('S',), ('A',)] * repeats
        return tokens, segmentation

    def test_proposes_the_missing_half_with_its_letter(self):
        tokens, segmentation = self.case(4)
        admitted, record = admit_halves(tokens, segmentation, self.mapping, self.prior)
        self.assertEqual(admitted, {'s:R': 'a'})
        self.assertEqual(record['tested'], 1)

    def test_singletons_are_never_proposed(self):
        tokens, segmentation = self.case(1)
        self.assertEqual(admit_halves(tokens, segmentation, self.mapping, self.prior)[0], {})

    def test_known_or_unknown_pairs_are_not_candidates(self):
        tokens, segmentation = self.case(2)
        both = dict(self.mapping, **{'s:R': 'a'})
        self.assertEqual(half_savings(tokens, segmentation, both, self.prior), {})
        neither = {u: c for u, c in self.mapping.items() if u != 'p:Q'}
        self.assertEqual(half_savings(tokens, segmentation, neither, self.prior), {})


if __name__ == '__main__':
    unittest.main()
