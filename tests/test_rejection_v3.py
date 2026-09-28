import unittest

from voynich.description_length import CharacterPrior, integer_bits
from voynich.rejection import transfer as per_run
from voynich.rejection_v3 import rescore, transfer


class OneLengthCodeTests(unittest.TestCase):
    def setUp(self):
        self.prior = CharacterPrior.fit(['la casa e la cosa'] * 50)
        self.mapping = {'u:L': 'l', 'u:A': 'a', 'u:C': 'c', 'u:S': 's'}

    def test_without_gaps_the_score_is_unchanged(self):
        tokens = ['L', 'A', 'C', 'A', 'S', 'A'] * 5
        self.assertAlmostEqual(transfer(tokens, self.mapping, self.prior)['bits_per_letter'],
                               per_run(tokens, self.mapping, self.prior)['bits_per_letter'])

    def test_each_gap_no_longer_adds_a_length_code(self):
        tokens = (['L', 'A', 'C', 'A', 'S', 'A', 'Z'] * 6)[:-1]   # five unreadable tokens
        old, new = per_run(tokens, self.mapping, self.prior), transfer(tokens, self.mapping, self.prior)
        runs = [r for r in old['recovered'].split('?') if r]
        self.assertEqual(len(runs), 6)
        expected = (sum(integer_bits(len(r)) for r in runs) - integer_bits(sum(map(len, runs)))) / old['recovered_letters']
        self.assertAlmostEqual(old['bits_per_letter'] - new['bits_per_letter'], expected)
        self.assertAlmostEqual(rescore(old, self.prior)['bits_per_letter'], new['bits_per_letter'])
        self.assertEqual(new['recovered'], old['recovered'])


if __name__ == '__main__':
    unittest.main()
