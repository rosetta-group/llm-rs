import unittest
from experiments.voynich_suspects import neighbours, near_hapax_rate


class SuspectTests(unittest.TestCase):
    def test_neighbours_are_exactly_one_edit(self):
        self.assertEqual(neighbours('dairn', ['daiin', 'dain', 'chedy', 'dy']), ['daiin', 'dain'])
        self.assertEqual(neighbours('daiin', ['daiin']), [])

    def test_near_hapax_rate(self):
        words = ['daiin'] * 5 + ['dairn', 'chedy']
        r = near_hapax_rate(words)
        self.assertAlmostEqual(r['near_hapax_share_of_tokens'], 1 / 7)
        self.assertAlmostEqual(r['hapax_share_of_types'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
