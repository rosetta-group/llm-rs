import math
import unittest

from experiments.boundaries import segment


class BoundaryTests(unittest.TestCase):
    def test_penalty_controls_merging_without_changing_letters(self):
        words = "nel mezzo del cammin di nostra vita".split()
        text = "".join(words)
        costs = {w: -math.log(1 / 1000) for w in words}
        original = segment(text, costs, .5)
        calibrated = segment(text, costs, 2.)
        self.assertNotEqual(original, " ".join(words))
        self.assertEqual(calibrated, " ".join(words))
        self.assertEqual(original.replace(" ", ""), text)
        self.assertEqual(calibrated.replace(" ", ""), text)
