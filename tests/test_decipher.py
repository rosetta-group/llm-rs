import unittest

import numpy as np

from voynich.decipher import ALPHABET, edit_distance, encode, frequency_key, language_model, normalize, recover, restore_spaces


class DecipherTests(unittest.TestCase):
    def test_normalization_declares_lost_spelling(self):
        self.assertEqual(normalize("È già: Jack, whisky!"), "e gia iacc uuhiscy")
        self.assertEqual(normalize("a 123 α b"), "a b")
        self.assertEqual(len(set(ALPHABET)), len(ALPHABET))

    def test_frequency_baseline_keeps_spaces_and_inverts_rank(self):
        values = encode("zzzzz yyy xx", ALPHABET + " ")
        counts = np.zeros(len(ALPHABET)); counts[0:3] = [5, 3, 2]
        key = frequency_key(values, counts)
        decoded = "".join((ALPHABET + " ")[i] for i in key[values])
        self.assertEqual(decoded, "aaaaa bbb cc")

    def test_recovery_returns_a_bijection_without_original_inputs(self):
        probabilities, counts = language_model(["la casa e sulla collina e il mare e lontano"] * 5)
        result = recover("zzzzz yyy xx", probabilities, counts, restarts=1, steps=100)
        self.assertEqual(sorted(result["key"]), list(range(len(ALPHABET))))
        self.assertEqual([i for i,c in enumerate(result["recovered"]) if c == " "], [5, 9])
        self.assertEqual(len(result["recovered"]), 12)

    def test_segmentation_and_word_error_account_for_insertions(self):
        self.assertEqual(restore_spaces("lacasa", {"la": 20, "casa": 10}), "la casa")
        self.assertEqual(edit_distance("a b c".split(), "a x b c".split()), 1)


if __name__ == "__main__":
    unittest.main()
