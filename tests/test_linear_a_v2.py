import collections
import unittest

import numpy as np

from linear_a import probes, probes_v2
from linear_a.spelling import parse_syllabic as word


class RepairedProbesTest(unittest.TestCase):
    def test_duplicates_cannot_create_morphological_variation(self):
        w = word("ka-ta")
        np.testing.assert_array_equal(probes_v2.profile([w, w]), probes_v2.profile([w]))
        self.assertEqual(probes_v2.profile([w, w])[0], 0.)
        self.assertEqual(probes_v2.profile([w, word("ka-ti")])[0], 1.)
        self.assertEqual(probes_v2.profile([w, word("ka-ti")])[1], 0.)

    def test_strict_threshold_must_be_reachable(self):
        for n in (50, 100, 139):
            with self.assertRaises(ValueError):
                probes_v2.require_resolution(n, .05/7)
        probes_v2.require_resolution(140, .05/7)
        result = probes_v2.monte_carlo(1, [0]*9999, .05/7)
        self.assertEqual(result["p"], .0001)
        self.assertTrue(result["below_alpha"])

    def test_matching_is_unique_and_exact_length_even_with_small_pool(self):
        pools = [[word(w) for w in ws] for ws in
                 [["ka-ta", "mi-no", "ka-ta-ri"], ["pu-ro", "ta-ra", "te-ka-ro", "ka-ta-ri"]]]
        rng = np.random.default_rng(123)
        quotas = probes_v2.common_quotas(pools, 300, rng)
        for pool in pools:
            sample = probes_v2.matched_unique(pool, quotas, rng)
            self.assertEqual(len(sample), len(set(sample)))
            self.assertEqual(collections.Counter(map(len, sample)), quotas)
        with self.assertRaises(ValueError):
            probes_v2.matched_unique(pools[0], {3: 2}, rng)

    def test_fast_substitution_index_preserves_original_counts(self):
        rng = np.random.default_rng(89)
        alphabet = [('', 'a'), ('r', 'e'), ('r', 'o'), ('k', 'a'), ('t', 'i')]
        pool = sorted(set(tuple(alphabet[i] for i in rng.integers(0, 5, size=n))
                          for n in (2, 3, 4, 5) for _ in range(35)))
        target = pool[::2]
        queries = pool[1::2] + pool[1:4]  # repeated null words retain the original weighting
        buckets = probes_v2.by_length(target)
        original, _ = probes.one_substitutions(queries, buckets)
        self.assertEqual(probes_v2.SubstitutionIndex(target).count(queries), original)


if __name__ == "__main__":
    unittest.main()
