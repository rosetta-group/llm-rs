from collections import Counter
import unittest

import numpy as np
import numba

from voynich.description_length import CharacterPrior
from voynich.rejection import decide
from voynich.rejection_development import frequency_copy, copying_diagnostics, decide_transfer
from voynich.variable_units import refine as legacy_refine, key_arrays
from voynich.variable_units_bounded import Scorer, pair_batches, refine


class BoundedRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = numba.get_num_threads()
        numba.set_num_threads(min(2, cls.old_threads))

    @classmethod
    def tearDownClass(cls):
        numba.set_num_threads(cls.old_threads)

    def setUp(self):
        self.prior = CharacterPrior.fit(['abacabaabcbabcac' * 20], order=4, alphabet='abc')

    def fixture(self, seed, size=8, length=75):
        rng = np.random.default_rng(seed)
        inventory = [str(i) for i in range(size)]
        units = inventory + list(rng.choice(inventory, length-size))
        mapping = {u: self.prior.alphabet[rng.integers(3)] for u in inventory}
        return units, mapping

    def test_pair_batches_preserve_order_and_boundaries(self):
        batches = list(pair_batches(5, 3))
        self.assertEqual([len(b) for b in batches], [3, 3, 3, 1])
        self.assertEqual(np.concatenate(batches).tolist(),
                         [[i, j] for i in range(5) for j in range(i+1, 5)])
        self.assertEqual(list(pair_batches(1, 3)), [])

    def test_every_incremental_proposal_matches_full_score(self):
        for seed in range(8):
            units, mapping = self.fixture(seed, size=11, length=113)
            scorer = Scorer(units, self.prior)
            first, second = key_arrays(scorer.inventory, mapping, self.prior)
            current = float(scorer.full(first, second)[0])
            pairs = np.concatenate(list(pair_batches(len(mapping), 7)))
            targets = np.concatenate((pairs[:, 0], np.repeat(np.arange(len(mapping)), 3)))
            partners = np.concatenate((pairs[:, 1], np.full(len(mapping)*3, -1)))
            replacements = np.concatenate((np.full(len(pairs), -1), np.tile(np.arange(3), len(mapping))))
            none = np.full(len(targets), -1)
            f, g = scorer.materialize(first, second, targets, partners, replacements, none)
            exact = scorer.full(f, g)
            approximate = scorer.approximate(first, current, targets, partners, replacements)
            np.testing.assert_allclose(exact, approximate, atol=1e-9, rtol=0)
            k, value, _ = scorer.best(first, second, current, targets, partners, replacements, none, True)
            self.assertEqual(k, int(np.argmin(exact)))
            self.assertEqual(value, exact[k])

    def test_overlapping_windows_and_uneven_homophones(self):
        units = ['x'] * 30 + ['y', 'z', 'y', 'z', 'w'] + ['x'] * 20
        mapping = {'x': 'a', 'y': 'b', 'z': 'b', 'w': 'c'}
        scorer = Scorer(units, self.prior)
        f, g = key_arrays(scorer.inventory, mapping, self.prior)
        current = float(scorer.full(f, g)[0])
        pairs = np.concatenate(list(pair_batches(4, 5)))
        none = np.full(len(pairs), -1)
        a, b = scorer.materialize(f, g, pairs[:, 0], pairs[:, 1], none, none)
        np.testing.assert_allclose(scorer.approximate(f, current, pairs[:, 0], pairs[:, 1], none),
                                   scorer.full(a, b), atol=1e-9, rtol=0)

    def test_complete_refinement_matches_legacy_for_both_backends(self):
        for seed in range(3):
            units, mapping = self.fixture(seed)
            old = legacy_refine(units, self.prior, mapping, kicks=1, kick_size=2, seed=seed, cap=60)
            self.assertFalse(old['cap_hit'])
            for backend in ('chunked', 'incremental'):
                new = refine(units, self.prior, mapping, kicks=1, kick_size=2, seed=seed,
                             cap=60, sweeps=None, max_evaluations=None, batch_size=3, backend=backend)
                self.assertEqual(new['stop_reason'], 'converged')
                for field in ('mapping', 'recovered', 'search_bits', 'evaluations', 'improvements', 'kicks'):
                    self.assertEqual(new[field], old[field], (seed, backend, field))

    def test_bigram_keys_fall_back_and_match_legacy(self):
        units, mapping = self.fixture(21, size=5, length=40)
        mapping['0'] = 'ab'
        old = legacy_refine(units, self.prior, mapping, bigrams=[(0, 1), (1, 2)], kicks=0, cap=60)
        new = refine(units, self.prior, mapping, bigrams=[(0, 1), (1, 2)], kicks=0,
                     cap=60, sweeps=None, max_evaluations=None, batch_size=2)
        self.assertEqual(new['backend'], 'chunked')
        self.assertEqual(new['mapping'], old['mapping'])
        self.assertEqual(new['search_bits'], old['search_bits'])

    def test_uniform_score_ties_retain_earliest_candidate(self):
        prior = CharacterPrior.fit(['abc' * 20], order=1, alphabet='abc')
        units = list('xyzxyz')
        mapping = dict(zip('xyz', 'abc'))
        old = legacy_refine(units, prior, mapping, kicks=0, cap=60)
        new = refine(units, prior, mapping, kicks=0, cap=60, batch_size=1)
        self.assertEqual(new['mapping'], old['mapping'])
        self.assertEqual(new['search_bits'], old['search_bits'])

    def test_work_limit_is_not_convergence_and_keeps_best_key(self):
        units, mapping = self.fixture(2)
        result = refine(units, self.prior, mapping, max_evaluations=0)
        self.assertEqual(result['mapping'], mapping)
        self.assertEqual(result['stop_reason'], 'evaluation_limit')
        self.assertEqual(result['evaluations'], 0)
        self.assertTrue(result['cap_hit'])
        self.assertEqual(result['status'], 'budget_exhausted')

    def test_time_and_sweep_limits_are_explicit(self):
        units, mapping = self.fixture(4)
        result = refine(units, self.prior, mapping, cap=0)
        self.assertEqual(result['stop_reason'], 'time_limit')
        result = refine(units, self.prior, mapping, sweeps=1, kicks=3, cap=60)
        self.assertEqual(result['sweeps'], 1)
        self.assertEqual(result['stop_reason'], 'sweep_limit')


class StrongerControlTests(unittest.TestCase):
    def test_inventory_frequencies_and_reproducibility(self):
        tokens = ['aa'] * 30 + ['bc'] * 20 + ['d'] * 10
        for seed in range(10):
            output = frequency_copy(tokens, seed)
            self.assertEqual(Counter(output), Counter(tokens))
            self.assertEqual(output, frequency_copy(tokens, seed))
            self.assertEqual(output, frequency_copy(tokens[::-1], seed))
        self.assertEqual(frequency_copy([], 0), [])

    def test_copying_increases_local_recurrence_at_fixed_frequencies(self):
        tokens = [str(i) for i in range(100)] * 20
        copied = copying_diagnostics(frequency_copy(tokens, 123, .9, 10), 10)
        random = copying_diagnostics(frequency_copy(tokens, 123, 0., 10), 10)
        self.assertGreater(copied['recent_repeat_rate'], random['recent_repeat_rate'] + .3)

    def test_development_rule_only_removes_fit_ceiling(self):
        rows = {'english': dict(fit_excess=.584, transfer_excess=.374, coverage=.989, cap_hit=False),
                'latin': dict(fit_excess=1.5, transfer_excess=2.3, coverage=.98, cap_hit=False)}
        self.assertIsNone(decide(rows)['accepted'])
        self.assertEqual(decide_transfer(rows)['accepted'], 'english')
        for field, value in [('coverage', .94), ('cap_hit', True), ('transfer_excess', .51)]:
            altered = {k: dict(v) for k, v in rows.items()}
            altered['english'][field] = value
            self.assertIsNone(decide_transfer(altered)['accepted'])
        altered = {k: dict(v) for k, v in rows.items()}
        altered['latin']['fit_excess'] = .6
        self.assertIn('fit_margin', decide_transfer(altered)['reasons'])


if __name__ == '__main__':
    unittest.main()
