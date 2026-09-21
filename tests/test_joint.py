import itertools
import unittest

import numpy as np

from voynich.description_length import CharacterPrior
from voynich.joint_segments import build_parses, candidate_pieces, forward_backward, resegment
from voynich.variable_units import (anneal, bigram_options, induce_pieces, objective, refine,
                                    variable_beam, variable_scores)


class VariableUnitTests(unittest.TestCase):
    def setUp(self):
        self.prior = CharacterPrior.fit(['abbaababaabbab' * 3] * 5, order=3, alphabet='ab')

    def test_partial_score_breaks_context_at_unmapped_units(self):
        prior = self.prior
        costs = np.concatenate(prior.logs)
        offsets = np.array([sum(len(p) for p in prior.logs[:n]) for n in range(prior.order)], dtype=np.int64)
        values = np.array([0, 1, 2, 0], dtype=np.int32)
        freq = np.array([2, 1, 1], dtype=np.int64)
        first = np.array([[0, -1, 1]], dtype=np.int16)
        second = np.array([[-1, -1, -1]], dtype=np.int16)
        got = variable_scores(first, second, values, freq, costs, offsets, 2, 3, 2., 3.)[0]
        # letters: a, (gap), b, a  -> a alone, b alone (context reset), a given b; two mapped units cost 2 bits each
        expected = prior.logs[0][0] + prior.logs[0][1] + prior.logs[1][1 * 2 + 0] + 2 * 2.
        self.assertAlmostEqual(got, expected)

    def test_two_letter_unit_and_collapse_residual(self):
        prior = self.prior
        costs = np.concatenate(prior.logs)
        offsets = np.array([sum(len(p) for p in prior.logs[:n]) for n in range(prior.order)], dtype=np.int64)
        values = np.array([0, 1], dtype=np.int32)
        freq = np.array([1, 1], dtype=np.int64)
        first = np.array([[0, 0]], dtype=np.int16)
        second = np.array([[1, -1]], dtype=np.int16)
        got = variable_scores(first, second, values, freq, costs, offsets, 2, 3, 2., 3.)[0]
        expected = prior.logs[0][0] + prior.logs[1][0 * 2 + 1] + prior.logs[2][(0 * 2 + 1) * 2 + 0] + 3. + 2.
        self.assertAlmostEqual(got, expected)
        collapsed = variable_scores(np.array([[0, 0]], dtype=np.int16), np.array([[-1, -1]], dtype=np.int16),
                                    values, freq, costs, offsets, 2, 3, 2., 3.)[0]
        plain = prior.logs[0][0] + prior.logs[1][0] + 2 * 2.
        self.assertAlmostEqual(collapsed, plain + 2 * np.log2(2))

    def test_beam_and_refine_recover_a_toy_substitution(self):
        text = 'abbaababaabbab' * 3
        units = [{'a': 'X', 'b': 'Y'}[c] for c in text]
        result = variable_beam(units, self.prior, (), width=8)
        self.assertEqual(result['recovered'], text)
        wrong = {'X': 'b', 'Y': 'a'}
        refined = refine(units, self.prior, wrong, (), kicks=2, cap=10)
        self.assertLessEqual(refined['search_bits'], objective(units, self.prior, wrong))
        annealed = anneal(units, self.prior, None, proposals=2000, restarts=2)
        self.assertIn(annealed['recovered'], (text, text.translate(str.maketrans('ab', 'ba'))))
        self.assertEqual(len(bigram_options(self.prior, 3)), 3)

    def test_piece_induction_splits_concatenated_tokens(self):
        rng = np.random.default_rng(0)
        # many pieces, so two-piece tokens are mostly unique and only splitting shares statistics
        pieces = [a + b for a in 'qkdocst' for b in 'yeoi'][:20]
        tokens = [''.join(rng.choice(pieces, size=2)) for _ in range(600)] + list(rng.choice(pieces, size=200))
        result = induce_pieces(tokens, roles=True)
        doubles = [t for t in tokens if len(t) == 4]
        correct = sum(result['segmentation'][t] == (t[:2], t[2:]) for t in doubles)
        self.assertGreater(correct / len(doubles), .9)
        self.assertTrue(all(result['segmentation'][t] == (t,) for t in tokens if len(t) == 2))


class JointSegmentTests(unittest.TestCase):
    def test_forward_backward_matches_brute_force(self):
        base = 2; rng = np.random.default_rng(0)
        tri = rng.uniform(size=(base, base, base)); tri /= tri.sum(axis=2, keepdims=True)
        tokens = ['ab', 'b', 'ab', 'aab']; pieces = {'a', 'b', 'ab', 'aab', 'aa'}
        inventory, (starts, lengths, first, second) = build_parses(tokens, pieces)
        e = []
        for _ in range(3):
            m = rng.uniform(size=(base, len(inventory))); m /= m.sum(axis=1, keepdims=True); e.append(m)
        cu, cp, cs, parse_post, lp1, lp2, ll = forward_backward(starts, lengths, first, second, base, tri, *e, np.ones((base, base)) / 4)
        total = 0.; bf_parse = np.zeros(len(lengths)); bf = [np.zeros_like(m) for m in e]
        for choice in itertools.product(*[range(starts[t], starts[t + 1]) for t in range(len(tokens))]):
            n = sum(lengths[p] for p in choice)
            for letters in itertools.product(range(base), repeat=n):
                for x1 in range(base):
                    for x2 in range(base):
                        p = 1. / base ** 2; prev = (x1, x2); pos = 0
                        for pi in choice:
                            if lengths[pi] == 1:
                                y = letters[pos]; p *= tri[prev[0], prev[1], y] * e[0][y, first[pi]]; prev = (prev[1], y); pos += 1
                            else:
                                a, b = letters[pos], letters[pos + 1]
                                p *= tri[prev[0], prev[1], a] * e[1][a, first[pi]] * tri[prev[1], a, b] * e[2][b, second[pi]]
                                prev = (a, b); pos += 2
                        total += p; pos = 0
                        for pi in choice:
                            bf_parse[pi] += p
                            if lengths[pi] == 1: bf[0][letters[pos], first[pi]] += p; pos += 1
                            else: bf[1][letters[pos], first[pi]] += p; bf[2][letters[pos + 1], second[pi]] += p; pos += 2
        self.assertAlmostEqual(float(ll), float(np.log(total)))
        np.testing.assert_allclose(parse_post, bf_parse / total, atol=1e-12)
        for got, want in zip((cu, cp, cs), bf):
            np.testing.assert_allclose(got, want / total, atol=1e-12)

    def test_whole_parse_is_gated_and_resegment_prefers_known_units(self):
        tokens = ['xy'] * 5 + ['x'] * 5 + ['y'] * 5 + ['zz']
        pieces = candidate_pieces(tokens, minimum=3)
        self.assertIn('x', pieces); self.assertIn('y', pieces); self.assertNotIn('zz', pieces)
        inventory, (starts, lengths, first, second) = build_parses(tokens, pieces)
        # 'xy' occurs 5 times so it keeps a whole parse and a split; 'zz' only has its fallback whole parse
        self.assertEqual(starts[1] - starts[0], 2)
        self.assertEqual(starts[-1] - starts[-2], 1)
        prior = CharacterPrior.fit(['abab' * 10], order=2, alphabet='ab')
        mapping = {'p:x': 'a', 's:y': 'b', 'u:x': 'a', 'u:y': 'b'}
        result = resegment(['xy', 'x', 'y', 'zz'], pieces | {'xy'}, mapping, prior)
        self.assertEqual(result['segmentation'][0], ('x', 'y'))
        self.assertTrue(result['recovered'].startswith('abab'))
        self.assertTrue(result['recovered'].endswith('?'))


if __name__ == '__main__':
    unittest.main()
