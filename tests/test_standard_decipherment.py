import itertools
import math
import unittest

import numpy as np

from voynich.description_length import CharacterPrior, description_length, infer_mapping
from voynich.homophonic import beam_search, expectation, partial_scores
from experiments.historical_sources import extract


class StandardDeciphermentTests(unittest.TestCase):
    def test_prior_is_conditional_and_charges_extra_characters(self):
        prior = CharacterPrior.fit(['ab aba bab'] * 4, order=3, alphabet='ab')
        for p in prior.probabilities:
            np.testing.assert_allclose(p.reshape(-1,2).sum(axis=1),1)
        self.assertGreater(prior.bits('ababab'),prior.bits('abab'))
        self.assertGreater(prior.bits('a'),0)

    def test_ambiguity_and_expansion_costs(self):
        prior = CharacterPrior.fit(['ababab']*20, order=3, alphabet='ab')
        exact = description_length('X Y X Y','units',{'X':'a','Y':'b'},prior)
        collapsed = description_length('X Y X Y','units',{'X':'a','Y':'a'},prior)
        self.assertEqual(exact['residual_bits'],0)
        self.assertEqual(collapsed['residual_bits'],4)
        expanded = description_length('X Y X Y','units',{'X':'ab','Y':'ab'},prior)
        self.assertGreater(expanded['key_bits'],exact['key_bits'])
        self.assertEqual(expanded['plaintext_characters'],8)
        ambiguous = description_length('X Y X','units',{'X':'a','Y':'aa'},prior)
        self.assertEqual(ambiguous['residual_bits'],2)

    def test_partial_context_breaks_at_gaps(self):
        prior = CharacterPrior.fit(['ababa'],order=3,alphabet='ab')
        keys = np.array([[0,-1,1]])
        actual = partial_scores(keys,np.array([0,1,2,0]),np.concatenate(prior.logs),np.array([0,2,6]),2,3)[0]
        expected = prior.logs[0][0]+prior.logs[0][1]+prior.logs[1][2]
        self.assertAlmostEqual(actual,expected)

    def test_full_beam_matches_exhaustive_homophonic_search(self):
        prior = CharacterPrior.fit(['abbaababaabb']*4,order=3,alphabet='ab')
        units = list('XYZXXYZ')
        result = beam_search(units,prior,width=8,max_homophones=3)
        options = [''.join(dict(zip('XYZ', letters))[c] for c in units)
                   for letters in itertools.product('ab',repeat=3)]
        self.assertAlmostEqual(prior.bits(result['recovered']), min(prior.bits(x) for x in options))
        with self.assertRaises(ValueError):
            infer_mapping('XX','ab')

    def test_hmm_forward_backward_against_exhaustive_paths(self):
        u = np.array([.6,.4]); b = np.array([[.7,.3],[.2,.8]])
        t = np.array([[[.8,.2],[.3,.7]],[[.6,.4],[.1,.9]]])
        e = np.array([[.8,.2],[.3,.7]]); observations = np.array([0,1,0,1])
        counts, posterior, ll = expectation(observations,e,u,b,t)
        truth = np.zeros((4,2)); total = 0.
        for path in itertools.product(range(2),repeat=4):
            p = u[path[0]]*b[path[0],path[1]]
            for i in range(2,4): p *= t[path[i-2],path[i-1],path[i]]
            for i in range(4): p *= e[path[i],observations[i]]
            total += p
            for i in range(4): truth[i,path[i]] += p
        np.testing.assert_allclose(posterior,truth/total,atol=1e-10)
        self.assertAlmostEqual(ll,math.log(total))
        np.testing.assert_allclose(counts.sum(axis=1),posterior.sum(axis=0))

    def test_historical_parser_excludes_notes_and_joins_drop_cap(self):
        html = '<div id="box_esterno"><p><span>P</span>resto Giovanni disse queste parole al nobile signore<br>Federigo.<sup>1</sup></p><div class="references"><p>Modern editorial words must never enter the historical training corpus.</p></div></div>'
        result = extract(html)
        self.assertEqual(result,['Presto Giovanni disse queste parole al nobile signore Federigo.'])


if __name__ == '__main__':
    unittest.main()
