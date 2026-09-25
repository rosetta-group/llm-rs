import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import language_id
from experiments import rejection_transfer as experiment
from voynich.description_length import CharacterPrior
from voynich.rejection import decide, stop_reason, transfer


def scores():
    return {
        'latin': dict(fit_excess=.3, transfer_excess=.35, coverage=.99, cap_hit=False),
        'english': dict(fit_excess=1.2, transfer_excess=1.3, coverage=.99, cap_hit=False),
        'italian': dict(fit_excess=1.4, transfer_excess=1.6, coverage=.99, cap_hit=False),
    }


class RejectionTests(unittest.TestCase):
    def test_missing_true_language_is_rejected_not_forced_to_runner_up(self):
        self.assertEqual(decide(scores())['accepted'], 'latin')
        result = decide(scores(), ('english', 'italian'))
        self.assertIsNone(result['accepted'])
        self.assertIn('fit_excess', result['reasons'])
        self.assertIn('transfer_excess', result['reasons'])

    def test_good_in_sample_score_does_not_override_failed_transfer(self):
        rows = scores()
        rows['latin']['transfer_excess'] = .51
        result = decide(rows)
        self.assertEqual(result['in_sample_accepted'], 'latin')
        self.assertIsNone(result['accepted'])
        self.assertIn('transfer_excess', result['reasons'])

    def test_margins_coverage_and_winner_must_all_pass(self):
        for change, reason in [({'coverage': .949}, 'coverage'),
                               ({'transfer_excess': 1.5}, 'winner_changed')]:
            rows = scores()
            rows['latin'].update(change)
            self.assertIn(reason, decide(rows)['reasons'])
        rows = scores()
        rows['english']['transfer_excess'] = .40
        self.assertIn('transfer_margin', decide(rows)['reasons'])
        # Invalid winner is not replaced by a convenient second-best candidate.
        rows['latin']['coverage'] = .1
        self.assertIsNone(decide(rows)['accepted'])

    def test_caps_are_inconclusive_including_wrong_candidate_cap(self):
        rows = scores()
        rows['italian']['cap_hit'] = True
        self.assertTrue(decide(rows)['inconclusive'])
        self.assertIsNone(decide(rows)['accepted'])

    def test_unscorable_and_one_candidate_do_not_pass(self):
        rows = scores()
        for row in rows.values():
            row['transfer_excess'] = None
        self.assertIn('transfer_unscorable', decide(rows)['reasons'])
        self.assertIsNone(decide(scores(), ('latin',))['accepted'])

    def test_threshold_boundaries(self):
        rows = scores()
        rows['latin'].update(fit_excess=.5, transfer_excess=.5, coverage=.95)
        rows['english'].update(fit_excess=.75, transfer_excess=.75)
        self.assertEqual(decide(rows)['accepted'], 'latin')


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.prior = CharacterPrior.fit(['abba' * 20], order=3, alphabet='ab')
        self.key = {'u:x': 'a', 'u:y': 'b'}

    def test_new_sequence_uses_exact_same_key(self):
        before = copy.deepcopy(self.key)
        result = transfer(['y', 'x', 'x', 'y'], self.key, self.prior)
        self.assertEqual(result['recovered'], 'baab')
        self.assertEqual(result['token_coverage'], 1.)
        self.assertEqual(self.key, before)

    def test_unknown_token_resets_context_and_counts_against_coverage(self):
        result = transfer(['x', 'y', 'unknown', 'y', 'x'], self.key, self.prior)
        self.assertEqual(result['recovered'], 'ab?ba')
        self.assertEqual(result['token_coverage'], .8)
        self.assertEqual(result['covered_runs'], 2)
        self.assertAlmostEqual(result['bits_per_letter'],
                               (self.prior.bits('ab') + self.prior.bits('ba')) / 4)
        self.assertNotIn('u:unknown', self.key)

    def test_no_covered_text_is_not_zero_cost(self):
        for tokens in ([], ['unknown']):
            result = transfer(tokens, self.key, self.prior)
            self.assertIsNone(result['bits_per_letter'])
            self.assertEqual(result['token_coverage'], 0.)

    def test_known_pieces_can_parse_unseen_combinations_without_learning(self):
        key = {'p:x': 'a', 's:y': 'b', 'p:z': 'b', 's:q': 'a'}
        before = dict(key)
        result = transfer(['xy', 'zq', 'xq'], key, self.prior)
        self.assertEqual(result['recovered'], 'abbaaa')
        self.assertEqual(before, key)


class IntegrityTests(unittest.TestCase):
    def test_atomic_seal_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            experiment.seal(path, {'x': 1})
            with self.assertRaises(FileExistsError):
                experiment.seal(path, {'x': 2})
            self.assertEqual(experiment.read(path), {'x': 1})
            self.assertEqual(list(Path(directory).glob('.pending-*')), [])

    def test_checkpoint_hash_mismatch_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit.json'
            experiment.seal(path, dict(provenance={'freeze': 'old'}, result={'key': 'a'}))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                experiment.checked_checkpoint(path, {'freeze': 'new'})

    def test_short_exact_and_long_partial_leaks_are_excluded(self):
        exact = {experiment.fingerprint('a short sentence')}
        shingles = experiment.grams('one two three four five six seven eight nine ten')
        self.assertTrue(experiment.overlaps('a short sentence', exact, shingles))
        self.assertTrue(experiment.overlaps('prefix one two three four five six seven eight suffix', exact, shingles))
        self.assertFalse(experiment.overlaps('different short sentence', exact, shingles))

    def test_sources_do_not_reuse_short_sentences_across_passages(self):
        rows = [dict(id='1', words=['a' * 5200]), dict(id='dup', words=['a' * 5200]),
                dict(id='2', words=['b' * 5200])]
        passages, excluded = experiment.pack_passages(rows, set(), set(), count=2)
        self.assertEqual([p['source_ids'] for p in passages], [['1'], ['2']])
        self.assertEqual(excluded['overlap'], 1)

    def test_copy_generator_is_reproducible_and_independent_of_plaintext(self):
        a = experiment.copy_text(['ab', 'bc', 'abc'], 300, 12)
        self.assertEqual(a, experiment.copy_text(['ab', 'bc', 'abc'], 300, 12))
        self.assertNotEqual(a, experiment.copy_text(['ab', 'bc', 'abc'], 300, 13))
        self.assertEqual(len(a), 300)
        self.assertTrue(all(a))
        self.assertTrue(set(''.join(a)) <= set('abc'))

    def test_futility_rule_counts_independent_positive_decisions(self):
        def row(kind, accepted=None, cap=False):
            return dict(kind=kind, language='latin', decision=dict(accepted=accepted, inconclusive=cap))
        self.assertIsNone(stop_reason([row('positive'), row('absent'), row('shuffle')]))
        self.assertEqual(stop_reason([row('positive'), row('positive')]), 'two_positives_rejected')
        self.assertEqual(stop_reason([row('copy', 'latin')]), 'negative_accepted')
        self.assertEqual(stop_reason([row('positive', 'english')]), 'wrong_language_accepted')
        self.assertEqual(stop_reason([row('copy', cap=True)]), 'inconclusive_compute_cap')

    def test_exporting_key_does_not_change_legacy_decoder_stages(self):
        prior = CharacterPrior.fit(['abba' * 20], order=3, alphabet='ab')
        s = dict(minimum=6, joint_restarts=4, joint_iterations=60, seed=3, prune_usage=3.,
                 repair_theta=5., repair_minimum=2, repair_passes=2, repair_usage_floor=1.,
                 refine_kicks=30, refine_kick_size=6, refine_cap=300)
        stage = dict(recovered='ab', segmentation=[('x',), ('y',)], cap_hit=False)
        refined = dict(recovered='ab', mapping={'u:x': 'a', 'u:y': 'b'}, cap_hit=False)
        calls = []
        for module, function in ((language_id, language_id.decode), (experiment, experiment.fit_key)):
            with patch.object(module, 'joint_em', return_value=stage) as em, \
                 patch.object(module, 'prune_and_rerun', return_value=stage) as prune, \
                 patch.object(module, 'repair', return_value=stage) as repair, \
                 patch.object(module, 'refine', return_value=refined) as refine:
                result = function(['x', 'y'], prior, s, 3600)
                self.assertEqual(result['recovered'], 'ab')
                calls.append([em.call_args, prune.call_args, repair.call_args, refine.call_args])
        self.assertEqual(calls[0], calls[1])


if __name__ == '__main__':
    unittest.main()
