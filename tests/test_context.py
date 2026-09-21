import unittest

import torch

from experiments.context import check_targets
from tests.test_character import document
from voynich.character import BOS, CharacterModel, score
from voynich.context import histories, score_context


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_histories_are_exact_and_keep_unreadable_context(self):
        self.assertEqual(list(histories("a?bcd", 2)), [([BOS], 97),
                         ([97, 63], 98), ([63, 98], 99), ([98, 99], 100)])
        for context in (1, 2, 8, 16, 32, 64, 128):
            text = "a?b.c\n" * 25
            rows = list(histories(text, context))
            self.assertEqual(len(rows), sum(c != "?" for c in text))
            for (history, target), i in zip(rows, (i for i, c in enumerate(text) if c != "?")):
                self.assertEqual(history, ([BOS] + list(map(ord, text[:i])))[-context:])
                self.assertEqual(target, ord(text[i]))

    def test_optimized_loss_matches_causal_stride_one_reference(self):
        torch.manual_seed(7)
        model = CharacterModel("gru", width=16, layers=2, context=16)
        docs = [document("1", "a?b.c\n" * 5), document("2", "xy?z.x")]
        for context in (1, 2, 8, 16):
            expected = score(model, docs, context=context, stride=1, batch_size=3)
            actual = score_context(model, docs, context, batch_size=7)
            check_targets(expected, actual)
            for a, b in zip(expected, actual):
                self.assertAlmostEqual(a["nll"], b["nll"], places=4)
                self.assertEqual(a["correct"], b["correct"])
            self.assertTrue(model.training)
            single = score_context(model, docs[1:], context, batch_size=1)[0]
            self.assertAlmostEqual(single["nll"], actual[1]["nll"], places=4)

    def test_refuses_test_pages_and_mismatched_targets(self):
        model = CharacterModel("gru", width=16, layers=1)
        with self.assertRaisesRegex(ValueError, "validation"):
            score_context(model, [document("1", "sealed", split="test")], 8)
        a = score_context(model, [document("1", "abc")], 8)
        b = score_context(model, [document("1", "abd")], 8)
        with self.assertRaisesRegex(ValueError, "identical targets"):
            check_targets(a, b)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_targets(a, a + a)


if __name__ == "__main__":
    unittest.main()
