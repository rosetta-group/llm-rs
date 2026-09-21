import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_character import document
from voynich.character import CharacterModel
from voynich.context import score_context
from voynich.matched import last_logits, pack_training, train
from experiments.matched import interaction, jobs, read, training_seed, PLAN


class MatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equal_targets_at_every_context_and_train_only(self):
        docs = [document("1", "ab?cd.\nab", "train"), document("2", "xyz", "train")]
        previous = None
        for context in (1, 8, 32, 128):
            ids, lengths, targets = pack_training(docs, context, "cpu")
            self.assertEqual(len(targets), 11)
            if previous is not None:
                self.assertTrue(torch.equal(previous, targets))
            self.assertEqual(ids[8, 0].item(), 256)  # Fresh page, not preceding page history.
            self.assertEqual(lengths[8].item(), 1)
            previous = targets
        with self.assertRaisesRegex(ValueError, "training pages"):
            pack_training([document("3", "sealed", "test")], 8, "cpu")

    def test_training_logits_match_exact_evaluator(self):
        model = CharacterModel("gru", width=16, layers=2).eval()
        docs = [document("1", "a?bc.d\nab", "train")]
        ids, lengths, targets = pack_training(docs, 8, "cpu")
        logits = last_logits(model, ids, lengths)
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        expected = score_context(model, [dict(docs[0], split="validation")], 8)[0]
        self.assertAlmostEqual(loss, expected["nll"], places=4)
        loss_with_grad = torch.nn.functional.cross_entropy(logits, targets)
        loss_with_grad.backward()
        self.assertGreater(model.recurrent.weight_hh_l0.grad.abs().sum().item(), 0)

    def test_fixed_final_exposure_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = [document("1", "ab.ab.ab.", "train"), document("2", "ab.ab."),
                    document("3", "private", "test")]
            (root / "documents.json").write_text(json.dumps(docs))
            initial = []
            for context in (2, 8):
                config = dict(output=str(root / str(context)), dataset=str(root), context=context,
                              width=16, layers=1, batch_size=4, eval_batch_size=4, epochs=2,
                              learning_rate=.001, seed=42, device="cpu")
                run = train(config)
                initial.append(run["initial_sha256"])
                self.assertEqual(run["training_characters_seen"], 18)
                self.assertEqual(run["epoch"], 2)
                self.assertEqual(run["primary"], "final_epoch")
                self.assertTrue(run["checkpoint_verified"])
                self.assertFalse(run["test_scored"])
                self.assertNotIn("3", run["train_pages"] + run["validation_pages"])
                self.assertEqual(json.loads((root / str(context) / "last.json").read_text())["epoch"], 2)
                with self.assertRaises(FileExistsError):
                    train(config)
            self.assertEqual(initial[0], initial[1])

    def test_interaction_compares_gains_without_equating_strings(self):
        def rows(bits, signature):
            return [dict(page=str(i), folio=str(i), units=10, nll=10 * bits * math.log(2),
                         target_sha256=signature) for i in range(2)]
        a, b, c, d = rows(4, "intact"), rows(3, "intact"), rows(5, "shuffled"), rows(4.5, "shuffled")
        result = interaction(a, b, c, d)
        self.assertAlmostEqual(result["extra_intact_gain_bpc"], .5)
        for v in result["interval_95"]:
            self.assertAlmostEqual(v, .5)
        d[0]["target_sha256"] = "different targets"
        with self.assertRaisesRegex(ValueError, "identical targets"):
            interaction(a, b, c, d)

    def test_plan_matches_exposure_and_model_across_18_runs(self):
        configs = jobs(read(PLAN))
        self.assertEqual(len(configs), 18)
        self.assertEqual({c["context"] for c in configs}, {8, 32, 128})
        self.assertEqual({c["epochs"] for c in configs}, {20})
        self.assertEqual({(c["width"], c["layers"], c["batch_size"]) for c in configs}, {(256, 3, 256)})

    def test_report_distinguishes_training_seed_from_bootstrap_seed(self):
        bootstrap = dict(seed=42, groups=15, delta_bits_per_character=.1)
        result = training_seed(bootstrap, 43)
        self.assertEqual(result["seed"], 43)
        self.assertEqual(result["bootstrap_seed"], 42)
        self.assertEqual(bootstrap["seed"], 42)


if __name__ == "__main__":
    unittest.main()
