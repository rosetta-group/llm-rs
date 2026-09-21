import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.characters import guard_disk, jobs, read, PLAN
from voynich.character import CharacterModel, score, split_documents, train, windows
from voynich.evaluate import target_signature


def document(page, text, split="validation"):
    return dict(page=page, folio=page, quire="A", currier="A", section="herbal",
                hand="1", text=text, split=split)


class CharacterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_windows_score_each_readable_character_once(self):
        text = "a?bc.d\n" * 5
        for context, stride in [(4, 2), (8, 1), (8, 8)]:
            seen = Counter()
            for w in windows(text, context, stride):
                for i, label in enumerate(w["labels"]):
                    if label != -100:
                        self.assertEqual(label, ord(text[w["start"] + i - 1]))
                        seen[w["start"] + i - 1] += 1
            self.assertEqual(seen, Counter({i: 1 for i, c in enumerate(text) if c != "?"}))

    def test_future_characters_cannot_change_past_predictions(self):
        for kind in ("transformer", "gru"):
            model = CharacterModel(kind, width=16, layers=2, context=8).eval()
            a = torch.tensor([[256, 97, 98, 99, 100]])
            b = torch.tensor([[256, 97, 98, 120, 121]])
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(a)[:, :3], model(b)[:, :3], atol=1e-6))

    def test_uniform_model_matches_analytical_loss_and_page_reset(self):
        for kind in ("transformer", "gru"):
            model = CharacterModel(kind, width=16, layers=1, context=8)
            with torch.no_grad():
                model.head.weight.zero_()
                model.head.bias.zero_()
            docs = [document("1", "abc?abc"), document("2", "ab.ab")]
            rows = score(model, docs, context=8, stride=4, batch_size=2)
            for doc, row in zip(docs, rows):
                self.assertEqual(row["units"], sum(c != "?" for c in doc["text"]))
                self.assertEqual(row["target_sha256"], target_signature(doc["text"]))
                self.assertAlmostEqual(row["nll"] / row["units"] / math.log(2), 8, places=5)
            self.assertEqual(rows[1], score(model, docs[1:], 8, 4, 2)[0])

    def test_test_documents_are_excluded(self):
        docs = [document("1", "ab", "train"), document("2", "cd"), document("3", "ef", "test")]
        a, b = split_documents(docs)
        self.assertEqual([d["page"] for d in a], ["1"])
        self.assertEqual([d["page"] for d in b], ["2"])

    def test_training_reloads_selected_weights_and_refuses_overwrite(self):
        for kind in ("transformer", "gru"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "documents.json").write_text(json.dumps([
                    document("1", "ab.ab." * 4, "train"), document("2", "ab.ab"),
                    document("3", "ab.ab"), document("4", "private test", "test")]))
                config = dict(kind=kind, width=16, layers=1, context=8, stride=4,
                              batch_size=4, learning_rate=.001, epochs=2, patience=4,
                              max_seconds=30, device="cpu", seed=42, dataset=str(root),
                              output=str(root / "output"))
                train(config)
                run = read(root / "output/run.json")
                self.assertTrue(run["checkpoint_verified"])
                self.assertEqual(run["training_characters_seen"], 48)
                self.assertNotIn("4", run["train_pages"] + run["validation_pages"])
                self.assertFalse(run["test_scored"])
                with self.assertRaises(FileExistsError):
                    train(config)

    def test_plan_has_no_downloads_and_reserves_disk(self):
        plan = read(PLAN)
        configs = jobs(plan)
        self.assertEqual(len(configs), 12)
        self.assertTrue(all(c["device"] == "mps" for c in configs))
        with patch("experiments.characters.shutil.disk_usage") as usage:
            usage.return_value.free = 19 * 2**30
            with self.assertRaisesRegex(RuntimeError, "reserved free disk"):
                guard_disk(plan)


if __name__ == "__main__":
    unittest.main()
