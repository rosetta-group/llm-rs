import json
import os
import subprocess
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from experiments import cloud


class CloudTests(unittest.TestCase):
    def test_stop_command_rejects_successful_root_help(self):
        responses = [subprocess.CompletedProcess([], 0, "Usage:\n  runpodctl [command]\n", ""),
                     subprocess.CompletedProcess([], 0, "Usage:\n  runpodctl stop pod [podId] [flags]\n", "")]
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "example"}), \
                patch.object(cloud.subprocess, "run", side_effect=responses):
            self.assertEqual(cloud.stop_command(), ["runpodctl", "stop", "pod", "example"])

    def test_stop_command_fails_closed_without_matching_usage(self):
        response = subprocess.CompletedProcess([], 0, "Usage:\n  runpodctl [command]\n", "")
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "example"}), \
                patch.object(cloud.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "No supported"):
                cloud.stop_command()

    def test_budget_does_not_reset_or_exceed_cap(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cloud, "STATE", Path(temp)):
            start = time.time()-60
            budget = cloud.arm(start, 1.20)
            self.assertEqual(budget["deadline"], start+4*3600)
            self.assertLessEqual(budget["max_compute_usd"]+budget["reserved_usd"],10)
            with self.assertRaisesRegex(ValueError,"already armed"):
                cloud.arm(start,1.20)

    def test_invalid_quotes_and_future_start_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cloud,"STATE",Path(temp)):
            for quote in (0,-1,1.51,float("nan"),float("inf")):
                with self.assertRaises(ValueError):
                    cloud.arm(time.time(),quote)
            with self.assertRaises(ValueError):
                cloud.arm(time.time()+60,1)

    def test_models_share_exposure_settings(self):
        a=cloud.config_for("1.7B","gc",3000)
        b=cloud.config_for("8B","gc",3000)
        for key in ("context","stride","max_steps","seed","train_batch_size","gradient_accumulation_steps",
                    "learning_rate","eval_steps","save_steps","max_eval_pages"):
            self.assertEqual(a[key],b[key])
        self.assertEqual(a["lora_alpha"]/a["lora_rank"],b["lora_alpha"]/b["lora_rank"])
        self.assertIsNone(a["max_eval_pages"])

    def test_pair_estimate_accounts_for_evaluation(self):
        timing=dict(step_seconds=[10,10,.1,.1],evaluation_seconds=[2])
        estimate=cloud.estimate_pair([timing,timing],3000)
        self.assertGreater(estimate,2*3000*.1)
        self.assertGreater(estimate,cloud.estimate_pair([timing,timing],500))

    def test_bundle_has_no_test_text_or_local_secrets(self):
        cloud.bundle()
        with tarfile.open(cloud.ROOT/"artifacts/cloud-upload.tar.gz") as archive:
            names=archive.getnames()
            self.assertNotIn("config.py",names)
            self.assertFalse(any("secret" in n or n.startswith(".git/") for n in names))
            for name in ("gc","gc-shuffle"):
                docs=json.load(archive.extractfile(f"artifacts/data/{name}/documents.json"))
                self.assertEqual({d["split"] for d in docs},{"train","validation"})
            manifest=json.load(archive.extractfile("bundle-manifest.json"))
            for name,sha in manifest.items():
                self.assertEqual(cloud.hashlib.sha256(archive.extractfile(name).read()).hexdigest(),sha)


if __name__=="__main__":
    os.chdir(cloud.ROOT)
    unittest.main()
