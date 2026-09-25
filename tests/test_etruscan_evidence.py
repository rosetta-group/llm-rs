import copy
import json
from pathlib import Path
import tempfile
import unittest

from etruscan.evidence import sha, validate


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.bundle = json.loads(Path("experiments/etruscan-evidence/annotations.json").read_text())

    def test_portable_bundle_retains_uncertainty(self):
        result = validate(self.bundle, verify_files=False)
        self.assertEqual(result["source_supported_relations"], 6)
        self.assertEqual(result["record_statuses"]["disputed"], 3)
        self.assertEqual(result["fresh_evaluation_records"], 0)

    def test_cannot_promote_exposed_cases_or_claim_expert_review(self):
        for field, value in (("fresh_evaluation_eligible", True), ("expert_review", "approved")):
            with self.subTest(field=field):
                bundle = copy.deepcopy(self.bundle)
                bundle["records"][0][field] = value
                with self.assertRaisesRegex(ValueError, "expert review or fresh evaluation"):
                    validate(bundle, verify_files=False)

    def test_disputed_alternatives_cannot_become_corroborated_graph(self):
        record = next(r for r in self.bundle["records"] if r["id"] == "Cr 3.18")
        self.assertEqual(len(record["relations"]), 2)
        record["status"] = "corroborated"
        with self.assertRaisesRegex(ValueError, "unsupported relations"):
            validate(self.bundle, verify_files=False)

    def test_evidence_requires_registered_source_and_pinpoint(self):
        relation = self.bundle["records"][0]["relations"][0]
        for refs in ([], [{"source": "invented", "locator": "p. 1"}],
                     [{"source": "schulze1993", "locator": ""}]):
            with self.subTest(refs=refs):
                relation["evidence"] = refs
                with self.assertRaises(ValueError):
                    validate(self.bundle, verify_files=False)

    def test_span_cannot_point_past_excerpt(self):
        self.bundle["records"][0]["spans"][0]["tokens"] = [999]
        with self.assertRaisesRegex(ValueError, "Invalid token span"):
            validate(self.bundle, verify_files=False)

    def test_participant_must_exist(self):
        self.bundle["records"][0]["relations"][0]["frame"][1] = "imaginary"
        with self.assertRaisesRegex(ValueError, "Unknown participant"):
            validate(self.bundle, verify_files=False)

    def test_frozen_transcription_cannot_be_silently_replaced(self):
        # Remove optional research PDFs; retain checks against the real frozen manifest.
        for source in self.bundle["sources"]:
            source["local_path"] = None
        self.bundle["records"][0]["tokens"][0] = "changed"
        with self.assertRaisesRegex(ValueError, "silently changed"):
            validate(self.bundle)

    def test_changed_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.pdf"
            path.write_bytes(b"original")
            self.bundle["sources"][0].update(local_path=str(path), sha256=sha(path))
            path.write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "Source bytes changed"):
                validate(self.bundle)


if __name__ == "__main__":
    unittest.main()
