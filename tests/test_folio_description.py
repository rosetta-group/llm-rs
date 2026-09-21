import csv
import hashlib
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from experiments.describe_folios import DATA, crop_image, fingerprint, panel_records
from voynich.folio_description import describe


class FolioDescriptionTests(unittest.TestCase):
    def paper(self):
        return np.full((400, 300, 3), [218, 203, 170], dtype=np.uint8)

    def test_blank_has_no_objects_or_domain_claim(self):
        result = describe(self.paper())
        self.assertEqual(result["circle_candidate_count"], 0)
        self.assertEqual(result["cool_component_count"], 0)
        self.assertEqual(result["domain"], "unassigned")
        self.assertEqual(result["layout"], "sparse_or_unresolved")
        self.assertIsNone(result["coarse_colour_symmetry"])

    def test_colours_and_spatial_direction(self):
        for colour, field in [([60, 120, 60], "green_fraction"), ([35, 65, 150], "blue_fraction"),
                              ([160, 50, 35], "warm_fraction")]:
            image = self.paper()
            image[50:140, 40:120] = colour
            result = describe(image)
            self.assertGreater(result[field], .05)
            if field != "warm_fraction":
                self.assertIn("upper left", result["description"])

    def test_ring_and_horizontal_lines(self):
        ring = self.paper()
        cv2.circle(ring, (150, 200), 90, (45, 40, 35), 3)
        self.assertGreaterEqual(describe(ring)["circle_candidate_count"], 1)
        lines = self.paper()
        for y in range(45, 355, 12):
            cv2.line(lines, (35, y), (265, y), (45, 40, 35), 2)
        result = describe(lines)
        self.assertEqual(result["circle_candidate_count"], 0)
        self.assertEqual(result["layout"], "horizontal_marks_dominant")

    def test_crop_coordinates_and_small_image_failure(self):
        image = Image.new("RGB", (1000, 800))
        self.assertEqual(crop_image(image, [.1, .25, .6, .75]).size, (500, 400))
        with self.assertRaises(ValueError):
            describe(np.zeros((10, 10, 3), np.uint8))

    def test_complete_catalogue_and_distinct_f101v_panels(self):
        rows = list(panel_records())
        self.assertEqual(len(rows), 228)
        self.assertEqual(len({p["folio_id"] for p, _, _ in rows}), 228)
        f101 = {p["folio_id"]: image["image_id"] for p, image, _ in rows if p["folio_id"].startswith("f101v")}
        self.assertEqual(f101, {"f101v1": "1006251", "f101v2": "1006250"})

    def test_analysis_id_tracks_heuristic_changes(self):
        first, _ = fingerprint()
        with patch.dict("experiments.describe_folios.CONFIG", {"long_edge": 401}):
            second, _ = fingerprint()
        self.assertNotEqual(first, second)

    def test_development_text_page_is_not_a_circle(self):
        image = np.asarray(Image.open(DATA / "images/yale_1006254.jpg").convert("RGB"))
        result = describe(image)
        self.assertEqual(result["circle_candidate_count"], 0)
        self.assertEqual(result["layout"], "horizontal_marks_dominant")

    def test_corrupt_source_fails_before_analysis(self):
        with patch("experiments.describe_folios.sha", return_value="wrong-image-hash"):
            with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
                list(panel_records())

    def test_archived_tables_and_output_hashes(self):
        archives = list((DATA / "analyses").glob("*/analysis.json"))
        self.assertTrue(archives, "Run the description pipeline before its archive integration check")
        for registry in archives:
            metadata = json.loads(registry.read_text())
            folder = registry.parent
            for name, expected in metadata["outputs"].items():
                self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected)
            rows = [json.loads(line) for line in (folder / "descriptions.jsonl").read_text().splitlines()]
            with (folder / "descriptions.csv").open(newline="") as source:
                table = list(csv.DictReader(source))
            self.assertEqual(len(rows), metadata["row_count"])
            self.assertEqual(len(rows), len(table))
            self.assertEqual(len({(r["analysis_id"], r["folio_id"]) for r in rows}), len(rows))
            for row, record in zip(rows, table):
                self.assertEqual(row["analysis_id"], metadata["analysis_id"])
                for key, value in row.items():
                    if value is None:
                        self.assertEqual(record[key], "")
                    elif isinstance(value, str):
                        self.assertEqual(record[key], value)
                    else:
                        self.assertEqual(json.loads(record[key]), value)


if __name__ == "__main__":
    unittest.main()
