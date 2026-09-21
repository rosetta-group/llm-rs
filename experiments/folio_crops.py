"""Propose foldout panel crops by matching catalogue thumbnails to Yale scans.

Produces proposals only. Inspect them before replacing the reviewed crops.json.
"""
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from experiments.download_folios import DATA, ROOT, fetch, digest, write_json


def propose():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    panels = json.loads((DATA / "sources/catalogue-index.json").read_text())["panels"]
    counts = Counter(oid for p in panels for oid in p["yale_image_ids"])
    sift = cv2.SIFT_create(contrastThreshold=.02)
    proposals = []
    for panel in panels:
        if not any(counts[oid] > 1 for oid in panel["yale_image_ids"]):
            continue
        thumb = DATA / "sources/panel-thumbnails" / (panel["folio_id"] + ".jpg")
        thumb.parent.mkdir(parents=True, exist_ok=True)
        if not thumb.exists():
            thumb.write_bytes(fetch(panel["thumbnail_url"]))
        query = cv2.imread(str(thumb), cv2.IMREAD_GRAYSCALE)
        query = cv2.resize(query, None, fx=2, fy=2)
        keys, desc = sift.detectAndCompute(query, None)
        best = None
        for oid in panel["yale_image_ids"]:
            image = cv2.imread(str(DATA / "images" / f"yale_{oid}.jpg"), cv2.IMREAD_GRAYSCALE)
            image_keys, image_desc = sift.detectAndCompute(image, None)
            matches = cv2.BFMatcher().knnMatch(desc, image_desc, k=2)
            good = [a for pair in matches if len(pair) == 2 for a, b in [pair] if a.distance < .72 * b.distance]
            if len(good) < 12:
                continue
            src = np.float32([keys[m.queryIdx].pt for m in good])
            dst = np.float32([image_keys[m.trainIdx].pt for m in good])
            transform, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5)
            if transform is None or mask.sum() < 12:
                continue
            h, w = query.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            points = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
            box = [max(0, float(points[:, 0].min() / image.shape[1])),
                   max(0, float(points[:, 1].min() / image.shape[0])),
                   min(1, float(points[:, 0].max() / image.shape[1])),
                   min(1, float(points[:, 1].max() / image.shape[0]))]
            if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                continue
            rec = {"folio_id": panel["folio_id"], "image_id": oid,
                   "bbox": [round(v, 4) for v in box], "inliers": int(mask.sum()), "matches": len(good),
                   "thumbnail_path": str(thumb.relative_to(ROOT)),
                   "thumbnail_sha256": digest(thumb.read_bytes()), "thumbnail_url": panel["thumbnail_url"]}
            if best is None or rec["inliers"] > best["inliers"]:
                best = rec
        if best is None:
            raise ValueError(f"Registration failed; inspect {panel['folio_id']}")
        proposals.append(best)
    return proposals


if __name__ == "__main__":
    path = ROOT / "tmp/folio-crop-proposals.json"
    write_json(path, propose())
    print(f"Review proposals at {path}; the frozen crop file was not changed.")
