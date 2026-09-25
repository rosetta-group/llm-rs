"""Audit, freeze, then evaluate the bounded Etruscan repair-and-abstain follow-up.

python -m experiments.etruscan_repair audit
python -m experiments.etruscan_repair freeze
python -m experiments.etruscan_repair run
"""

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from etruscan import classes, context, corpus, repair

OUT = Path("experiments/etruscan-repair")
SALT = "etruscan-repair-2026-09-24"
NULLS = 99
BOOTSTRAPS = 1000


def write(path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dataset():
    records, text_audit = repair.clean_texts()
    texts = [r[2] for r in records]
    vocabulary = {t for text in texts for t in text}
    labels, glosses, label_audit = repair.repaired_labels()
    labels = {t: c for t, c in labels.items() if t in vocabulary}
    old = json.loads(Path("experiments/etruscan-fresh/results.json").read_text())
    development = set(old["primary_etp_plus_ciep"]["predictions"])
    suffixes = sorted({corpus.normalise(s.strip().strip("-"))
                       for s in (corpus.LARTH / "ETPSuff.txt").read_text().splitlines() if s.strip()})
    suffixes = [s for s in suffixes if s and re_letters(s)]
    # Include excluded development words in the grouping universe, even if absent
    # from the clean texts. Their related forms must not return through a back door.
    groups = repair.family_groups(vocabulary | development | set(glosses), glosses, suffixes)
    reserved = {groups[t] for t in development}
    excluded = {t: c for t, c in labels.items() if groups[t] in reserved}
    labels = {t: c for t, c in labels.items() if groups[t] not in reserved}
    anchors = {t: "NUM" for t in vocabulary if t.startswith("N:")}
    return texts, labels, groups, anchors, {"text_records": records, "text_audit": text_audit,
                                          "label_audit": label_audit, "development_words": sorted(development),
                                          "excluded_development_relatives": excluded}


def re_letters(value):
    return value.isascii() and value.isalpha()


def audit():
    texts, labels, groups, anchors, detail = dataset()
    old_words = corpus.glossed_words()
    repaired, _, _ = repair.repaired_labels()
    vocabulary = {t for text in texts for t in text}
    changes = {}
    for t in sorted(vocabulary):
        old = classes.etruscan_label(t, old_words)
        new = repaired.get(t) if not t.startswith("N:") else "NUM"
        if old != new:
            changes[t] = {"old": old, "new": new,
                          "tokens": sum(text.count(t) for text in texts),
                          "source_rows": detail["label_audit"]["accepted_rows"].get(t, [])}
    etp_sample = sorted(detail["text_records"], key=lambda r: repair.digest(SALT + r[1] + " ".join(r[2])))[:20]
    all_rows = corpus.load_rows().drop_duplicates(subset=["ID", "Etruscan", "key"])
    ciep = [r for r in corpus.texts(all_rows, numerals=True) if r[0] == "CIEP"]
    ciep_sample = sorted(ciep, key=lambda r: repair.digest(SALT + r[1]))[:10]
    label_sample = sorted(changes, key=lambda t: repair.digest(SALT + t))[:20]
    members = collections.defaultdict(list)
    for t in labels:
        members[groups[t]].append(t)
    result = {"texts": len(texts), "tokens": sum(map(len, texts)), "types": len(vocabulary),
              "labelled_types": len(labels), "classes": dict(collections.Counter(labels.values())),
              "families": len(members), "roman_anchors": anchors,
              "largest_families": sorted([sorted(v) for v in members.values()], key=lambda v: (-len(v), v))[:15],
              "changes": changes, "etp_sample": etp_sample, "ciep_sample": ciep_sample,
              "label_sample": {t: changes[t] for t in label_sample}, **detail}
    write(OUT / "audit.json", result)
    print(json.dumps({k: result[k] for k in ("texts", "tokens", "types", "labelled_types", "classes", "families", "largest_families", "etp_sample", "ciep_sample", "label_sample")}, indent=2))


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    if (OUT / "results.json").exists():
        raise RuntimeError("Cannot freeze after results exist")
    files = ["etruscan/repair.py", "etruscan/corpus.py", "etruscan/classes.py", "etruscan/context.py",
             "experiments/etruscan_repair.py", "tests/test_etruscan_repair.py",
             "experiments/etruscan-repair/PROTOCOL.md", "experiments/etruscan-repair/AUDIT.md",
             "experiments/etruscan-repair/audit.json", "experiments/etruscan-fresh/results.json"]
    sources = [corpus.LARTH / f for f in ("Etruscan.csv", "ETP_POS.csv", "ETPWords.txt", "ETPNames.txt", "ETPSuff.txt")]
    write(OUT / "freeze.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                               "files": {p: hash_file(Path(p)) for p in files},
                               "sources": {str(p): hash_file(p) for p in sources},
                               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}})


def verify_freeze():
    frozen = json.loads((OUT / "freeze.json").read_text())
    for section in ("files", "sources"):
        for path, expected in frozen[section].items():
            if hash_file(Path(path)) != expected:
                raise RuntimeError(f"Frozen input changed: {path}")
    return frozen


def record_predictions(predictions, gold, groups, train, fold, threshold=None):
    majority = collections.Counter(train.values()).most_common(1)[0][0]
    return [dict(predictions[t], word=t, gold=gold[t], family=groups[t], fold=fold,
                 majority=majority, accepted=bool(threshold is not None and predictions[t]["evidence"]
                                                and predictions[t]["score"] >= threshold)) for t in sorted(gold)]


def evaluate(texts, labels, groups, anchors):
    buckets = repair.folds(labels, groups, 5, SALT)
    results = {m: [] for m in repair.METHODS}
    calibration = []
    original_m2 = []
    for f, held in enumerate(buckets):
        train = {t: labels[t] for t in sorted(labels) if t not in held}
        gold = {t: labels[t] for t in sorted(held)}
        inner_records = {m: [] for m in repair.METHODS}
        for j, inner_held in enumerate(repair.folds(train, groups, 3, SALT + str(f))):
            inner_train = {t: c for t, c in train.items() if t not in inner_held}
            inner_gold = {t: train[t] for t in sorted(inner_held)}
            for method in repair.METHODS:
                pred = repair.predict(texts, {**anchors, **inner_train}, groups, method == "context_endings")
                inner_records[method] += record_predictions(pred, inner_gold, groups, inner_train, j)
        for method in repair.METHODS:
            threshold = repair.choose_threshold(inner_records[method])
            pred = repair.predict(texts, {**anchors, **train}, groups, method == "context_endings")
            results[method] += record_predictions(pred, gold, groups, train, f, threshold)
            calibration.append({"fold": f, "method": method, "threshold": threshold,
                                "records": inner_records[method]})
        # Descriptive bridge to the old method on the exact new splits and labels.
        old_pred = context.neighbour_classes(texts, {**anchors, **train})
        original_m2 += [dict(word=t, gold=gold[t], predicted=old_pred[t], family=groups[t], fold=f,
                            majority=collections.Counter(train.values()).most_common(1)[0][0], accepted=True)
                        for t in sorted(gold)]
        print(f"Outer fold {f+1}/5 complete", flush=True)
    return results, calibration, original_m2, buckets


def family_intervals(records, comparison):
    """Paired family bootstrap, never treating repeated forms as independent."""
    by_family = collections.defaultdict(list)
    for i, row in enumerate(records):
        by_family[row["family"]].append(i)
    keys = sorted(by_family)
    rng = np.random.default_rng(240924)
    precisions, differences = [], []
    for _ in range(BOOTSTRAPS):
        indexes = [i for g in rng.choice(keys, size=len(keys), replace=True) for i in by_family[g]]
        a = repair.metrics([records[i] for i in indexes])
        b = repair.metrics([comparison[i] for i in indexes])
        # Zero-call samples count as zero for the conservative precision bound.
        precisions.append(a["accepted_precision"] or 0.0)
        differences.append(a["full_balanced_accuracy"] - b["full_balanced_accuracy"])
    return {"accepted_precision_family_interval": list(map(float, np.quantile(precisions, [0.025, 0.975]))),
            "balanced_accuracy_lift_family_interval": list(map(float, np.quantile(differences, [0.025, 0.975])))}


def null_scores(texts, labels, groups, anchors, buckets):
    """Full-coverage seed-label null; does not claim selective-policy p-values."""
    result = {m: [] for m in repair.METHODS}
    for rep in range(NULLS):
        records = {m: [] for m in repair.METHODS}
        for f, held in enumerate(buckets):
            train = {t: labels[t] for t in sorted(labels) if t not in held}
            # Shuffle the fixed numeral anchors too, so this is a label-free null.
            all_seeds = {**anchors, **train}
            all_keys = sorted(all_seeds)
            all_values = list(all_seeds[t] for t in all_keys)
            np.random.default_rng(90000 + rep * 5 + f).shuffle(all_values)
            all_seeds = dict(zip(all_keys, all_values))
            gold = {t: labels[t] for t in sorted(held)}
            for method in repair.METHODS:
                pred = repair.predict(texts, all_seeds, groups, method == "context_endings")
                records[method] += record_predictions(pred, gold, groups, train, f)
        for method in repair.METHODS:
            result[method].append(repair.metrics(records[method])["full_balanced_accuracy"])
        if (rep + 1) % 20 == 0:
            print(f"Label null {rep+1}/{NULLS} complete", flush=True)
    return result


def run():
    if (OUT / "results.json").exists():
        raise FileExistsError("Results already exist")
    frozen = verify_freeze()
    texts, labels, groups, anchors, _ = dataset()
    result, calibration, m2, buckets = evaluate(texts, labels, groups, anchors)
    nulls = null_scores(texts, labels, groups, anchors, buckets)
    summaries = {}
    for method in repair.METHODS:
        rows = sorted(result[method], key=lambda r: r["word"])
        baseline = sorted(result["context"], key=lambda r: r["word"])
        summary = repair.metrics(rows)
        summary.update(family_intervals(rows, baseline))
        summary["p_seed_label_null_full_classifier"] = (1 + sum(v >= summary["full_balanced_accuracy"] for v in nulls[method])) / (NULLS + 1)
        precision = summary["accepted_precision"] or 0.0
        checks = {"precision_ge_80pct": precision >= 0.8,
                  "coverage_ge_20pct": summary["coverage"] >= 0.2,
                  "accepted_families_ge_20": summary["accepted_families"] >= 20,
                  "family_precision_lower_ge_70pct": summary["accepted_precision_family_interval"][0] >= 0.7,
                  "beats_majority_on_same_calls_by_5pp": precision >= (summary["majority_on_accepted"] or 0.0) + 0.05,
                  "nonname_calls_ge_10": summary["nonname_calls"] >= 10,
                  "nonname_families_ge_3": summary["nonname_families"] >= 3,
                  "nonname_precision_ge_70pct": (summary["nonname_precision"] or 0.0) >= 0.7,
                  "seed_label_null_p_le_05": summary["p_seed_label_null_full_classifier"] <= 0.05}
        summary["gate_checks"] = checks
        summary["passed"] = all(checks.values())
        summaries[method] = summary
    write(OUT / "results.json", {"freeze_sha256": hash_file(OUT / "freeze.json"), "runtime": frozen["runtime"],
                                "summaries": summaries, "records": result, "calibration": calibration,
                                "nulls": nulls, "repaired_M2_descriptive": repair.metrics(m2),
                                "repaired_M2_records": m2,
                                "splits": [sorted(bucket) for bucket in buckets]})
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "freeze", "run"))
    args = parser.parse_args()
    {"audit": audit, "freeze": freeze, "run": run}[args.action]()


if __name__ == "__main__":
    main()
