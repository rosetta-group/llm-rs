"""Thirty-record names-as-scaffolding pilot: build, freeze, run.

The manifest is a small manually audited selection, not a random corpus sample.
Public features and training frames are separated before the predictor runs.
"""

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd

from etruscan import corpus, scaffolding as model

OUT = Path("experiments/etruscan-scaffolding")
NULLS = 999

# Frames are transcriptions of the supplied English meanings. Entity indices are
# source-token spans; they are NOT semantic roles. Deity/name spans are favourable
# manually supplied anchors, an explicit limitation of this pilot.
def child(a="e0", b="e1"):
    return ["CHILD_OF", a, b]


def spouse(a="e0", b="e1"):
    return ["SPOUSE_OF", a, b]


def gift(a="e0", b="UNSPECIFIED"):
    return ["TRANSFER", a, "OBJECT", b]


def own():
    return ["OWNED_BY", "OBJECT", "e0"]


def made():
    return ["MADE", "e0", "OBJECT"]


# id, entity spans, deity entity indices, gold graph, target family, target edge,
# editorial note. Selection and annotations are fixed before model scores exist.
SPECS = [
    ("ETP 192", [[0,1],[2]], [], [child()], None, [], "Name-first order is Cleusinas Laris; father Laris is a distinct entity."),
    ("Cr 1.10", [[0,1],[2]], [], [child()], None, [], "Explicit son formula; later empty dictionary rows are not used."),
    ("Cl 1.1134", [[0,1],[2]], [], [child()], None, [], "Son is supplied in parentheses in the edition, with no explicit kin word."),
    ("Vc 1.84", [[0,1],[2]], [], [child()], None, [], "Surname precedes given name; son is supplied by the editor."),
    ("Ta 1.62", [[0,1],[2]], [], [child()], None, [], "Explicit son formula."),
    ("Cl 1.1006", [[0,1],[2]], [], [child()], None, [], "Abbreviated father's name ath; son is editorially supplied."),
    ("Ta 1.13", [[0,1],[2,3],[5,6]], [], [child(),spouse("e0","e2")], "daughter", [child()], "Two distinct relations: daughter of Larce Spantus; wife of Arnth Partunus."),
    ("Ta 1.59", [[0,1],[2],[4]], [], [child(),child("e1","e2")], "daughter", [child()], "Nested pedigree; the second son relation is editorially supplied."),
    ("Cl 1.1885", [[0,1,2],[3],[4]], [], [spouse(),child("e0","e2")], "daughter", [child("e0","e2")], "Wife relation precedes daughter relation; larthia has conflicting gender/case entries, retained as unknown."),
    ("Cl 1.373", [[0,1],[2]], [], [spouse()], None, [], "Wife is editorially supplied; negative contrast to the implicit son formula."),
    ("ETP 230", [[0,1,2],[3]], [], [spouse()], None, [], "Wife is editorially supplied; first name abbreviated tha."),
    ("At 1.111", [[0,1],[3,4]], [], [spouse()], None, [], "Source marks aleth{na}s: restoration is retained only inside a supplied name anchor; no restored relation word."),
    ("Cr 2.20", [[1]], [], [own()], None, [], "Container belongs to Karkana; object type is not a model feature."),
    ("Cr 2.18", [[2]], [], [own()], None, [], "Wine pitcher belongs to Karkana."),
    ("Cr 2.2", [[2]], [], [own()], None, [], "Plate belongs to Larice."),
    ("Vs 1.86", [[1,2]], [], [own()], None, [], "Tomb of Larice Telathuras; OWNED_BY means of/associated with, not a legal ownership claim."),
    ("Sp 2.36", [[2,3]], [], [own()], None, [], "Libation vessel of Tata Tulalu."),
    ("ETP 269", [[2,3],[4]], [], [gift("e0","e1")], "mulu", [gift("e0","e1")], "Tetana Velkasnas gives to Velellia; both endpoints specified."),
    ("Vt 3.1", [[2,3]], [], [gift()], "mulu", [gift()], "Surname-before-given-name donor; no recipient stated."),
    ("Cr 3.11", [[2,3]], [], [gift()], "mulu", [gift()], "Donor Mamarce Velkhanas; no recipient stated."),
    ("Cr 3.9", [[1,2]], [], [gift()], "mulu", [gift()], "Donor Spurie Utas; no recipient stated."),
    ("Cr 3.10", [[2,3]], [], [gift("UNSPECIFIED","e0")], "mulu", [gift("UNSPECIFIED","e0")], "Passive given to Laris Velkhainas; must not reverse recipient into donor."),
    ("Cr 3.7", [[1,2]], [], [gift("UNSPECIFIED","e0")], None, [], "Independent aliqu wording: donated to Spurie Teithurnas."),
    ("Co 3.7", [[0,1],[3]], [1], [gift("e0","e1")], None, [], "Larthia Ateinei dedicates to Matrns; larthia dictionary ambiguity preserved."),
    ("ETP 339", [[0,1],[3]], [1], [gift("e0","e1")], None, [], "Arnth Saupunias dedicates greaves to Minerva; object type not supplied to model."),
    ("ETP 238", [[2,3],[4,5]], [1], [gift("e0","e1")], None, [], "Uni Huinthnaia is one deity with an epithet."),
    ("Ve 3.30", [[1,3]], [], [gift()], None, [], "Donor name is discontinuous around the dedication verb; one entity, not two."),
    ("ETP 189", [[2,3],[4],[6]], [1,2], [gift("e0","e1"),child("e2","e1")], None, [], "Espi is mother of Catha, so child edge runs Catha to Espi."),
    ("ETP 304", [[1]], [], [made()], None, [], "Tite made this; contrasts making with transfer."),
    ("Ve 6.2", [[1]], [], [made()], None, [], "Source mi<ni> reconstructs the object pronoun; favourable supplied reading, separately flagged."),
]


def write(path, content):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    rows = corpus.load_rows()
    records = []
    for tid, spans, deities, frames, family, targets, note in SPECS:
        candidates = rows[(rows.ID == tid) & rows.Translation.fillna("").str.strip().ne("")]
        row = candidates.iloc[0]  # Explicitly pin the first translated source row.
        words = [w.replace("kh", "ch") for w in corpus.tokens(row.Etruscan, numerals=True)]
        records.append({"id": tid, "source_index": int(row.name), "raw": row.Etruscan,
                        "translation": row.Translation, "tokens": words,
                        "entities": [{"tokens": span, "kind": "DEITY" if i in deities else "PERSON"}
                                     for i, span in enumerate(spans)],
                        "gold": frames, "targets": targets, "target_family": family,
                        "probe_tokens": [i for i,w in enumerate(words) if
                                         (family=="daughter" and w.startswith("sec")) or
                                         (family=="mulu" and w.startswith("mul"))],
                        "note": note, "source_reading_has_restoration": any(c in row.Etruscan for c in "[]{}<>")})
    write(OUT / "manifest.json", records)
    morphology = model.name_morphology()
    write(OUT / "public.json", [model.public_view(r, morphology) for r in records])
    print(f"Built {len(records)} audited records, {sum(bool(r['targets']) for r in records)} hidden-family tests.")


def load():
    manifest = json.loads((OUT / "manifest.json").read_text())
    public = json.loads((OUT / "public.json").read_text())
    views = {r["id"]: r for r in public}
    train, test = [], []
    for r in manifest:
        if r["targets"]:
            test.append(r)
        else:
            train.append({"public": views[r["id"]], "frames": r["gold"]})
    return manifest, views, train, test


def audit_barrier():
    manifest, views, train, test = load()
    train_ids = {r["public"]["id"] for r in train}
    test_ids = {r["id"] for r in test}
    assert len(manifest) == 30 and len(views) == 30
    assert train_ids.isdisjoint(test_ids)
    assert len(train) == 22 and len(test) == 8
    assert len({r["raw"].strip() for r in manifest}) == 30
    for r in manifest:
        ids = {f"e{i}" for i in range(len(r["entities"]))}
        for f in r["gold"]:
            model.frame_key(f)
            assert set(f[1:]) <= ids | {"OBJECT", "UNSPECIFIED"}
        assert {model.frame_key(f) for f in r["targets"]} <= {model.frame_key(f) for f in r["gold"]}
        if r["id"] in train_ids:
            assert not any(w.startswith(("mul", "sech", "sec")) for w in r["tokens"]), r["id"]
            assert "daughter" not in r["translation"].lower()
    # Stronger than deleting a glossary entry: predictor has no non-name spelling
    # or glossary at all. Its exact allowed public schema is checked in tests.
    morphology = model.name_morphology()
    assert views == {r["id"]: model.public_view(r, morphology) for r in manifest}
    return {"records": len(manifest), "training_monuments": len(train), "hidden_monuments": len(test),
            "restored_records": [r["id"] for r in manifest if r["source_reading_has_restoration"]],
            "public_duplicate_templates": [[a["id"], b["id"]] for i,a in enumerate(manifest)
                for b in manifest[:i] if model.public_fingerprint(views[a["id"]]) == model.public_fingerprint(views[b["id"]])],
            "exclusion_checked": True}


def freeze():
    if (OUT / "results.json").exists():
        raise RuntimeError("Cannot freeze after scoring")
    files = ["etruscan/scaffolding.py", "etruscan/corpus.py", "etruscan/classes.py",
             "experiments/etruscan_scaffolding.py", "tests/test_etruscan_scaffolding.py",
             *[str(OUT / name) for name in ("PROTOCOL.md", "AUDIT.md", "manifest.json", "public.json")]]
    sources = [str(corpus.LARTH / f) for f in ("Etruscan.csv", "ETP_POS.csv")]
    barrier = audit_barrier()
    write(OUT / "freeze.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                               "files": {p:sha(p) for p in files}, "sources": {p:sha(p) for p in sources},
                               "barrier": barrier,
                               "runtime": {"python":platform.python_version(), "numpy":np.__version__, "pandas":pd.__version__}})
    print(json.dumps(barrier, indent=2))


def verify():
    frozen = json.loads((OUT / "freeze.json").read_text())
    for kind in ("files", "sources"):
        for path, expected in frozen[kind].items():
            if sha(path) != expected:
                raise RuntimeError(f"Frozen input changed: {path}")
    audit_barrier()
    return frozen


def evaluate(train, test, views, method="anchored"):
    out = []
    joint = model.predict_joint(train,[views[r["id"]] for r in test])[0] if method=="joint" else None
    for r in test:
        view = views[r["id"]]
        if method == "joint":
            prediction = joint[r["id"]]
        elif method == "count_only":
            prediction = model.count_baseline(train, view)
        else:
            prediction = model.predict(train, view, morphology=method != "no_morphology")
        out.append({"id":r["id"], "target_family":r["target_family"], "targets":r["targets"],
                    "gold":r["gold"], "prediction":prediction})
    return out


def shuffle_frames(train, rng):
    """Shuffle whole translations/graphs among equal-entity-count inscriptions.

    This preserves valid participant indices and class frequencies, while breaking
    the relation between the inscription skeleton and its English role graph.
    """
    buckets = collections.defaultdict(list)
    for i, r in enumerate(train):
        buckets[len(r["public"]["entities"])].append(i)
    output = [{"public":r["public"], "frames":r["frames"]} for r in train]
    for indexes in buckets.values():
        permuted = indexes.copy()
        rng.shuffle(permuted)
        for i,j in zip(indexes, permuted):
            output[i]["frames"] = train[j]["frames"]
    return output


def run():
    if (OUT / "results.json").exists():
        raise FileExistsError("Results already exist")
    verify()
    manifest, views, train, test = load()
    records = {method:evaluate(train, test, views, method) for method in ("joint", "anchored", "no_morphology", "count_only")}
    summaries = {method:model.score(rs) for method,rs in records.items()}
    # Leave-one-monument-out on the 22 training records is a descriptive competence
    # check only. No threshold or parameter is selected from its scores.
    development = []
    by_id = {r["id"]:r for r in manifest}
    for i,r in enumerate(train):
        item = dict(by_id[r["public"]["id"]], targets=by_id[r["public"]["id"]]["gold"], target_family="development")
        development += evaluate(train[:i]+train[i+1:], [item], views)
    rng = random.Random(240925)
    nulls = []
    for i in range(NULLS):
        shuffled = shuffle_frames(train, rng)
        nulls.append(model.score(evaluate(shuffled,test,views,"joint")))
        if (i+1) % 200 == 0:
            print(f"Translation permutation {i+1}/{NULLS}", flush=True)
    primary = summaries["joint"]
    p = (1+sum(s["target_recall"]>=primary["target_recall"] for s in nulls))/(NULLS+1)
    checks = {"target_recall_ge_75pct":primary["target_recall"]>=0.75,
              "each_target_family_recall_ge_60pct":all(v["recall"]>=0.6 for v in primary["by_target"].values()),
              "edge_precision_ge_80pct":(primary["edge_precision"] or 0)>=0.8,
              "coverage_ge_50pct":primary["coverage"]>=0.5,
              "beats_count_baseline_by_25pp":primary["target_recall"]>=summaries["count_only"]["target_recall"]+0.25,
              "translation_shuffle_p_le_05":p<=0.05}
    result = {"freeze_sha256":sha(OUT/"freeze.json"), "records":records, "summaries":summaries,
              "translation_nulls":nulls, "p_translation_shuffle":p,
              "gate_checks":checks, "passed":all(checks.values()),
              "development_records":development, "development_summary":model.score(development),
              "joint_family_evidence":model.predict_joint(train,[views[r["id"]] for r in test])[1]}
    write(OUT/"results.json", result)
    print(json.dumps({k:result[k] for k in ("summaries","development_summary","p_translation_shuffle","gate_checks","passed")},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build","freeze","run"))
    args=parser.parse_args()
    {"build":build,"freeze":freeze,"run":run}[args.action]()


if __name__ == "__main__":
    main()
