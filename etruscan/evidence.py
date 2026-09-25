"""Integrity checks for cited Etruscan development annotations, not a predictor."""

import collections
import hashlib
import json
from pathlib import Path

from etruscan.scaffolding import frame_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(bundle,verify_files=True):
    if bundle.get("schema_version")!=1:
        raise ValueError("Unknown evidence schema")
    if bundle.get("purpose")!="development_evidence_only":
        raise ValueError("Exposed inscriptions cannot become a fresh evaluation set")
    sources={s["id"]:s for s in bundle["sources"]}
    if len(sources)!=len(bundle["sources"]):
        raise ValueError("Duplicate source identifier")
    if verify_files:
        for source in sources.values():
            if source.get("local_path") and sha(source["local_path"])!=source["sha256"]:
                raise ValueError(f"Source bytes changed: {source['id']}")
    seen=set()
    for record in bundle["records"]:
        if record["id"] in seen:
            raise ValueError("Duplicate inscription")
        seen.add(record["id"])
        if record["expert_review"]!="pending" or record["fresh_evaluation_eligible"]:
            raise ValueError("This source audit does not constitute expert review or fresh evaluation")
        if record["status"] not in {"corroborated","partially_corroborated","disputed","unverified"}:
            raise ValueError("Unknown record status")
        original=record["original"]
        if verify_files:
            if sha(original["manifest_path"])!=original["manifest_sha256"]:
                raise ValueError("Historical manifest changed")
            reference=next(r for r in json.loads(Path(original["manifest_path"]).read_text()) if r["id"]==record["id"])
            if any(record[key]!=reference[key] for key in ("tokens","entities")) or original["gold"]!=reference["gold"]:
                raise ValueError("Audit silently changed the original text, anchors or reference graph")
        n=len(record["tokens"])
        def spans(indices):
            if len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<n for i in indices):
                raise ValueError("Invalid token span")
        def evidence(refs):
            for ref in refs:
                if ref["source"] not in sources or not ref.get("locator"):
                    raise ValueError("Unknown source or missing pinpoint citation")
        for item in record["spans"]+record["relations"]:
            spans(item["tokens"])
            evidence(item["evidence"])
            if item["status"] in {"corroborated","disputed"} and not item["evidence"]:
                raise ValueError("Supported/disputed annotation needs a source")
            if item["status"] not in {"corroborated","disputed","carried_forward"}:
                raise ValueError("Unknown annotation status")
        entities={f"e{i}" for i in range(len(record["entities"]))}
        for relation in record["relations"]:
            frame_key(relation["frame"])
            if any(a not in entities|{"OBJECT","UNSPECIFIED"} for a in relation["frame"][1:]):
                raise ValueError("Unknown participant")
        for issue in record["issues"]:
            evidence(issue["evidence"])
        if record["status"]=="corroborated" and (not record["relations"] or any(r["status"]!="corroborated" for r in record["relations"])):
            raise ValueError("Whole-graph corroboration cannot include unsupported relations")
        if record["status"]=="unverified" and any(r["status"]=="corroborated" for r in record["relations"]):
            raise ValueError("Unverified record cannot assert corroborated relations")
    return {"records":len(seen),"sources":len(sources),
            "pinned_pdfs":sum(bool(s.get("local_path")) for s in sources.values()),
            "record_statuses":dict(collections.Counter(r["status"] for r in bundle["records"])),
            "source_supported_relations":sum(r["status"]=="corroborated" for x in bundle["records"] for r in x["relations"]),
            "pending_expert_review":len(seen),"fresh_evaluation_records":0}
