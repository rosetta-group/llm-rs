"""Read IVTFF without treating its annotations as manuscript text."""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

VERSION = 1
PAGE = re.compile(r"^<(f(?:\d+[rv]\d*|Ros))>\s*(.*)$")
LOCUS = re.compile(r"^<(f(?:\d+[rv]\d*|Ros))\.(\d+),(.)([PLCR])(\w)(?:;([^>]+))?>\s*(.*)$")
SECTIONS = dict(H="herbal", T="text", A="astronomical", C="cosmological",
                Z="zodiac", B="balneological", P="pharmaceutical", S="stars")
DRAWING = "\x1c"
MISALIGNED_DRAWING = "\x1d"
BOUNDARIES = ".,\n" + DRAWING + MISALIGNED_DRAWING
ROSETTES = {"85", "86", "Ros"}  # One foldout, spanning both numbered folios.


def cluster(value, group):
    return "85-86" if group == "folio" and value in ROSETTES else value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def normalise(raw, boundary="marked", reading="mask"):
    if boundary not in {"marked", "separate", "merged"}:
        raise ValueError("boundary must be marked, separate, or merged")
    if reading not in {"mask", "first", "last"}:
        raise ValueError("reading must be mask, first, or last")
    text = raw.replace("<->", DRAWING).replace("<~>", MISALIGNED_DRAWING)
    text = re.sub(r"<[^>]*>", "", text)

    def alternative(match):
        choices = match[1].split(":")
        return "?" if reading == "mask" else choices[0 if reading == "first" else -1]

    text = re.sub(r"\[([^]]+)\]", alternative, text)
    # @128; ... @255; each denote one extended transcription character.
    def extended(match):
        value = int(match[1])
        if not 128 <= value <= 255:
            raise ValueError(f"Invalid IVTFF extended character: {match[0]}")
        return chr(value)

    text = re.sub(r"@(\d{3});", extended, text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[ \t\r]", "", text)
    if boundary == "separate":
        text = text.replace(",", ".")
    elif boundary == "merged":
        text = text.replace(",", "")
    return text


def read_ivtff(path):
    source = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not source or not source[0].startswith("#=IVTFF "):
        raise ValueError("Expected an IVTFF header")
    alphabet = source[0].split()[1]
    lines, logical = [], []
    pending = ""
    for number, raw in enumerate(source[1:], 2):
        if not raw.strip() or raw.startswith("#"):
            if pending:
                raise ValueError(f"Comment or blank inside wrapped locus at {number}")
            continue
        if pending:
            if not raw.startswith("/"):
                raise ValueError(f"Missing continuation at {number}")
            raw = pending + raw[1:]
        if raw.endswith("/"):
            pending = raw[:-1]
            continue
        pending = ""
        logical.append((number, raw))
    if pending:
        raise ValueError("Unfinished wrapped locus")
    current, metadata, paragraph, active = None, {}, 0, False
    seen = set()
    for number, raw in logical:
        page = PAGE.match(raw)
        if page:
            current = page[1]
            metadata = dict(re.findall(r"\$([A-Z])=(\S+?)(?=\s|>)", page[2]))
            paragraph, active = 0, False
            continue
        match = LOCUS.match(raw)
        if not match or match[1] != current:
            raise ValueError(f"Invalid locus or page at source line {number}")
        page, count, locator, kind, subtype, transcriber, text = match.groups()
        locus = f"{page}.{count}"
        if locus in seen:
            raise ValueError(f"Multiple readings of {locus}; select one transcription first")
        seen.add(locus)
        metadata.update(dict(re.findall(r"<@([A-Z])=([^>]+)>", text)))
        start, end = "<%>" in text, "<$>" in text
        if kind == "P" and (start or not active):
            paragraph += 1
            active = True
        folio = re.match(r"f(\d+)", page)
        lines.append(dict(
            id=locus, page=page, folio=folio[1] if folio else "Ros",
            quire=metadata.get("Q", "unknown"), section=SECTIONS.get(metadata.get("I"), "unknown"),
            currier=metadata.get("L", "unknown"), hand=metadata.get("H", "unknown"),
            paragraph=paragraph if kind == "P" else None, paragraph_start=start,
            paragraph_end=end, locator=locator, kind=kind, subtype=subtype,
            transcriber=transcriber, raw=text, variables=dict(metadata), source_line=number,
        ))
        if end:
            active = False
    if not lines:
        raise ValueError("No manuscript loci found")
    return dict(version=VERSION, source=str(Path(path).resolve()), sha256=digest(path),
                alphabet=alphabet, header=source[0], lines=lines)


def make_manifest(corpus, seed=42, group="folio"):
    if group not in {"folio", "quire"}:
        raise ValueError("Split groups must be folio or quire")
    groups = sorted({line[group] for line in corpus["lines"]})
    if "unknown" in groups or len(groups) < 3:
        raise ValueError("At least three known groups are required")
    bundles = {g: [g] for g in groups}
    if group == "folio" and ROSETTES.intersection(groups):
        members = sorted(ROSETTES.intersection(groups))
        bundles = {g: [g] for g in groups if g not in ROSETTES}
        bundles["85-86"] = members
        groups = sorted(bundles)
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required")
    random.Random(seed).shuffle(groups)
    n = max(1, round(len(groups) * .15))
    assignments = {member: "test" if i < n else "validation" if i < 2*n else "train"
                   for i, g in enumerate(groups) for member in bundles[g]}
    return dict(version=VERSION, source_sha256=corpus["sha256"], seed=seed,
                group=group, assignments=assignments)


def validate_manifest(corpus, manifest, allow_other_source=False):
    if manifest["version"] != VERSION:
        raise ValueError("Unsupported split manifest version")
    if not allow_other_source and corpus["sha256"] != manifest["source_sha256"]:
        raise ValueError("Source changed; create a new manifest or explicitly match another transcription")
    if set(manifest["assignments"].values()) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test groups")
    if manifest["group"] not in {"folio", "quire"}:
        raise ValueError("Invalid grouping")
    if manifest["group"] == "folio":
        sides = {manifest["assignments"][g] for g in ROSETTES if g in manifest["assignments"]}
        if len(sides) > 1:
            raise ValueError("The Rosettes foldout (85, 86, Ros) must stay in one split")


def shuffle_line(text, seed, locus):
    # Keep boundary types and slots fixed; shuffle only complete written forms.
    pieces = re.split("([.,\x1c\x1d])", text)
    indices = [i for i in range(0, len(pieces), 2) if pieces[i]]
    words = [pieces[i] for i in indices]
    rng = random.Random(f"{seed}:{locus}")
    rng.shuffle(words)
    for i, word in zip(indices, words):
        pieces[i] = word
    return "".join(pieces)


def documents(corpus, manifest, boundary="marked", reading="mask", control="intact", seed=42,
              pages=None, allow_other_source=False):
    validate_manifest(corpus, manifest, allow_other_source)
    if control not in {"intact", "shuffle"}:
        raise ValueError("Control must be intact or shuffle")
    docs = {}
    for line in corpus["lines"]:
        if line["kind"] != "P" or line["locator"] == "!" or (pages is not None and line["page"] not in pages):
            continue
        group = line[manifest["group"]]
        if group not in manifest["assignments"]:
            if allow_other_source:
                continue
            raise ValueError(f"Group missing from manifest: {group}")
        text = normalise(line["raw"], boundary, reading)
        if control == "shuffle":
            text = shuffle_line(text, seed, line["id"])
        if not text:
            continue
        if line["page"] not in docs:
            docs[line["page"]] = {k: line[k] for k in ("page", "folio", "quire", "section", "currier", "hand")}
            docs[line["page"]].update(text="", loci=[], split=manifest["assignments"][group])
        doc = docs[line["page"]]
        if doc["text"]:
            new_paragraph = doc["loci"][-1]["paragraph"] != line["paragraph"]
            doc["text"] += "\n\n" if new_paragraph else "\n"
        begin = len(doc["text"])
        doc["text"] += text
        doc["loci"].append(dict(id=line["id"], start=begin, end=len(doc["text"]),
                                 paragraph=line["paragraph"], variables=line["variables"]))
    return list(docs.values())


def build(source, output, manifest_path, seed=42, group="folio", **kwargs):
    corpus = read_ivtff(source)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["seed"] != seed or manifest["group"] != group:
            raise ValueError("Existing manifest uses different settings; choose a new path")
    else:
        manifest = make_manifest(corpus, seed, group)
        write_json(manifest_path, manifest)
    docs = documents(corpus, manifest, **kwargs)
    if set(d["split"] for d in docs) != {"train", "validation", "test"}:
        raise ValueError("A split has no paragraph text")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(seed=seed, group=group, **kwargs)
    audit = dict(source=corpus["source"], source_sha256=corpus["sha256"], alphabet=corpus["alphabet"],
                 preprocessing_version=VERSION, settings=settings,
                 manifest_sha256=digest(manifest_path), loci=len(corpus["lines"]),
                 locus_types=dict(Counter(l["kind"] for l in corpus["lines"])),
                 splits={s: dict(pages=sum(d["split"] == s for d in docs),
                                 characters=sum(len(d["text"]) for d in docs if d["split"] == s),
                                 groups=len({cluster(d[group], group) for d in docs if d["split"] == s}),
                                 currier=dict(Counter(d["currier"] for d in docs if d["split"] == s)))
                         for s in ("train", "validation", "test")})
    write_json(output / "corpus.json", corpus)
    write_json(output / "manifest.json", manifest)
    write_json(output / "documents.json", docs)
    write_json(output / "audit.json", audit)
    return audit


def load_documents(dataset):
    dataset = Path(dataset)
    return json.loads((dataset / "documents.json").read_text())
