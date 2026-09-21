"""Read surface text from pinned corpora; never use syntactic word splits as spelling."""

import re
import unicodedata as ud
from pathlib import Path


def written_words(text):
    """Letter runs with internal apostrophes/hyphens; diacritics stay with letters."""
    words, current = [], []
    text = ud.normalize("NFC", text).lower().replace("’", "'")
    for i, char in enumerate(text):
        category = ud.category(char)[0]
        if category in {"L", "M"} and (current or category == "L"):
            current.append(char)
        elif char in "'-" and current and i + 1 < len(text) and ud.category(text[i + 1])[0] == "L":
            current.append(char)
        else:
            if current:
                words.append("".join(current))
                current = []
    if current:
        words.append("".join(current))
    return words


def letter_form(word):
    return "".join(c for c in ud.normalize("NFD", word) if ud.category(c)[0] == "L")


def conllu_sentences(path):
    """Prefer original # text; otherwise reconstruct range tokens and SpaceAfter."""
    for block in Path(path).read_text().strip().split("\n\n"):
        metadata, pieces, skip = {}, [], 0
        for line in block.splitlines():
            if line.startswith("# ") and " = " in line:
                key, value = line[2:].split(" = ", 1)
                metadata[key] = value
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                if len(fields) != 10:
                    raise ValueError("Invalid CoNLL-U row")
                index = fields[0]
                if "." in index:
                    continue
                if "-" in index:
                    skip = int(index.split("-")[1])
                elif int(index) <= skip:
                    continue
                pieces.append(fields[1] + ("" if "SpaceAfter=No" in fields[9] else " "))
        text = metadata.get("text", "".join(pieces).strip())
        if text:
            yield dict(id=metadata.get("sent_id", str(len(pieces))), text=text,
                       words=written_words(text), metadata=metadata)


def voynich_segments(docs, split="train", uncertain="split"):
    """Uncertain glyphs break adjacency. Count written forms, not inferred morphemes."""
    if split != "train":
        raise ValueError("Descriptive comparison is restricted to training pages")
    if uncertain not in {"split", "merge"}:
        raise ValueError("Unknown boundary rule")
    for doc in docs:
        if doc["split"] != split:
            continue
        for i, line in enumerate(doc["text"].splitlines()):
            if uncertain == "merge":
                line = line.replace(",", "")
            words = [w if "?" not in w else None for w in re.split(r"[.,\x1c\x1d]+", line) if w]
            if words:
                yield dict(id=f"{doc['page']}:{i}", page=doc["page"], currier=doc["currier"],
                           section=doc["section"], words=words)
