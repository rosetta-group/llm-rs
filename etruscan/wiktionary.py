"""Etruscan glosses from the Wiktionary extract pinned on the Linear A track (kaikki.org, CC BY-SA).

Entries are headed in Old Italic script. Each entry and each of its listed forms is
transliterated to the corpus spelling (``etruscan.corpus.normalise``) and given the entry's class.
"""

import json
import re
from pathlib import Path

from etruscan import corpus
from etruscan.classes import KIN_GLOSS, LIFE_GLOSS

EXTRACT = Path("artifacts/linear-a-sources/kaikki/Etruscan.jsonl")

OLD_ITALIC = {"𐌀": "a", "𐌁": "b", "𐌂": "c", "𐌃": "d", "𐌄": "e", "𐌅": "v", "𐌆": "z", "𐌇": "h",
              "𐌈": "th", "𐌉": "i", "𐌊": "k", "𐌋": "l", "𐌌": "m", "𐌍": "n", "𐌎": "s", "𐌏": "o",
              "𐌐": "p", "𐌑": "s", "𐌒": "q", "𐌓": "r", "𐌔": "s", "𐌕": "t", "𐌖": "u", "𐌗": "ch",
              "𐌘": "ph", "𐌙": "ch", "𐌚": "f", "𐌛": "r", "𐌜": "ch", "𐌝": "i", "𐌞": "u", "𐌟": "s"}
NOT_PERSON = re.compile(r"\b(god|goddess|deity|city|town|river|month|hero|mytholog)", re.I)
NOT_PERSON_CATEGORIES = {"Mythology", "Gods", "Mythological figures", "Months", "Religion"}
SKIP_GLOSS = re.compile(r"^(romanization of|abbreviation of|The meaning of this term is uncertain)", re.I)
# "genitive singular of 𐌀𐌅𐌉𐌋 (avil)": the form takes the class of the Old Italic headword it names.
FORM_OF = re.compile(r"^[^(]*\bof ([𐌀-𐌯]+)")
# Household status, as "freedman" is in KIN_GLOSS; Wiktionary glosses lautni as "slave".
KIN_EXTRA = {"slave", "slaves", "household"}


def transliterate(word):
    word = "".join(OLD_ITALIC.get(ch, ch) for ch in word)
    return corpus.normalise(word)


def entry_class(entry):
    senses = entry.get("senses", [])
    glosses = [g for s in senses for g in s.get("glosses", []) if not SKIP_GLOSS.match(g)]
    if not glosses:
        return None
    categories = {c["name"] for s in senses for c in s.get("categories", [])}
    text = " ".join(glosses)
    if entry["pos"] == "name":
        return "OTHER" if NOT_PERSON.search(text) or categories & NOT_PERSON_CATEGORIES else "NAME"
    if entry["pos"] == "num":
        return "NUM"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & (KIN_GLOSS | KIN_EXTRA):
        return "KIN"
    if words & LIFE_GLOSS:
        return "LIFE"
    return "OTHER"


def form_of(entry):
    """Old Italic headword if every usable gloss is "... of <headword>", else None."""
    glosses = [g for s in entry.get("senses", []) for g in s.get("glosses", []) if not SKIP_GLOSS.match(g)]
    targets = {m.group(1) if (m := FORM_OF.match(g)) else None for g in glosses}
    return targets.pop() if len(targets) == 1 and None not in targets else None


def labels(path=EXTRACT):
    """``{spelling: class}``; a spelling given two different classes is dropped as ambiguous."""
    with open(path, encoding="utf-8") as handle:
        entries = [e for e in map(json.loads, handle) if e["pos"] not in ("romanization", "suffix")]
    heads = {}
    for entry in entries:
        if form_of(entry) is None and (c := entry_class(entry)):
            heads.setdefault(entry["word"], c)
    out, clash = {}, set()
    for entry in entries:
        target = form_of(entry)
        if entry["pos"] in ("name", "num"):
            c = entry_class(entry)
        elif target is not None:
            c = heads.get(target)
        else:
            c = entry_class(entry)
        if c is None:
            continue
        forms = {entry["word"]} | {f["form"] for f in entry.get("forms", []) if f.get("form")}
        for form in forms:
            w = transliterate(form)
            if not w or " " in w or "-" in w:
                continue
            if out.get(w, c) != c:
                clash.add(w)
            out[w] = c
    return {w: c for w, c in out.items() if w not in clash}
