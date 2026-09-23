"""Candidate-language lexicons from pinned Wiktionary extracts (kaikki.org JSONL).

Each lexicon is a mapping from a spelled syllable tuple to the source headwords that produce it.
Only lemma romanizations are used: inflected forms exist for Greek and Hebrew but not for the
small lexicons, and using them would make lexicon size an even larger confound.
"""

import json
import re
from pathlib import Path

from linear_a.spelling import parse_syllabic, spell

SOURCE_DIR = Path("artifacts/linear-a-sources/kaikki")

# The Mycenaean file is the Linear B control, never a candidate language.
CANDIDATES = {
    "Greek": "AncientGreek",
    "Hittite": "Hittite",
    "Akkadian": "Akkadian",
    "Ugaritic": "Ugaritic",
    "Hebrew": "Hebrew",
    "Etruscan": "Etruscan",
    "Egyptian": "Egyptian",
    "Sumerian": "Sumerian",
}
EXCLUDED_POS = {
    "character", "syllable", "symbol", "letter", "punct", "suffix", "prefix", "affix", "infix",
    "interfix", "circumfix", "root", "combining_form", "diacritic", "abbrev",
}
_LATIN = re.compile(r"^[a-zA-ZÀ-ɏḀ-ỿꜢ-ꟿɐ-ʯ'ʿʾ.\- ]+$")


def _romanizations(entry):
    forms = [
        form["form"] for form in entry.get("forms", [])
        if "romanization" in form.get("tags", []) and form.get("form")
    ]
    if forms:
        # All-capital romanizations are Sumerograms or Akkadograms, not words of the language.
        return [form for form in forms if any(ch.islower() for ch in form)]
    word = entry.get("word", "")
    return [word] if _LATIN.match(word) else []


def read_entries(stem, source_dir=SOURCE_DIR):
    with open(Path(source_dir) / f"{stem}.jsonl", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def build_lexicon(stem, source_dir=SOURCE_DIR, min_syllables=1):
    """Return ``{syllables: sorted headwords}`` for one language."""
    lexicon = {}
    for entry in read_entries(stem, source_dir):
        if entry.get("pos") in EXCLUDED_POS:
            continue
        for roman in _romanizations(entry):
            # Multi-word expressions are not single Linear A words.
            if " " in roman.strip():
                continue
            syllables = spell(roman)
            if syllables is None or len(syllables) < min_syllables:
                continue
            lexicon.setdefault(syllables, set()).add(entry.get("word", roman))
    return {key: sorted(value) for key, value in lexicon.items()}


_COGNATE = re.compile(r"Ancient Greek \S+ \(([^)]+)\)")


def mycenaean_words(source_dir=SOURCE_DIR, min_syllables=2):
    """Linear B words with a known Greek reading: ``{syllables: record}``.

    ``cognate`` is the first Ancient Greek romanization named in the etymology or gloss, when
    there is one; it is used only to score whether a match found the right Greek word.
    """
    words = {}
    for entry in read_entries("MycenaeanGreek", source_dir):
        if entry.get("pos") in EXCLUDED_POS:
            continue
        romans = [
            form["form"] for form in entry.get("forms", [])
            if "romanization" in form.get("tags", []) and "-" in form.get("form", "")
        ]
        if not romans:
            continue
        syllables = parse_syllabic(romans[0])
        if syllables is None or len(syllables) < min_syllables:
            continue
        glosses = " ".join(" ".join(s.get("glosses", [])) for s in entry.get("senses", []))
        text = entry.get("etymology_text", "") + " " + glosses
        match = _COGNATE.search(text)
        record = words.setdefault(syllables, {
            "linear_b": romans[0], "pos": entry.get("pos"), "cognate": None,
        })
        if record["cognate"] is None and match:
            record["cognate"] = match.group(1)
    return words
