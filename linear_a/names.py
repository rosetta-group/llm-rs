"""Proper-name lexicons (round three): people, places and gods, spelled with Linear B rules.

Sources (pinned in experiments/linear-a-names/sources.json):

- Greek: Wiktionary Ancient Greek entries with part of speech ``name``.
- Anatolian: LAMAN, names attested in Hittite texts (CC BY-SA 4.0); phonetic spellings only.
  Hittite, Luwian and Hurrian names are mixed and not separable.
- Levant: Oracc ``aemw`` proper nouns for Ugarit, Amarna and Alalakh (CC0).
- Babylonia: Oracc ``rimanum``, an Old Babylonian archive (CC0).

Egyptian names are unvocalised and are not used (see the round-three protocol).
"""

import csv
import json
import re
from pathlib import Path

from linear_a.lexicons import _romanizations, read_entries
from linear_a.spelling import spell

NAMES_DIR = Path("artifacts/linear-a-sources/names")
QPN_KINDS = {"PN", "GN", "DN", "SN", "RN", "EN", "WN"}
QPN_FILES = {
    "Levant": ["aemw/ugarit/gloss-qpn.json", "aemw/amarna/gloss-qpn.json",
               "aemw/alalakh/idrimi/gloss-qpn.json"],
    "Babylonia": ["rimanum/gloss-qpn.json"],
}
_LOGOGRAPHIC = re.compile(r"[A-ZŠḪṢṬ]{2,}|[/\d{}]|OR")


def _add(lexicon, text, source):
    if not text or " " in text.strip() or _LOGOGRAPHIC.search(text):
        return
    syllables = spell(text)
    if syllables is not None:
        lexicon.setdefault(syllables, set()).add(source)


def wiktionary_names(stem):
    lexicon = {}
    for entry in read_entries(stem):
        if entry.get("pos") != "name":
            continue
        for roman in _romanizations(entry):
            _add(lexicon, roman, entry.get("word", roman))
    return lexicon


def laman_names(path=NAMES_DIR / "laman_names.csv"):
    lexicon = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Writing Type"] != "phonetic":
                continue
            for text in [row["Name"]] + [v.strip() for v in row["Variant Forms"].split(",")]:
                _add(lexicon, text, row["Name"])
    return lexicon


def qpn_names(files):
    lexicon = {}
    for name in files:
        for entry in json.loads((NAMES_DIR / name).read_text(encoding="utf-8"))["entries"]:
            if entry.get("pos") not in QPN_KINDS:
                continue
            texts = [entry.get("cf", "")] + [n.get("n", "") for n in entry.get("norms", [])]
            for text in texts:
                # Consonantal Ugaritic normalisations have no vowels; keep only vocalised forms.
                if re.search(r"[aeiouāēīōūâêîôû]", text.lower()):
                    _add(lexicon, text, entry.get("cf", text))
    return lexicon


def build_name_lexicons(min_syllables):
    raw = {
        "Greek": wiktionary_names("AncientGreek"),
        "Anatolian": laman_names(),
        "Levant": qpn_names(QPN_FILES["Levant"]),
        "Babylonia": qpn_names(QPN_FILES["Babylonia"]),
    }
    return {
        language: {form: sorted(heads) for form, heads in lexicon.items()
                   if len(form) >= min_syllables}
        for language, lexicon in raw.items()
    }
