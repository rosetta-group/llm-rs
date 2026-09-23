"""Lexicons with a proper-name flag per spelled form (round two).

Round one's ``lexicons.py`` is frozen; this module adds the part-of-speech information that the
context test needs and reuses round one's reading and spelling rules unchanged.
"""

from linear_a.lexicons import EXCLUDED_POS, SOURCE_DIR, _romanizations, read_entries
from linear_a.spelling import spell


def build_named_lexicon(stem, source_dir=SOURCE_DIR, min_syllables=2):
    """``{syllables: {"headwords": [...], "name_share": float}}``.

    ``name_share`` is the share of the form's source entries whose part of speech is ``name``
    (personal names, place names, theonyms).
    """
    forms = {}
    for entry in read_entries(stem, source_dir):
        pos = entry.get("pos")
        if pos in EXCLUDED_POS:
            continue
        for roman in _romanizations(entry):
            if " " in roman.strip():
                continue
            syllables = spell(roman)
            if syllables is None or len(syllables) < min_syllables:
                continue
            record = forms.setdefault(syllables, {"headwords": set(), "names": 0, "entries": 0})
            record["headwords"].add(entry.get("word", roman))
            record["entries"] += 1
            record["names"] += pos == "name"
    return {
        key: {"headwords": sorted(r["headwords"]), "name_share": r["names"] / r["entries"]}
        for key, r in forms.items()
    }
