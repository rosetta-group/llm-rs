"""Etruscan texts from the pinned Larth dataset (ETP and CIEP parts) and the ETP word list."""

import ast
import re
from pathlib import Path

import pandas as pd

LARTH = Path("artifacts/etruscan-sources/larth")

# One lossy Latin-letter spelling for both parts. CIEP already writes th/ch/ph and one s;
# ETP writes Greek letters and several sibilants. Sibilants are merged.
LETTERS = [("σ'", "s"), ("ς'", "s"), ("s'", "s"), ("θ", "th"), ("χ", "ch"), ("φ", "ph"),
           ("σ", "s"), ("ς", "s"), ("ś", "s"), ("š", "s"), ("ê", "e"), ("0", "th")]
SEPARATORS = re.compile(r"[\s:·•|/\;]+")
MARKS = re.compile(r"[.\[\]{}()<>'‘’?!*]")


def normalise(word):
    word = word.lower()
    for old, new in LETTERS:
        word = word.replace(old, new)
    return MARKS.sub("", word)


ROMAN = re.compile(r"[IVXLC]+")


def tokens(text, numerals=False):
    """Word tokens of one line; a token containing ``-`` or a digit is kept but marked damaged.

    With ``numerals``, an upper-case Roman numeral becomes ``N:<numeral>`` before lower-casing,
    so that ``XV`` is not read as a word and ``ci`` ("three") is not read as a numeral.
    A single ``L`` or ``C`` is left as a word: it is usually an abbreviated name.
    """
    if not isinstance(text, str):
        return []
    out = []
    for raw in SEPARATORS.split(text):
        bare = MARKS.sub("", raw)
        if numerals and ROMAN.fullmatch(bare) and (len(bare) > 1 or bare in "IVX"):
            out.append("N:" + bare.lower())
        elif word := normalise(raw):
            out.append(word)
    return out


def damaged(token):
    return "-" in token or any(c.isdigit() for c in token)


def source(row_id):
    """``ETP`` for ETP and Zikh Rasna rows (letter sigla such as ``Cr 2.20``), ``CIEP`` for CIE numbers."""
    return "ETP" if re.match(r"[A-Za-z]", str(row_id).strip()) else "CIEP"


def load_rows(path=LARTH / "Etruscan.csv"):
    rows = pd.read_csv(path, index_col=0)
    rows["ID"] = rows["ID"].astype(str).str.strip()
    rows["source"] = rows["ID"].map(source)
    return rows


def texts(rows, numerals=False):
    """``[(source, id, [tokens])]``. ETP rows are whole texts; CIEP rows are lines, joined by CIE number in line order."""
    out = []
    for _, row in rows[rows.source == "ETP"].iterrows():
        out.append(("ETP", row.ID, tokens(row.Etruscan, numerals)))
    ciep = rows[rows.source == "CIEP"].copy()
    ciep["line"] = ciep["key"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for cid, group in ciep.sort_values(["ID", "line"], kind="stable").groupby("ID", sort=False):
        out.append(("CIEP", cid, [t for text in group.Etruscan for t in tokens(text, numerals)]))
    return out


def glossed_words(path=LARTH / "ETP_POS.csv"):
    """``{normalised form: (POS string, [glosses])}`` from the ETP word list, suffix entries excluded."""
    table = pd.read_csv(path, index_col=0)
    out = {}
    for _, row in table.iterrows():
        if row.get("Is suffix") is True or not isinstance(row.Etruscan, str):
            continue
        try:
            glosses = [g for _, g in ast.literal_eval(row.Translations)]
        except (ValueError, SyntaxError):
            glosses = []
        out.setdefault(normalise(row.Etruscan), (row.POS, glosses))
    return out


def _bag(words):
    import collections
    return collections.Counter(w[2:] if w.startswith("N:") else w for w in words)


def dedupe_within_id(texts, threshold=0.8):
    """Drop a text whose word multiset overlaps an earlier text with the same ID by Jaccard >= ``threshold``.

    Texts with different IDs are kept even when identical: short ownership texts such as
    ``mi larices`` recur on different objects. Returns ``(kept, dropped count)``.
    """
    kept, seen, dropped = [], {}, 0
    for source, tid, words in texts:
        bag = _bag(words)
        earlier = seen.setdefault((source, tid), [])
        if bag and any(sum((bag & b).values()) / sum((bag | b).values()) >= threshold for b in earlier):
            dropped += 1
            continue
        earlier.append(bag)
        kept.append((source, tid, words))
    return kept, dropped
