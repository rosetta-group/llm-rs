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


def tokens(text):
    """Word tokens of one line; a token containing ``-`` or a digit is kept but marked damaged."""
    if not isinstance(text, str):
        return []
    return [w for w in (normalise(t) for t in SEPARATORS.split(text)) if w]


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


def texts(rows):
    """``[(source, id, [tokens])]``. ETP rows are whole texts; CIEP rows are lines, joined by CIE number in line order."""
    out = []
    for _, row in rows[rows.source == "ETP"].iterrows():
        out.append(("ETP", row.ID, tokens(row.Etruscan)))
    ciep = rows[rows.source == "CIEP"].copy()
    ciep["line"] = ciep["key"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for cid, group in ciep.sort_values(["ID", "line"], kind="stable").groupby("ID", sort=False):
        out.append(("CIEP", cid, [t for text in group.Etruscan for t in tokens(text)]))
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
