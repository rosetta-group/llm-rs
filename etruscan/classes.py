"""Meaning classes for Etruscan (from the ETP word list) and Latin (from rules), and the Latin epitaph pool.

Five classes, the same in both languages:

    NAME   personal names (praenomen, family name, cognomen), any case
    KIN    kinship and household status: son, wife, freedman ...
    NUM    numerals, as words or as Roman numerals
    LIFE   age, time, living, dying, burial: years, lived, died, tomb ...
    OTHER  every other known word: pronouns, verbs of giving, gods, places, objects

A type with no label is left out of seeds and scoring but still serves as context.
"""

import re

import pandas as pd

from etruscan import corpus

CLASSES = ("NAME", "KIN", "NUM", "LIFE", "OTHER")

KIN_GLOSS = {"son", "daughter", "wife", "husband", "mother", "father", "brother", "sister",
             "grandson", "granddaughter", "grandfather", "grandmother", "grandchild", "nephew",
             "niece", "uncle", "aunt", "freedman", "freedwoman", "descendant", "descendants",
             "sons", "daughters", "parents", "spouse", "children", "child"}
LIFE_GLOSS = {"year", "years", "lived", "live", "dead", "died", "die", "age", "aged", "tomb",
              "grave", "buried", "burial", "month", "months", "day", "days", "life", "death"}
NAME_POS = re.compile(r"\b(prae|nomen|cogn)\b|masc name|fem name")


def etruscan_label(token, glossed):
    """Class of an Etruscan type, or None if it has no usable entry."""
    if token.startswith("N:"):
        return "NUM"
    entry = glossed.get(token)
    if entry is None:
        return None
    pos, glosses = entry
    pos = pos if isinstance(pos, str) else ""
    words = {w for g in glosses for w in re.findall(r"[a-z]+", g.lower())}
    if not pos and not words:
        return None
    if NAME_POS.search(pos):
        return "NAME"
    if re.search(r"\bnum\b", pos):
        return "NUM"
    if words & KIN_GLOSS:
        return "KIN"
    if words & LIFE_GLOSS:
        return "LIFE"
    return "OTHER"


LATIN_KIN = set("""
filius fili filio filium filii filiis filiorum filia filiae filiam filiabus filis
frater fratri fratris fratrem fratres fratribus soror sorori sororis sororem
pater patri patris patrem mater matri matris matrem parens parenti parentis parentes parentibus
coniux coniunx coniugi coniugis coniugem uxor uxori uxoris uxorem maritus marito mariti maritum
nepos nepoti nepotis nepotes neptis nepti avus avo avi avia aviae
libertus liberto liberti libertum liberta libertae libertam libertis libertabus
conlibertus conliberto conliberti conliberta conlibertae colliberto collibertae
patronus patrono patroni patronum patrona patronae patronis
alumnus alumno alumni alumna alumnae contubernalis contubernali socer socero socrus gener genero
nurus nurui vitricus vitrico privignus privigno privigna privignae
""".split())
LATIN_LIFE = set("""
vixit vixerunt vivus viva vivi vivae vivit annos annis annorum anno annum ann an
menses mensibus mensium mense mensem dies diebus dierum die horas horis
obiit obierunt defunctus defuncta defuncto defunctae decessit aetatis
situs sita siti sitae sepulcrum sepulcro sepulchrum monumentum monumento tumulus tumulo
ossa cineres mortuus mortua
""".split())
LATIN_GODS = {"Dis", "Diis", "Deis", "Manibus", "Deo", "Dea", "Sacrum"}
ROMAN = re.compile(r"[IVXLC]+")
LATIN_LETTERS = re.compile(r"[A-Za-z]+")


def latin_label(token):
    """Class of a Latin token as it appears in LIRE's interpretive text, or None."""
    if len(token) == 1:
        return "NUM" if token in "IVX" else None
    if ROMAN.fullmatch(token):
        return "NUM"
    if token in LATIN_GODS:
        return "OTHER"
    if token[0].isupper():
        return "NAME"
    if token in LATIN_KIN:
        return "KIN"
    if token in LATIN_LIFE:
        return "LIFE"
    return "OTHER"


def latin_type(token):
    """Type string for Latin, marking numerals the way Etruscan numerals are marked."""
    return "N:" + token.lower() if ROMAN.fullmatch(token) and latin_label(token) == "NUM" else token


def latin_epitaphs(path=corpus.Path("artifacts/etruscan-sources/lire/LIRE_v3-0.parquet")):
    """Pagan Latin epitaphs as token lists: Latin letters only, HTML residue (lt, gt) dropped."""
    table = pd.read_parquet(path, columns=["type_of_inscription_clean", "inscr_type", "clean_text_interpretive_word"])
    epitaph = table.type_of_inscription_clean.eq("epitaph") | table.inscr_type.fillna("").eq("tituli sepulcrales")
    out = []
    for text in table.loc[epitaph, "clean_text_interpretive_word"].fillna(""):
        words = [w for w in text.split() if w not in ("lt", "gt")]
        if words and all(LATIN_LETTERS.fullmatch(w) for w in words):
            out.append(words)
    return out
