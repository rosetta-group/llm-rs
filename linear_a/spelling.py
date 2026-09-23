"""Syllable representation and Linear-B-style spelling of alphabetic words.

A syllable is ``(consonant, vowel)``: consonant is ``""`` for a bare vowel sign, vowel is one of
``a e i o u`` or ``?`` for an unknown vowel (consonantal transliterations).

The spelling rules are the standard Linear B conventions (Ventris and Chadwick):

- voicing and aspiration are not written except for ``d``: b/p/ph -> p, g/k/kh -> k, t/th -> t;
- r and l are one series; labiovelars q are written k here, in both scripts;
- in a consonant cluster, a first s, m, n, r or l is dropped; a first stop is written with the
  vowel of the following syllable (Knossos -> ko-no-so);
- word-final consonants are dropped; h and glottal signs are dropped;
- a vowel after i gets a j glide and a vowel after u a w glide (xenwia -> ke-se-wi-ja);
- a word with no vowels at all (a consonantal transliteration) gets ``?`` after each consonant.
"""

import re
import unicodedata

VOWELS = "aeiou"
CONSONANTS = "djkmnprstwz"
CODA_DROPPED = set("smnr")

# Characters that survive diacritic stripping but need mapping. Applied after NFD stripping.
_LETTER_MAP = {
    "b": "p", "f": "p", "v": "w", "g": "k", "c": "k", "q": "k", "x": "ks", "l": "r",
    "y": "j", "h": "", "ə": "e", "ǝ": "e", "æ": "e", "ø": "o", "œ": "e",
    "ꞽ": "i", "ı": "i", "θ": "t", "χ": "k", "φ": "p", "ψ": "ps", "ž": "z", "ʒ": "z", "ʃ": "s", "ŋ": "n",
}
_DROP = set("ʿʾʔ'’ʼ`ʰʷʲ·.-=()[]⸢⸣<>{}/ ꜣꜥ")


def strip_diacritics(text):
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_letters(word):
    """Lower-case letter string over ``VOWELS + CONSONANTS``, or ``None`` if unusable."""
    text = strip_diacritics(word)
    out = []
    for ch in text:
        if ch in _DROP:
            continue
        ch = _LETTER_MAP.get(ch, ch)
        out.append(ch)
    text = "".join(out)
    # ph/th/kh already lost their h; y between consonants or at an edge next to one is a vowel.
    text = re.sub(r"(?<![aeiou])j(?![aeiou])", "i", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    if not text or any(ch not in VOWELS + CONSONANTS for ch in text):
        return None
    return text


def spell(word):
    """Spell an alphabetic word as Linear-B-style syllables; ``None`` if unusable."""
    letters = normalise_letters(word)
    if letters is None:
        return None
    if not any(ch in VOWELS for ch in letters):
        return tuple((ch, "?") for ch in letters)
    syllables = []
    cluster = []
    for ch in letters:
        if ch in VOWELS:
            for first in cluster[:-1]:
                if first not in CODA_DROPPED:
                    syllables.append((first, ch))
            onset = cluster[-1] if cluster else ""
            if not onset and syllables and syllables[-1][1] in "iu":
                onset = "j" if syllables[-1][1] == "i" else "w"
            syllables.append((onset, ch))
            cluster = []
        else:
            cluster.append(ch)
    # Final consonants are dropped.
    return tuple(syllables) or None


_SIGN = re.compile(r"^([dhjkmnpqrstwz]?)([aeiou])[23]?$")


def parse_syllabic(word):
    """Parse a transliterated syllabic word such as ``ma-ra-tu-wo`` or ``qe-ra2-u``.

    Returns ``None`` when any sign has no assumed phonetic value (``*79``, ``A306``, ``[?]``).
    Complex Linear B signs are reduced to first consonant plus vowel (``nwa`` -> ``na``).
    """
    out = []
    for sign in word.lower().replace("?", "").split("-"):
        sign = re.sub(r"^([dkmnprst])[wj]", r"\1", sign)
        match = _SIGN.match(sign)
        if not match:
            return None
        consonant, vowel = match.groups()
        consonant = {"q": "k", "h": ""}.get(consonant, consonant)
        out.append((consonant, vowel))
    return tuple(out) or None


def render(syllables):
    return "-".join(c + v for c, v in syllables)
