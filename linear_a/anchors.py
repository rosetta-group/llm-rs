"""Place-name anchors: do Linear B readings of Cretan toponyms occur in Linear A?

If the assumed sign values are roughly right, Cretan place names written in Linear B should also
appear in Linear A, which was written in the same places a century or two earlier. The list is
fixed here, before looking, from the standard Knossos toponyms (Ventris and Chadwick; Bennet).
The null is the number found in pseudo-corpora from a syllable bigram model of Linear A.
"""

import numpy as np

from linear_a.matching import BigramNull
from linear_a.spelling import parse_syllabic

TOPONYMS = {
    "pa-i-to": "Phaistos", "ko-no-so": "Knossos", "a-mi-ni-so": "Amnisos",
    "ku-do-ni-ja": "Kydonia", "tu-ri-so": "Tylissos", "su-ki-ri-ta": "Sybrita",
    "ru-ki-to": "Lyktos", "ra-to": "Lato", "u-ta-no": "Itanos", "se-to-i-ja": "Setoia",
    "tu-ni-ja": "Tunia", "di-ka-ta": "Dikte", "da-wo": "Dawos", "e-ko-so": "Eksos",
}


def _contains(word, part):
    n = len(part)
    return any(word[i:i + n] == part for i in range(len(word) - n + 1))


def found(words, toponyms):
    """``{toponym: {"exact": [...], "inside": [...]}}`` over syllable tuples."""
    out = {}
    for name, syllables in toponyms.items():
        exact = [w for w in words if w == syllables]
        inside = [w for w in words if w != syllables and _contains(w, syllables)]
        out[name] = {"exact": exact, "inside": inside}
    return out


def anchor_test(words, rng, repeats=200):
    toponyms = {name: parse_syllabic(name) for name in TOPONYMS}
    hits = found(words, toponyms)
    observed_exact = sum(1 for h in hits.values() if h["exact"])
    observed_any = sum(1 for h in hits.values() if h["exact"] or h["inside"])
    null = BigramNull(words, rng)
    lengths = [len(w) for w in words]
    exact_null, any_null = [], []
    for _ in range(repeats):
        pseudo = set(null.sample(n) for n in lengths)
        h = found(pseudo, toponyms)
        exact_null.append(sum(1 for v in h.values() if v["exact"]))
        any_null.append(sum(1 for v in h.values() if v["exact"] or v["inside"]))
    exact_null, any_null = np.array(exact_null), np.array(any_null)
    return {
        "toponyms": len(TOPONYMS),
        "hits": {name: {"place": TOPONYMS[name],
                        "exact": ["-".join(c + v for c, v in w) for w in h["exact"]],
                        "inside": ["-".join(c + v for c, v in w) for w in h["inside"]]}
                 for name, h in hits.items()},
        "observed_exact": observed_exact,
        "observed_exact_or_inside": observed_any,
        "null_exact_mean": float(exact_null.mean()),
        "null_any_mean": float(any_null.mean()),
        "p_exact": float((exact_null >= observed_exact).mean()),
        "p_any": float((any_null >= observed_any).mean()),
        "null_repeats": repeats,
    }
