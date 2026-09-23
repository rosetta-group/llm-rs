"""Word contexts on administrative tablets, for Linear A and Linear B.

Every word token gets one context label from what follows it on the same line:

- ``entry``: a number follows, directly or after logograms (``a-du VIR 20``);
- ``logogram``: a logogram follows and no number does;
- ``header``: the first word of the document, when it is not an entry;
- ``other``: everything else.

In Linear B an entry word is usually a personal name or a place name; that is how the tablets
were read after 1952. Whether the same holds in Linear A is not assumed; only the labels are.
"""

import collections
import re
import unicodedata

from linear_a.spelling import parse_syllabic

LABELS = ("entry", "logogram", "header", "other")
PRIORITY = {label: i for i, label in enumerate(LABELS)}


def label_line_items(items, first_line):
    """``items`` is a list of ``(kind, value)`` with kind in word, logogram, number.

    Returns ``[(word, label)]`` for the words on the line.
    """
    out = []
    for i, (kind, value) in enumerate(items):
        if kind != "word":
            continue
        label = "other"
        rest = items[i + 1:]
        j = 0
        while j < len(rest) and rest[j][0] == "logogram":
            j += 1
        if j < len(rest) and rest[j][0] == "number":
            label = "entry"
        elif j > 0:
            label = "logogram"
        elif first_line and not out and i == 0:
            label = "header"
        out.append((value, label))
    return out


def type_labels(tokens):
    """Majority label per word type; ties broken by ``LABELS`` order."""
    counts = collections.defaultdict(collections.Counter)
    for word, label in tokens:
        counts[word][label] += 1
    return {
        word: min(c, key=lambda lab: (-c[lab], PRIORITY[lab]))
        for word, c in counts.items()
    }


# ---- Linear A (Navarre collation: unicode_text lines, SigLA sign roles) ----------------------

_A_SIGN = re.compile(r"LINEAR A SIGN (AB|A)(\d+)(.*)$")


def _sign_type(ch):
    match = _A_SIGN.match(unicodedata.name(ch, ""))
    if not match:
        return None
    prefix, number, rest = match.groups()
    number = number.lstrip("0") or "0"
    if prefix == "AB":
        number = number.zfill(2)
    return prefix + number + rest.split(" ")[0]


def sign_table(corpus):
    """``{sign type: (reading or None, logogram share)}`` from SigLA per-occurrence roles."""
    roles = collections.defaultdict(collections.Counter)
    readings = collections.defaultdict(collections.Counter)
    for record in corpus.values():
        for sign in record.get("signs") or []:
            roles[sign["type"]][sign["role"]] += 1
            if sign["role"] == "syllabogram" and sign.get("reading"):
                readings[sign["type"]][sign["reading"]] += 1
    table = {}
    for sign_type, c in roles.items():
        total = sum(c.values())
        reading = readings[sign_type].most_common(1)[0][0] if readings[sign_type] else None
        table[sign_type] = (reading, (c["logogram"] + c["transaction"]) / total)
    return table


def _is_numeral(ch):
    return 0x10107 <= ord(ch) <= 0x10133


def _is_fraction(ch):
    return 0x10740 <= ord(ch) <= 0x10755


def linear_a_tokens(corpus, logogram_share=0.8):
    """``(document, word syllables, label)`` for every readable word of two or more signs.

    A run of signs between dividers is split before any sign that is a logogram in at least
    ``logogram_share`` of its SigLA occurrences; a one-sign run with a majority logogram role is
    a logogram. Signs absent from SigLA's table count as unreadable.
    """
    table = sign_table(corpus)
    # Unicode variants (AB131A, AB131B) share the SigLA type of their base sign.
    for sign_type in [t for t in table if t]:
        for variant in "ABC":
            table.setdefault(sign_type + variant, table[sign_type])
    out = []
    for doc, record in sorted(corpus.items()):
        text = record.get("unicode_text") or ""
        first_line = True
        for line in text.split("\n"):
            items, run = [], []

            def flush():
                if not run:
                    return
                if len(run) == 1 and table.get(run[0], (None, 0))[1] >= 0.5:
                    items.append(("logogram", run[0]))
                else:
                    readings = [table.get(t, (None, 0))[0] for t in run]
                    word = None
                    if all(readings):
                        word = parse_syllabic("-".join(readings))
                    items.append(("word", word))
                run.clear()

            for ch in line:
                sign_type = _sign_type(ch)
                if _is_numeral(ch) or _is_fraction(ch):
                    flush()
                    if not items or items[-1][0] != "number":
                        items.append(("number", None))
                elif sign_type is not None and not _is_fraction(ch):
                    if table.get(sign_type, (None, 0))[1] >= logogram_share:
                        flush()
                        items.append(("logogram", sign_type))
                    else:
                        run.append(sign_type)
                else:
                    flush()
            flush()
            if not items:
                continue
            for word, label in label_line_items(items, first_line):
                if word is not None and len(word) >= 2:
                    out.append((doc, word, label))
            first_line = False
    return out


# ---- Linear B (DĀMOS transcriptions) ------------------------------------------------------

_B_WORD = re.compile(r"^[a-z0-9*]+(?:-[a-z0-9*]+)+$|^[a-z]+[23]?$")
_B_NUMBER = re.compile(r"^\d+$")
_B_MEASURE = re.compile(r"^[STVZMNPQL]$")
_B_LOGOGRAM = re.compile(r"^(?:\*\d+[A-Z]*|[A-Z][A-Z+]*[A-Z]?(?:\+[A-Za-z]+)*)$")


def _clean_b(token):
    return re.sub(r"[\[\]⟦⟧<>{}̣?!]|^,|,$", "", token)


def linear_b_line_items(line):
    items = []
    line = re.sub(r"^\s*\.\S*\s*", "", line)  # line label such as ".2" or ".A"
    for raw in re.split(r"[\s,/]+", line):
        token = _clean_b(raw)
        if not token or token.startswith("vac") or token.startswith("'"):
            continue
        if _B_NUMBER.match(token) or _B_MEASURE.match(token):
            if not items or items[-1][0] != "number":
                items.append(("number", None))
        elif _B_WORD.match(token) and any(ch.isalpha() for ch in token):
            items.append(("word", token))
        elif _B_LOGOGRAM.match(token):
            items.append(("logogram", token))
    return items


def linear_b_tokens(items):
    """``(document, word syllables, label)`` from DĀMOS item JSON records."""
    out = []
    for item in items:
        doc = item.get("heading_short") or str(item.get("tablenumber"))
        first_line = True
        for line in (item.get("content") or "").split("\n"):
            parsed = linear_b_line_items(line)
            if not parsed:
                continue
            for word, label in label_line_items(parsed, first_line):
                syllables = parse_syllabic(word)
                if syllables is not None and len(syllables) >= 2:
                    out.append((doc, syllables, label))
            first_line = False
    return out
