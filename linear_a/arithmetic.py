"""Check ``ku-ro`` totals against the entries they close.

``ku-ro`` is read as "total" because its number equals the sum of the numbers above it on many
Hagia Triada tablets. Checking that sum is a test of the corpus parse (lines, numerals, word
divisions), not of any language hypothesis. Fraction signs are ignored: their values are disputed,
so only integer parts are compared.
"""

KU, RO = "\U00010642", "\U00010601"
# Aegean numbers: U+10107..1010F units, then tens, hundreds, thousands, ten thousands.
_BASES = [(0x10107, 1), (0x10110, 10), (0x10119, 100), (0x10122, 1000), (0x1012B, 10000)]


def numeral_value(ch):
    code = ord(ch)
    for start, unit in _BASES:
        if start <= code < start + 9:
            return (code - start + 1) * unit
    return 0


def line_number(line):
    return sum(numeral_value(ch) for ch in line)


def check_totals(corpus):
    """One row per ``ku-ro`` line: document, stated total, integer sum of entries since the last."""
    rows = []
    for doc, record in sorted(corpus.items()):
        text = record.get("unicode_text") or ""
        if KU + RO not in text:
            continue
        running = 0
        for line in text.split("\n"):
            number = line_number(line)
            if KU + RO in line:
                if number:
                    rows.append({"document": doc, "stated": number, "sum": running,
                                 "difference": number - running})
                running = 0
            else:
                running += number
    return rows
