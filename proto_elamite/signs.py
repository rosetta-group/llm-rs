"""Object signs per tablet, arithmetic system labels, and a one-sided Fisher exact test."""

import re
from math import comb

from proto_elamite import numerals as N

_NUMERAL = re.compile(r"^(\d+|n|x)\(N\d+[A-Z]?\)")


def tablet_signs(atf):
    """Base object signs on inscribed lines: variants and damage marks removed, compounds split."""
    out = set()
    for raw in atf.splitlines():
        line = raw.strip()
        if not N._LINE.match(line):
            continue
        body = line.split(".", 1)[1] if "." in line else line
        body = re.sub(r"^[\w']*\.\s", "", body.strip())
        for token in re.split(r"[\s,]+", body):
            if _NUMERAL.match(re.sub(r"[#?!*\[\]<>]", "", token)):
                continue
            token = token.strip("#?!*[]<>()")
            if not token or token.startswith(("$", "#")):
                continue
            for part in re.split(r"[|.+×&]", token):
                part = re.sub(r"[#?!*\[\]<>]", "", part)
                part = re.sub(r"(~[A-Za-z0-9]+|@[A-Za-z]+)+$", "", part)
                if part and not re.fullmatch(r"\d+", part) and part not in ("x", "X", "..."):
                    out.add(part)
    return out


def label(t):
    """'capacity', 'counting' or None, from N01 and N14 only (see the round-three protocol)."""
    if not t["usable"]:
        return None
    sums, total = N.entry_sum(t), t["total"]
    d = {u: sums[u] - total[u] for u in set(sums) | set(total)}
    if any(v for u, v in d.items() if u not in ("N01", "N14")):
        return None
    a, b = d.get("N01", 0), d.get("N14", 0)
    six, ten = a + 6 * b == 0, a + 10 * b == 0
    if six and not ten:
        return "capacity"
    if ten and not six:
        return "counting"
    return None


def fisher_greater(a, b, c, d):
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] with fixed margins (hypergeometric)."""
    row, col, n = a + b, a + c, a + b + c + d
    denominator = comb(n, col)
    return sum(comb(row, x) * comb(n - row, col - x)
               for x in range(a, min(row, col) + 1)) / denominator
