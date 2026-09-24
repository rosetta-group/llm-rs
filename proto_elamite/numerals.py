"""Parse CDLI ATF accounts into entries and totals, and solve two-sign tablets for ratios."""

import collections
import glob
import json
import re
from fractions import Fraction
from pathlib import Path

SOURCES = Path("artifacts/proto-elamite-sources")
_NUMERAL = re.compile(r"(\d+|n|x)\((N\d+[A-Z]?)\)([#?!*]*)")
_LINE = re.compile(r"^\d+[.\w']*\.\s")


def records(folder):
    out = []
    for path in sorted(glob.glob(str(SOURCES / folder / "page-*.json"))):
        out.extend(json.loads(Path(path).read_text()))
    return out


def is_administrative(record):
    return any(g.get("genre", {}).get("genre") == "Administrative" for g in record.get("genres") or [])


def parse(atf):
    """``(entries, total, notations, clean)``; entries and total are unit -> count Counters."""
    surface, entries, total, notations, clean = None, [], None, [], True
    for raw in atf.splitlines():
        line = raw.strip()
        if line.startswith("@"):
            word = line[1:].split()[0] if len(line) > 1 else ""
            if word in ("obverse", "reverse"):
                surface = word
            continue
        if not _LINE.match(line):
            continue
        tokens = _NUMERAL.findall(line)
        if not tokens:
            continue
        if any(k in ("n", "x") or "?" in flags for k, _, flags in tokens) or re.search(r"\[|\]|\.\.\.", line):
            clean = False
            continue
        counts = collections.Counter()
        for k, unit, _ in tokens:
            counts[unit] += int(k)
        notations.append([unit for _, unit, _ in tokens])
        if surface == "obverse":
            entries.append(counts)
        elif surface == "reverse" and total is None:
            total = counts
    return entries, total, notations, clean


def tablets(folder):
    out = []
    for record in records(folder):
        if not is_administrative(record):
            continue
        atf = (record.get("inscription") or {}).get("atf") or ""
        entries, total, notations, clean = parse(atf)
        out.append({"id": record["id"], "designation": record.get("designation"),
                    "entries": entries, "total": total, "notations": notations, "clean": clean,
                    "usable": clean and len(entries) >= 2 and total is not None})
    return out


def precedence(all_tablets):
    """Larger sign of each pair: the one that more often comes first inside a notation."""
    before = collections.Counter()
    for t in all_tablets:
        for notation in t["notations"]:
            for i, a in enumerate(notation):
                for b in notation[i + 1:]:
                    if a != b:
                        before[(a, b)] += 1
    return before


def larger(a, b, before):
    return (a, b) if before[(a, b)] >= before[(b, a)] else (b, a)


def two_sign(t):
    units = set()
    for e in t["entries"] + [t["total"]]:
        units |= set(e)
    return tuple(sorted(units)) if len(units) == 2 else None


def candidate(entries_sum, total, big, small, low=2, high=60):
    """Integer ratio r in [low, high] with S_small + r S_big == T_small + r T_big, else None."""
    denominator = total[big] - entries_sum[big]
    if denominator == 0:
        return None
    r = Fraction(entries_sum[small] - total[small], denominator)
    if r.denominator == 1 and low <= r.numerator <= high:
        return r.numerator
    return None


def entry_sum(t):
    s = collections.Counter()
    for e in t["entries"]:
        s.update(e)
    return s
