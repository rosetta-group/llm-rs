"""Proto-Elamite round five (exploratory): constant quantity ratios between consecutive entries.

    python -m experiments.proto_elamite_round_five

See experiments/proto-elamite-ratios/PROTOCOL.md. Writes results.json and refuses to overwrite it.
"""

import collections
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from linear_a.controls import rng_for
from proto_elamite import numerals as N

OUT = Path("experiments/proto-elamite-ratios")
NUMERAL = re.compile(r"(\d+)\((N\d+[A-Z]?)\)")
MIN_TABLETS, NULL_RUNS, MIN_SUPPORT = 5, 2000, 4


def entries(atf):
    out = []
    for raw in atf.splitlines():
        line = raw.strip()
        if not N._LINE.match(line):
            continue
        body = line.split(".", 1)[1]
        if "," not in body:
            continue
        signs, numbers = body.split(",", 1)
        head = [t for t in signs.split() if t.startswith(("M", "|"))]
        if not head:
            continue
        head = re.sub(r"[#?!]", "", head[0]).split("~")[0]
        tokens = NUMERAL.findall(numbers)
        quantity = None
        if tokens and not re.search(r"\[|\]|x\(|n\(|\?", numbers) and all(u in ("N01", "N14") for _, u in tokens):
            n01 = sum(int(k) for k, u in tokens if u == "N01")
            n14 = sum(int(k) for k, u in tokens if u == "N14")
            quantity = (n01 + 10 * n14, n01 + 6 * n14)
        out.append((head, quantity))
    return out


def ratios(qa, qb):
    return {Fraction(b, a) for a, b in zip(qa, qb) if a}


def statistic(sets):
    support = collections.Counter(r for s in sets for r in s)
    return (max(support.values()), support.most_common(3)) if support else (0, [])


def main():
    path = OUT / "results.json"
    if path.exists():
        sys.exit("results.json exists; refusing to overwrite")
    occurrences = collections.defaultdict(dict)  # pair -> tablet -> (qa, qb) first occurrence
    tablets_with = collections.defaultdict(set)
    for record in N.records("proto-elamite"):
        if not N.is_administrative(record):
            continue
        e = entries((record.get("inscription") or {}).get("atf") or "")
        for (a, qa), (b, qb) in zip(e, e[1:]):
            if a == b:
                continue
            tablets_with[(a, b)].add(record["designation"])
            if qa and qb and record["designation"] not in occurrences[(a, b)]:
                occurrences[(a, b)][record["designation"]] = (qa, qb)
    pairs = sorted(p for p, t in tablets_with.items() if len(t) >= MIN_TABLETS)
    alpha = 0.05 / len(pairs)
    rows = []
    for pair in pairs:
        items = list(occurrences[pair].items())
        if len(items) < 2:
            rows.append({"pair": " -> ".join(pair), "tablets": len(tablets_with[pair]),
                         "with_quantities": len(items), "p": None, "survives": False})
            continue
        qa = [q[0] for _, q in items]
        qb = [q[1] for _, q in items]
        observed, top = statistic([ratios(a, b) for a, b in zip(qa, qb)])
        generator = rng_for("pe-ratios", *pair)
        n = len(items)
        null = []
        for _ in range(NULL_RUNS):
            perm = generator.permutation(n)
            while np.any(perm == np.arange(n)):
                perm = generator.permutation(n)
            null.append(statistic([ratios(qa[i], qb[perm[i]]) for i in range(n)])[0])
        p = float((np.sum(np.array(null) >= observed) + 1) / (NULL_RUNS + 1))
        top_r = top[0][0] if top else None
        rows.append({"pair": " -> ".join(pair), "tablets": len(tablets_with[pair]),
                     "with_quantities": n, "support": observed,
                     "top_ratios": [[str(r), c] for r, c in top],
                     "null_mean": float(np.mean(null)), "p": p,
                     "survives": p < alpha and observed >= MIN_SUPPORT,
                     "examples": [d for d, (a, b) in items if top_r in ratios(a, b)][:6]})
        print(json.dumps(rows[-1]), flush=True)
    result = {"pairs_tested": len(pairs), "alpha": alpha, "rows": rows,
              "survivors": [r["pair"] for r in rows if r["survives"]]}
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print("survivors", result["survivors"])


if __name__ == "__main__":
    main()
