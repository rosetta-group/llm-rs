"""Linear A round four: seven targeted probes.

    python -m experiments.linear_a_probes [probe numbers]

See experiments/linear-a-probes/PROTOCOL.md. Each probe writes results-<n>.json and refuses to
overwrite it.
"""

import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

from experiments.linear_a_context import DAMOS
from linear_a import contexts, corpus, names, probes
from linear_a.lexicons import read_entries, _romanizations
from linear_a.matching import Matcher
from linear_a.spelling import parse_syllabic, render, spell

OUT = Path("experiments/linear-a-probes")
LISTS = json.loads((OUT / "lists.json").read_text())
ALPHA = 0.05 / 7
NULL_RUNS = 200


def rng(*keys):
    from linear_a.controls import rng_for
    return rng_for("probes", *keys)


# ---- data -----------------------------------------------------------------------------------

_cache = {}


def linear_a():
    if "la" not in _cache:
        data = corpus.load()
        tokens = contexts.linear_a_tokens(data)
        labels = contexts.type_labels((w, l) for _, w, l in tokens)
        docs = collections.defaultdict(set)
        for doc, w, _ in tokens:
            docs[w].add(doc)
        _cache["la"] = {"corpus": data, "tokens": tokens, "labels": labels, "docs": docs}
    return _cache["la"]


def linear_b():
    if "lb" not in _cache:
        rows = []
        for path in sorted(glob.glob(str(DAMOS / "items" / "*.json"))):
            item = json.loads(Path(path).read_text())["item"]
            if not item:
                continue
            series = (item.get("series") or "?") + (item.get("subseries") or "")
            for doc, w, label in contexts.linear_b_tokens([item]):
                rows.append((doc, w, label, series, item.get("ishort") or "?"))
        labels = contexts.type_labels((w, l) for _, w, l, _, _ in rows)
        series = collections.defaultdict(collections.Counter)
        sites = collections.defaultdict(collections.Counter)
        for _, w, _, s, site in rows:
            series[w][s] += 1
            sites[w][site] += 1
        _cache["lb"] = {"rows": rows, "labels": labels, "series": series, "sites": sites}
    return _cache["lb"]


def write(n, result):
    path = OUT / f"results-{n}.json"
    if path.exists():
        sys.exit(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")


# ---- probe 1 --------------------------------------------------------------------------------


def probe_1():
    la_words = sorted(linear_a()["labels"])
    lb_words = sorted(linear_b()["labels"])
    rows, la_hits = [], 0
    for entry in LISTS["kom_el_hetan"]["names"]:
        sk = entry["skeleton"]
        lb = [w for w in lb_words if probes.skeleton_matches(sk, probes.word_skeleton(w))]
        la = [w for w in la_words if probes.skeleton_matches(sk, probes.word_skeleton(w))]
        expected = parse_syllabic(entry["linear_b"]) if entry["linear_b"] else None
        rows.append({**entry, "linear_b_found": expected in lb if expected else None,
                     "linear_b_matches": len(lb), "linear_a_matches": [render(w) for w in la]})
        la_hits += bool(la)
    control = [r["linear_b_found"] for r in rows if r["linear_b_found"] is not None]
    null = []
    for corpus_ in probes.null_corpora(la_words, rng(1), NULL_RUNS):
        null.append(sum(any(probes.skeleton_matches(e["skeleton"], probes.word_skeleton(w))
                            for w in corpus_) for e in LISTS["kom_el_hetan"]["names"]))
    result = {"names": rows, "control_found": sum(control), "control_total": len(control),
              "control_passed": sum(control) >= 5, "linear_a_names_matched": la_hits,
              "null_mean": float(np.mean(null)), "p": probes.upper_p(la_hits, null)}
    write(1, result)
    return {k: result[k] for k in ("control_found", "control_total", "linear_a_names_matched",
                                   "null_mean", "p")}


# ---- probe 2 --------------------------------------------------------------------------------


def probe_2():
    la_words = sorted(linear_a()["labels"])
    variants = {}
    for group in LISTS["keftiu_names"]["names"]:
        for text in group:
            s = spell(text)
            if s:
                variants.setdefault(s, []).append(group[0])
    matcher = Matcher({"keftiu": {k: v for k, v in variants.items()}})

    def hits(words):
        near = matcher.nearest(words, "keftiu")
        found = collections.defaultdict(list)
        forms = matcher.forms["keftiu"]
        for w, (d, i) in zip(words, near):
            if d <= 0.25:
                found[variants[forms[i]][0]].append((render(w), render(forms[i]), round(d, 2)))
        for w in words:
            for v, owners in variants.items():
                if len(w) >= 2 and len(v) >= 2 and w[:2] == v[:2]:
                    found[owners[0]].append((render(w), render(v), "stem"))
        return found

    observed = hits(la_words)
    null = [len(hits(c)) for c in probes.null_corpora(la_words, rng(2), NULL_RUNS)]
    result = {"spelled": {render(k): v for k, v in variants.items()},
              "hits": {k: sorted({tuple(map(str, x)) for x in v}) for k, v in observed.items()},
              "names_with_hits": len(observed), "null_mean": float(np.mean(null)),
              "p": probes.upper_p(len(observed), null)}
    write(2, result)
    return {k: result[k] for k in ("names_with_hits", "null_mean", "p")}


# ---- probe 3 --------------------------------------------------------------------------------


def _laman_gods():
    lexicon = {}
    with open(names.NAMES_DIR / "laman_names.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Type"] == "deity" and row["Writing Type"] == "phonetic":
                names._add(lexicon, row["Name"], row["Name"])
    return lexicon


def _qpn_gods(files):
    lexicon = {}
    for name in files:
        for entry in json.loads((names.NAMES_DIR / name).read_text(encoding="utf-8"))["entries"]:
            if entry.get("pos") == "DN":
                for text in [entry.get("cf", "")] + [n.get("n", "") for n in entry.get("norms", [])]:
                    if re.search(r"[aeiouāēīū]", text.lower()):
                        names._add(lexicon, text, entry.get("cf", text))
    return lexicon


def _greek_gods():
    keys = LISTS["greek_god_keys"]
    lexicon = {}
    for entry in read_entries("AncientGreek"):
        if entry.get("pos") != "name":
            continue
        gloss = " ".join(" ".join(s.get("glosses", [])) for s in entry.get("senses", [])).lower()
        if any(re.search(rf"\b{k}\b", gloss) for k in keys):
            for roman in _romanizations(entry):
                names._add(lexicon, roman, entry.get("word", roman))
    return lexicon


def _stratified_permutation(lengths, labels, generator):
    labels = labels.copy()
    for n in np.unique(lengths):
        idx = np.flatnonzero(lengths == n)
        labels[idx] = generator.permutation(labels[idx])
    return labels


def probe_3():
    la = linear_a()
    doc_types = collections.defaultdict(set)
    for d, w, _ in la["tokens"]:
        doc_types[w].add(la["corpus"][d].get("type"))
    libation = sorted(w for w, t in doc_types.items() if t & set(LISTS["libation_types"]))
    tablet = sorted(w for w, t in doc_types.items() if "tablet" in t and la["labels"][w] == "entry"
                    and w not in set(libation))
    lexicons = {"Anatolian gods": _laman_gods(),
                "Levant gods": _qpn_gods(names.QPN_FILES["Levant"]),
                "Babylonian gods": _qpn_gods(names.QPN_FILES["Babylonia"]),
                "Greek gods": _greek_gods()}
    matcher = Matcher({k: {f: sorted(v) for f, v in lx.items() if len(f) >= 2}
                       for k, lx in lexicons.items()})
    words = libation + tablet
    is_lib = np.array([1] * len(libation) + [0] * len(tablet))
    lengths = np.array([len(w) for w in words])
    generator = rng(3)
    out = {}
    for name in matcher.names:
        for theta in (0.0, 0.2):
            hit = (matcher.distances(words, name) <= theta).astype(float)
            observed = hit[is_lib == 1].mean() - hit[is_lib == 0].mean()
            null = []
            for _ in range(2000):
                perm = _stratified_permutation(lengths, is_lib, generator)
                null.append(hit[perm == 1].mean() - hit[perm == 0].mean())
            matched = [(render(w), render(matcher.best_form(w, name)[1]),
                        matcher.best_form(w, name)[2][:2])
                       for w, h, l in zip(words, hit, is_lib) if h and l]
            out[f"{name} theta={theta}"] = {
                "libation_rate": float(hit[is_lib == 1].mean()),
                "tablet_rate": float(hit[is_lib == 0].mean()),
                "difference": float(observed), "p": probes.upper_p(observed, null),
                "libation_matches": matched}
    # Control: Mycenaean theonyms against the Greek god list, versus their own bigram null.
    keys = LISTS["mycenaean_theonym_keys"]
    myc = []
    for entry in read_entries("MycenaeanGreek"):
        gloss = " ".join(" ".join(s.get("glosses", [])) for s in entry.get("senses", []))
        if entry.get("pos") in ("name", "noun") and any(re.search(rf"\b{k}\b", gloss) for k in keys):
            for f in entry.get("forms", []):
                if "romanization" in f.get("tags", []) and "-" in f.get("form", ""):
                    s = parse_syllabic(f["form"])
                    if s and len(s) >= 2:
                        myc.append(s)
                    break
    myc = sorted(set(myc))
    real = int((matcher.distances(myc, "Greek gods") <= 0.2).sum())
    null = [int((matcher.distances(c, "Greek gods") <= 0.2).sum())
            for c in probes.null_corpora(myc, rng(3, "control"), NULL_RUNS)]
    result = {"libation_words": len(libation), "tablet_entry_words": len(tablet),
              "lexicon_sizes": {k: len(v) for k, v in matcher.forms.items()}, "tests": out,
              "control": {"mycenaean_theonyms": [render(w) for w in myc], "matches": real,
                          "null_mean": float(np.mean(null)), "p": probes.upper_p(real, null)}}
    write(3, result)
    return {"libation": len(libation), "tablet": len(tablet),
            "p": {k: round(v["p"], 4) for k, v in out.items()},
            "control": {k: result["control"][k] for k in ("matches", "null_mean", "p")}}


# ---- probe 4 --------------------------------------------------------------------------------


def _stems():
    stems = {}
    for item in LISTS["trade_words"]["items"]:
        forms = [parse_syllabic(item["linear_b"])] if item["linear_b"] else []
        forms += [spell(s) for s in item["semitic"] + item.get("other", [])]
        for f in forms:
            s = probes.stem(f)
            if s:
                stems.setdefault(s, set()).add(item["concept"])
    return stems


def _stem_hits(words, stems):
    return {s: [w for w in words if probes.starts_with(w, s)] for s in stems}


def _gordon():
    la = linear_a()
    rows = []
    for item in LISTS["gordon_proposals"]["items"]:
        target = parse_syllabic(item["linear_a"])
        contexts_ = []
        for doc, record in sorted(la["corpus"].items()):
            signs = record.get("signs") or []
            readings = [parse_syllabic(s["reading"]) if s.get("reading") and s["role"] == "syllabogram"
                        else None for s in signs]
            flat = [r[0] if r else None for r in readings]
            n = len(target)
            for i in range(len(flat) - n + 1):
                if flat[i:i + n] == list(target):
                    after = [s.get("reading") or s["type"] for s in signs[i + n:i + n + 3]]
                    roles = [s["role"] for s in signs[i + n:i + n + 3]]
                    contexts_.append({"document": doc, "next": after, "roles": roles})
        rows.append({**item, "occurrences": contexts_,
                     "label": la["labels"].get(target)})
    return rows


def probe_4():
    stems = _stems()
    la_words = sorted(w for w in linear_a()["labels"] if len(w) >= 2)
    observed = _stem_hits(la_words, stems)
    found = sum(bool(v) for v in observed.values())
    null = [sum(any(probes.starts_with(w, s) for w in c) for s in stems)
            for c in probes.null_corpora(la_words, rng(4), NULL_RUNS)]
    # Control: Linear B stems in a Linear-A-sized DĀMOS sample.
    lb_words = sorted(w for w in linear_b()["labels"] if len(w) >= 2)
    generator = rng(4, "control")
    lb_stems = {s: c for s, c in stems.items()
                if any(item["linear_b"] and probes.stem(parse_syllabic(item["linear_b"])) == s
                       for item in LISTS["trade_words"]["items"])}
    control = []
    for d in range(20):
        sample = [lb_words[i] for i in generator.choice(len(lb_words), len(la_words), replace=False)]
        real = sum(any(probes.starts_with(w, s) for w in sample) for s in lb_stems)
        nulls = [sum(any(probes.starts_with(w, s) for w in c) for s in lb_stems)
                 for c in probes.null_corpora(sample, generator, 50)]
        control.append(probes.upper_p(real, nulls))
    result = {
        "stems": {render(s): sorted(c) for s, c in stems.items()},
        "linear_a_hits": {render(s): [render(w) for w in v] for s, v in observed.items() if v},
        "stems_found": found, "stems_total": len(stems), "null_mean": float(np.mean(null)),
        "p": probes.upper_p(found, null),
        "control_p_median": float(np.median(control)),
        "control_share_below_alpha": float(np.mean(np.array(control) < ALPHA)),
        "gordon": _gordon(),
    }
    write(4, result)
    return {k: result[k] for k in ("stems_found", "stems_total", "null_mean", "p",
                                   "control_p_median", "control_share_below_alpha")}


# ---- probe 5 --------------------------------------------------------------------------------


def _running_greek(limit=200000):
    out = []
    for entry in read_entries("AncientGreek"):
        for f in entry.get("forms", []):
            if f.get("roman"):
                s = spell(f["roman"])
                if s and len(s) >= 2:
                    out.append(s)
        if len(out) >= limit:
            break
    return sorted(set(out))


def _running_akkadian():
    out = set()
    for name in ["aemw/amarna/gloss-akk-x-mbperi.json", "aemw/ugarit/gloss-akk-x-mbperi.json",
                 "aemw/ugarit/gloss-akk-x-midass.json", "rimanum/gloss-akk-x-oldbab.json"]:
        for entry in json.loads((names.NAMES_DIR / name).read_text(encoding="utf-8"))["entries"]:
            for n in entry.get("norms", []):
                s = spell(n.get("n", ""))
                if s and len(s) >= 2:
                    out.add(s)
    return sorted(out)


def _samples(words, size, key, count=20):
    generator = rng(5, key)
    words = list(words)
    return [probes.profile([words[i] for i in generator.choice(len(words), size, replace=False)])
            for _ in range(count)]


def probe_5():
    la = linear_a()["labels"]
    lb = linear_b()["labels"]
    la_all = [w for w in la if len(w) >= 2]
    la_entry = [w for w, l in la.items() if l == "entry"]
    lb_all = [w for w in lb if len(w) >= 2]
    lb_entry = [w for w, l in lb.items() if l == "entry"]
    running = {"Greek running words": _running_greek(), "Akkadian running words": _running_akkadian()}
    name_lists = {k: list(v) for k, v in names.build_name_lexicons(2).items()}
    n_all, n_entry = 500, 300
    running_ref = {k: _samples(v, n_all, k) for k, v in running.items()}
    names_ref = {k: _samples(v, n_entry, k) for k, v in name_lists.items()}
    lb_near, _ = probes.nearest_profiles(_samples(lb_all, n_all, "lb"), running_ref)
    lbe_near, _ = probes.nearest_profiles(_samples(lb_entry, n_entry, "lbe"), names_ref)
    la_near, centres_running = probes.nearest_profiles(_samples(la_all, n_all, "la"), running_ref)
    lae_near, centres_names = probes.nearest_profiles(_samples(la_entry, n_entry, "lae"), names_ref)
    result = {
        "features": probes.FEATURES,
        "control_lb_running": dict(collections.Counter(lb_near)),
        "control_lb_entry_names": dict(collections.Counter(lbe_near)),
        "control_passed": lb_near.count("Greek running words") >= 18
        and lbe_near.count("Greek") >= 18,
        "linear_a_running": dict(collections.Counter(la_near)),
        "linear_a_entry_names": dict(collections.Counter(lae_near)),
        "profiles": {
            "Linear A all": np.mean(_samples(la_all, n_all, "la"), axis=0).tolist(),
            "Linear A entry": np.mean(_samples(la_entry, n_entry, "lae"), axis=0).tolist(),
            "Linear B all": np.mean(_samples(lb_all, n_all, "lb"), axis=0).tolist(),
            "Linear B entry": np.mean(_samples(lb_entry, n_entry, "lbe"), axis=0).tolist(),
            **centres_running, **centres_names},
    }
    write(5, result)
    return {k: result[k] for k in ("control_lb_running", "control_lb_entry_names",
                                   "control_passed", "linear_a_running", "linear_a_entry_names")}


# ---- probe 6 --------------------------------------------------------------------------------


def _by_length(words):
    out = collections.defaultdict(list)
    for w in words:
        out[len(w)].append(w)
    return out


PRE_NAMED = {"re->ro", "ru->ro"}


def _pre_named(counts):
    return sum(n for (pos, a, b), n in counts.items()
               if pos == "final" and f"{a[0]}{a[1]}->{b[0]}{b[1]}" in PRE_NAMED)


def probe_6():
    la_words = sorted(linear_a()["labels"])
    lb_by_length = _by_length(linear_b()["labels"])
    halves = {h: [w for w in la_words if probes.half(w) == h] for h in (0, 1)}
    real0, _ = probes.one_substitutions(halves[0], lb_by_length)
    nulls0 = [probes.one_substitutions(c, lb_by_length)[0]
              for c in probes.null_corpora(halves[0], rng(6, 0), 100)]
    discovered = []
    for key, n in real0.items():
        arr = np.array([c[key] for c in nulls0], dtype=float)
        z = (n - arr.mean()) / max(arr.std(), 1.0)
        if n >= 3 and z >= 3:
            discovered.append((key, n, float(arr.mean()), float(z)))
    keys = [k for k, *_ in discovered]
    real1, pairs1 = probes.one_substitutions(halves[1], lb_by_length)
    nulls1 = [probes.one_substitutions(c, lb_by_length)[0]
              for c in probes.null_corpora(halves[1], rng(6, 1), 100)]
    pooled = sum(real1[k] for k in keys)
    pooled_null = [sum(c[k] for k in keys) for c in nulls1]
    per_rule = []
    for k in keys:
        arr = [c[k] for c in nulls1]
        per_rule.append({"rule": probes.rule_key(k), "confirmation_pairs": real1[k],
                         "confirmation_null": float(np.mean(arr)), "p": probes.upper_p(real1[k], arr)})
    # Pre-named rule on all words.
    real_all, pairs_all = probes.one_substitutions(la_words, lb_by_length)
    nulls_all = [probes.one_substitutions(c, lb_by_length)[0]
                 for c in probes.null_corpora(la_words, rng(6, "all"), 100)]
    named = _pre_named(real_all)
    named_null = [_pre_named(c) for c in nulls_all]
    result = {
        "discovery": [{"rule": probes.rule_key(k), "pairs": n, "null": m, "z": z}
                      for k, n, m, z in discovered],
        "confirmation_pooled": pooled, "confirmation_null_mean": float(np.mean(pooled_null)),
        "confirmation_p": probes.upper_p(pooled, pooled_null), "per_rule": per_rule,
        "confirmed_rules": [r["rule"] for r in per_rule if r["p"] < ALPHA],
        "pre_named": {"pairs": named, "null_mean": float(np.mean(named_null)),
                      "p": probes.upper_p(named, named_null), "note": "seen before the protocol"},
        "pre_named_pairs": [(render(w), render(v)) for w, v, (pos, a, b) in pairs_all
                            if pos == "final" and f"{a[0]}{a[1]}->{b[0]}{b[1]}" in PRE_NAMED],
    }
    write(6, result)
    return {k: result[k] for k in ("confirmation_pooled", "confirmation_null_mean",
                                   "confirmation_p", "confirmed_rules", "pre_named")}


# ---- probe 7 --------------------------------------------------------------------------------


def probe_7():
    la = linear_a()
    lb = linear_b()
    rules = set(PRE_NAMED)
    confirmed = OUT / "results-6.json"
    if confirmed.exists():
        rules |= {r.split(":")[1] for r in json.loads(confirmed.read_text())["confirmed_rules"]
                  if r.startswith("final:")}
    lb_set = set(lb["labels"])
    carried = {}
    for w in la["labels"]:
        if len(w) < 2:
            continue
        if w in lb_set:
            carried[w] = w
            continue
        for rule in rules:
            a, b = rule.split("->")
            if render(w[-1:]) == a:
                v = w[:-1] + (parse_syllabic(b)[0],)
                if v in lb_set:
                    carried[w] = v
                    break
    words = sorted(carried)
    la_entry = np.array([la["labels"][w] == "entry" for w in words])
    lb_entry = np.array([lb["labels"][carried[w]] == "entry" for w in words])
    observed = int((la_entry == lb_entry).sum())
    generator = rng(7)
    null = [int((generator.permutation(la_entry) == lb_entry).sum()) for _ in range(10000)]
    table = [{"linear_a": render(w), "linear_b": render(carried[w]),
              "la_label": la["labels"][w], "la_documents": sorted(la["docs"][w])[:6],
              "lb_label": lb["labels"][carried[w]],
              "lb_series": dict(lb["series"][carried[w]].most_common(4)),
              "lb_sites": dict(lb["sites"][carried[w]].most_common(3))} for w in words]
    result = {"rules_used": sorted(rules), "carried": len(words), "agreement": observed,
              "null_mean": float(np.mean(null)), "p": probes.upper_p(observed, null),
              "long_words": [r for r in table if len(r["linear_a"].split("-")) >= 3],
              "table": table}
    write(7, result)
    return {k: result[k] for k in ("rules_used", "carried", "agreement", "null_mean", "p")}


PROBES = {1: probe_1, 2: probe_2, 3: probe_3, 4: probe_4, 5: probe_5, 6: probe_6, 7: probe_7}


def main():
    chosen = [int(a) for a in sys.argv[1:]] or sorted(PROBES)
    for n in chosen:
        print(n, json.dumps(PROBES[n](), ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
