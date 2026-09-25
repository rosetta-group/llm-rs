"""Versioned round-three data repair and a small, abstaining word-class model.

Earlier experiments deliberately keep using their frozen loader and predictors.
This module does not edit source data or turn guessed classes into seed labels.
"""

import ast
import collections
import hashlib
import re

import numpy as np
import pandas as pd

from etruscan import classes, corpus, context

CLASSES = classes.CLASSES
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
METHODS = ("context", "context_endings")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def repaired_labels(table=None):
    """Merge usable dictionary rows; reject uncertain entries and conflicting classes.

    POS alone supports NAME/NUM, but not the semantic catch-all OTHER. The source's
    suffix/inference flags remain respected, including suspicious flags such as apa.
    Returned glosses are used ONLY to build conservative evaluation groups.
    """
    if table is None:
        table = pd.read_csv(corpus.LARTH / "ETP_POS.csv", index_col=0)
    entries = collections.defaultdict(list)
    excluded = collections.Counter()
    for index, row in table.iterrows():
        if not isinstance(row.Etruscan, str):
            continue
        if bool(row["Is suffix"]) or bool(row["Is inferred"]):
            excluded["suffix_or_inferred_row"] += 1
            continue
        if re.search(r"[\[\]{}<>?*\-]", row.Etruscan):
            excluded["damaged_headword"] += 1
            continue
        word = corpus.normalise(row.Etruscan)
        if not re.fullmatch(r"[a-z]+", word):
            excluded["nonalphabetic_headword"] += 1
            continue
        pos = row.POS if isinstance(row.POS, str) else ""
        pairs = ast.literal_eval(row.Translations)
        glosses = [g for certain, g in pairs if certain and g.strip() and "?" not in g]
        if "?" in pos or (not glosses and not (classes.NAME_POS.search(pos) or re.search(r"\bnum\b", pos))):
            excluded["uncertain_or_no_semantic_label"] += 1
            continue
        label = classes.etruscan_label(word, {word: (pos, glosses)})
        if label:
            entries[word].append({"row": int(index), "label": label, "glosses": glosses})
    labels, glosses, clashes = {}, {}, {}
    for word, rows in sorted(entries.items()):
        found = {r["label"] for r in rows}
        if len(found) != 1:
            clashes[word] = rows
            continue
        labels[word] = next(iter(found))
        glosses[word] = sorted({g for r in rows for g in r["glosses"]})
    return labels, glosses, {"excluded_rows": dict(excluded), "conflicts": clashes,
                              "accepted_rows": dict(entries)}


def clean_texts():
    """ETP only: keep explicit word divisions; exclude damaged/reconstructed records.

    Do not concatenate surviving fragments across a damaged word. Entire records
    with brackets, restoration marks, damaged tokens, or <2 words are quarantined.
    """
    rows = corpus.load_rows().drop_duplicates(subset=["ID", "Etruscan", "key"])
    kept, dropped = [], []
    for _, row in rows.loc[rows.source == "ETP"].iterrows():
        raw = row.Etruscan if isinstance(row.Etruscan, str) else ""
        # Larth's Latin-letter export uses kh for the same chi rendered ch here.
        words = [t.replace("kh", "ch") for t in corpus.tokens(raw, numerals=True)]
        reason = None
        if row.ID == "Cr 5.2":
            reason = "audited_ambiguous_word_divisions"
        elif re.search(r"[\[\]{}<>?*\-]", raw):
            reason = "damage_or_restoration_marks"
        elif any(not re.fullmatch(r"[a-z]+|N:[ivxlc]+", t) for t in words):
            reason = "nonalphabetic_or_damaged_token"
        elif len(words) < 2:
            reason = "fewer_than_two_words"
        record = {"id": row.ID, "raw": raw, "words": words}
        if reason:
            dropped.append(dict(record, reason=reason))
        else:
            kept.append(("ETP", row.ID, words))
    kept, duplicates = corpus.dedupe_within_id(kept)
    return kept, {"quarantined": dropped, "near_duplicates_removed": duplicates}


def family_groups(types, glosses, suffixes):
    """Conservative connected components, not claimed to be linguistic lemmas.

    Link attested base+suffix variants (base >=3 letters), and words with the same
    certain source gloss. Singularise a fixed few glosses to join clan/clenar etc.
    These keys are never features. Overgrouping makes the test more conservative.
    """
    types = sorted(set(types))
    parents = {t: t for t in types}

    def root(t):
        while parents[t] != t:
            parents[t] = parents[parents[t]]
            t = parents[t]
        return t

    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)

    singular = {"sons": "son", "daughters": "daughter", "years": "year", "gods": "god",
                "children": "child", "grandchildren": "grandchild", "days": "day",
                "months": "month", "freedmen": "freedman", "freedwomen": "freedwoman"}
    gloss_owner = {}
    for t in types:
        if t.startswith("N:"):
            continue
        for suffix in suffixes:
            if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                base = t[:-len(suffix)]
                if base in parents:
                    union(t, base)
        for g in glosses.get(t, []):
            g = re.sub(r"\b[a-z]+\b", lambda m: singular.get(m[0], m[0]), g.lower())
            if g in gloss_owner:
                union(t, gloss_owner[g])
            gloss_owner[g] = t
    return {t: root(t) for t in types}


def folds(labels, groups, count, salt):
    """Deterministic family-disjoint folds balanced by type count, without labels."""
    members = collections.defaultdict(list)
    for t in sorted(labels):
        members[groups[t]].append(t)
    buckets = [set() for _ in range(count)]
    for g in sorted(members, key=lambda g: (-len(members[g]), digest(salt + g))):
        bucket = min(range(count), key=lambda i: (len(buckets[i]), i))
        buckets[bucket].update(members[g])
    return buckets


def features(texts, seeds, groups, endings=False):
    rows = collections.defaultdict(collections.Counter)
    counts = collections.Counter()
    known = collections.Counter()
    for text in texts:
        for i, t in enumerate(text):
            counts[t] += 1
            rows[t][context.position(i, len(text))] += 1
            for side, j in (("L", i - 1), ("R", i + 1)):
                if j < 0 or j >= len(text):
                    label = "BOUNDARY"
                else:
                    neighbour = text[j]
                    # Hide the target's whole family even for training rows.
                    label = seeds.get(neighbour, "UNK") if groups[neighbour] != groups[t] else "UNK"
                    known[t] += label != "UNK"
                rows[t][side + ":" + label] += 1
    for t in rows:
        rows[t] = {f: n / counts[t] for f, n in rows[t].items()}
        if endings and not t.startswith("N:"):
            for n in (1, 2, 3):
                if len(t) > n:  # Do not use the entire word as an identity feature.
                    rows[t][f"S{n}:" + t[-n:]] = 1.0
    return rows, known


def predict(texts, seeds, groups, endings=False):
    """Laplace-smoothed multinomial naive Bayes, empirical priors, no propagation.

    Scores are ranking scores, not calibrated probabilities. Acceptance is chosen
    from inner held-out predictions, separately from outer-fold evaluation.
    """
    rows, known = features(texts, seeds, groups, endings)
    vocabulary = sorted({f for t in seeds if t in rows for f in rows[t]})
    index = {f: i for i, f in enumerate(vocabulary)}
    types = sorted(rows)
    matrix = np.zeros((len(types), len(vocabulary)))
    for i, t in enumerate(types):
        for f, value in rows[t].items():
            if f in index:
                matrix[i, index[f]] = value
    label_counts = collections.Counter(seeds.values())
    likelihood = np.ones((len(CLASSES), len(vocabulary)))
    for i, t in enumerate(types):
        if t in seeds:
            likelihood[CLASSES.index(seeds[t])] += matrix[i]
    likelihood /= likelihood.sum(axis=1, keepdims=True)
    priors = np.array([label_counts[c] + 1 for c in CLASSES], dtype=float)
    priors /= priors.sum()
    logits = matrix @ np.log(likelihood).T + np.log(priors)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    suffix_support = collections.defaultdict(set)
    if endings:
        for t in seeds:
            for n in (2, 3):
                if len(t) > n:
                    suffix_support[t[-n:]].add(groups[t])
    out = {}
    for i, t in enumerate(types):
        best = int(np.argmax(probability[i]))
        support = max((len(suffix_support[t[-n:]]) for n in (2, 3) if len(t) > n), default=0)
        out[t] = {"predicted": CLASSES[best], "score": float(probability[i, best]),
                  "evidence": bool(known[t] > 0 or support >= 3),
                  "known_neighbours": int(known[t]), "suffix_family_support": support}
    return out


def wilson_lower(correct, total, z=1.96):
    if not total:
        return 0.0
    p = correct / total
    return float((p + z*z/(2*total) - z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total))


def choose_threshold(records):
    """Pick maximum calibration coverage subject to fixed accuracy/support bars."""
    eligible = []
    for threshold in THRESHOLDS:
        accepted = [r for r in records if r["evidence"] and r["score"] >= threshold]
        correct = sum(r["predicted"] == r["gold"] for r in accepted)
        if (len(accepted) >= 15 and len({r["family"] for r in accepted}) >= 5
                and correct / len(accepted) >= 0.8 and wilson_lower(correct, len(accepted)) >= 0.65):
            eligible.append((len(accepted), -threshold, threshold))
    return max(eligible)[2] if eligible else None


def metrics(records):
    accepted = [r for r in records if r.get("accepted")]
    correct = sum(r["predicted"] == r["gold"] for r in accepted)
    recalls, per_class = [], {}
    for c in CLASSES:
        items = [r for r in records if r["gold"] == c]
        calls = [r for r in accepted if r["predicted"] == c]
        recall = sum(r["predicted"] == c for r in items) / len(items) if items else None
        if recall is not None:
            recalls.append(recall)
        per_class[c] = {"items": len(items), "calls": len(calls), "full_recall": recall,
                        "accepted_precision": sum(r["gold"] == c for r in calls) / len(calls) if calls else None}
    nonname = [r for r in accepted if r["predicted"] != "NAME"]
    return {"items": len(records), "accepted": len(accepted),
            "coverage": len(accepted) / len(records) if records else 0.0,
            "accepted_precision": correct / len(accepted) if accepted else None,
            "accepted_families": len({r["family"] for r in accepted}),
            "majority_on_accepted": sum(r["majority"] == r["gold"] for r in accepted) / len(accepted) if accepted else None,
            "full_balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "full_accuracy": sum(r["predicted"] == r["gold"] for r in records) / len(records) if records else None,
            "nonname_calls": len(nonname), "nonname_families": len({r["family"] for r in nonname}),
            "nonname_precision": sum(r["predicted"] == r["gold"] for r in nonname) / len(nonname) if nonname else None,
            "per_class": per_class}
