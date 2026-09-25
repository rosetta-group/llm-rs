"""Whole-inscription template alignment using anonymised, audited name anchors.

Only public inscription structure enters the predictor. Test translations,
relation annotations, target words, and non-name dictionary meanings do not.
"""

import collections
import hashlib
import math
import re

import pandas as pd

from etruscan import classes, corpus

K = 3
MIN_SUPPORT = 0.6
MAX_DISTANCE = 0.55
DEICTICS = {"mi", "mini", "ecn", "cn", "itun"}
ARITY = {"CHILD_OF": 2, "SPOUSE_OF": 2, "OWNED_BY": 2, "TRANSFER": 3, "MADE": 2}


def name_morphology(table=None):
    """Only name POS/case/gender; merge conflicts instead of choosing first row."""
    if table is None:
        table = pd.read_csv(corpus.LARTH / "ETP_POS.csv", index_col=0)
    result = collections.defaultdict(list)
    for _, row in table.iterrows():
        if not isinstance(row.Etruscan, str) or not isinstance(row.POS, str):
            continue
        if bool(row["Is suffix"]) or bool(row["Is inferred"]):
            continue
        pos = row.POS
        if not (classes.NAME_POS.search(pos) or re.search(r"\btheo\b", pos)):
            continue
        cases = []
        if re.search(r"\b(nom|acc)\b", pos):
            cases.append("DIRECT")
        if re.search(r"\bgen\b", pos):
            cases.append("GEN")
        if re.search(r"\bpert\b", pos):
            cases.append("PERT")
        gender = "M" if "masc" in pos else "F" if "fem" in pos else "?"
        result[corpus.normalise(row.Etruscan)].append(
            {"case": cases[0] if len(cases) == 1 else "?", "gender": gender,
             "personal": bool(re.search(r"\bprae\b", pos))})
    return dict(result)


def public_view(record, morphology):
    """One-way information barrier: no gloss, target flag, spelling, or gold survives.

    The deliberately favourable input is a manual inventory of name spans and
    person/deity types. Morphological attributes come only from the pinned lexicon.
    """
    owners = {}
    attributes = {}
    for index, entity in enumerate(record["entities"]):
        eid = f"e{index}"
        rows = [r for i in entity["tokens"] for r in morphology.get(record["tokens"][i], [])]
        personal = [r for r in rows if r["personal"]]
        chosen = personal or rows

        def consensus(key):
            values = {r[key] for r in chosen if r[key] != "?"}
            return next(iter(values)) if len(values) == 1 else "?"

        attributes[eid] = {"kind": entity["kind"], "case": consensus("case"), "gender": consensus("gender")}
        for i in entity["tokens"]:
            if i in owners:
                raise ValueError("Overlapping entity spans")
            owners[i] = eid
    sequence = []
    for i, word in enumerate(record["tokens"]):
        if i in owners:
            eid = owners[i]
            if sequence and sequence[-1].get("entity") == eid:
                continue
            sequence.append(dict(attributes[eid], entity=eid))
        elif word in DEICTICS:
            sequence.append({"kind": "DEICTIC"})
        else:
            sequence.append({"kind": "W", "lexeme": hashlib.sha256(word[:3].encode()).hexdigest()[:16]})
    probes = {hashlib.sha256(record["tokens"][i][:3].encode()).hexdigest()[:16]
              for i in record.get("probe_tokens", [])}
    return {"id": record["id"], "sequence": sequence, "entities": attributes,
            "probes": sorted(probes)}


def public_fingerprint(view):
    """Record identical abstract templates; distinct monuments may share one."""
    return tuple(tuple(sorted((k,v) for k,v in token.items() if k != "lexeme")) for token in view["sequence"])


def substitution(a, b, morphology=True):
    ea, eb = "entity" in a, "entity" in b
    if ea != eb:
        return 2.0
    if not ea:
        return 0.0 if a["kind"] == b["kind"] else 0.8
    cost = 0.0 if a["kind"] == b["kind"] else 0.5
    if morphology:
        for key, penalty in (("case", 0.45), ("gender", 0.15)):
            if a[key] != "?" and b[key] != "?" and a[key] != b[key]:
                cost += penalty
    return cost


def gap(token):
    return 1.0 if "entity" in token else 0.5


def align(source, target, morphology=True):
    """Weighted sequence alignment and a consistent, injective name correspondence."""
    a, b = source["sequence"], target["sequence"]
    costs = [[0.0] * (len(b)+1) for _ in range(len(a)+1)]
    back = {}
    for i in range(1, len(a)+1):
        costs[i][0] = costs[i-1][0] + gap(a[i-1])
        back[i, 0] = "delete"
    for j in range(1, len(b)+1):
        costs[0][j] = costs[0][j-1] + gap(b[j-1])
        back[0, j] = "insert"
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            options = [(costs[i-1][j-1] + substitution(a[i-1], b[j-1], morphology), 0, "match"),
                       (costs[i-1][j] + gap(a[i-1]), 1, "delete"),
                       (costs[i][j-1] + gap(b[j-1]), 2, "insert")]
            cost, _, action = min(options)
            costs[i][j], back[i, j] = cost, action
    mapping, pairs, i, j = {}, [], len(a), len(b)
    while i or j:
        action = back[i, j]
        if action == "match":
            if "entity" in a[i-1] and "entity" in b[j-1]:
                pairs.append((a[i-1]["entity"], b[j-1]["entity"]))
            i, j = i-1, j-1
        elif action == "delete":
            i -= 1
        else:
            j -= 1
    reverse, conflicts = {}, set()
    for left, right in pairs:
        if left in mapping and mapping[left] != right:
            conflicts.add(left)
        if right in reverse and reverse[right] != left:
            conflicts.update((left, reverse[right]))
        mapping[left], reverse[right] = right, left
    mapping = {k: v for k, v in mapping.items() if k not in conflicts}
    normalizer = max(sum(map(gap, a)), sum(map(gap, b)), 1.0)
    return costs[len(a)][len(b)] / normalizer, mapping


def frame_key(frame):
    relation, *args = frame
    if relation not in ARITY or len(args) != ARITY[relation]:
        raise ValueError(f"Invalid frame {frame}")
    if relation == "SPOUSE_OF":
        args = sorted(args)
    return tuple([relation] + args)


def transfer(frame, mapping):
    relation, *args = frame
    mapped = [arg if arg in ("OBJECT", "UNSPECIFIED") else mapping.get(arg) for arg in args]
    if any(x is None for x in mapped):
        return None
    if relation in ("CHILD_OF", "SPOUSE_OF") and mapped[0] == mapped[1]:
        return None
    if relation == "TRANSFER" and mapped[0] == mapped[2] and mapped[0] != "UNSPECIFIED":
        return None
    return frame_key([relation] + mapped)


def predict(train, query, morphology=True):
    """Vote over role graphs transferred through the three closest alignments."""
    neighbours = []
    for example in train:
        if example["public"]["id"] == query["id"]:
            raise ValueError("A monument appears in both train and test")
        distance, mapping = align(example["public"], query, morphology)
        if distance <= MAX_DISTANCE:
            neighbours.append((distance, example["public"]["id"], mapping, example["frames"]))
    neighbours = sorted(neighbours, key=lambda x: (x[0], x[1]))[:K]
    votes, denominator = collections.Counter(), 0.0
    evidence = []
    for distance, tid, mapping, frames in neighbours:
        weight = 1.0 / (0.1 + distance)
        denominator += weight
        copied = {f for frame in frames if (f := transfer(frame, mapping)) is not None}
        for f in copied:
            votes[f] += weight
        evidence.append({"source": tid, "distance": distance, "mapping": mapping,
                         "transferred": [list(f) for f in sorted(copied)]})
    ranked = [{"frame": list(f), "support": n / denominator} for f, n in sorted(votes.items(), key=lambda p: (-p[1], p[0]))]
    accepted = [r["frame"] for r in ranked if r["support"] >= MIN_SUPPORT]
    return {"accepted": accepted, "ranked": ranked, "neighbours": evidence}


def count_baseline(train, query):
    """Most common whole role graph for the same entity count; no text or morphology."""
    count = len(query["entities"])
    graphs = collections.Counter(tuple(sorted(frame_key(f) for f in ex["frames"]))
                                 for ex in train if len(ex["public"]["entities"]) == count)
    if not graphs:
        return {"accepted": []}
    graph = min(graphs, key=lambda g: (-graphs[g], g))
    return {"accepted": [list(f) for f in graph]}


def predict_joint(train, queries):
    """A small constraint tournament over recurring anonymous word stems.

    For each possible role graph, retain its best whole-text alignment. Pool
    relation evidence across occurrences of the same queried anonymous stem.
    Select endpoints separately, abstaining if competing role assignments tie.
    Scores are heuristic support, not calibrated probabilities. Spellings and
    translations of queried words never enter this function.
    """
    candidates, families = {}, collections.defaultdict(list)
    views = {q["id"]: q for q in queries}
    for query in queries:
        if len(query.get("probes", [])) != 1:
            raise ValueError("Pilot requires one queried stem per inscription")
        families[query["probes"][0]].append(query["id"])
        scores = {}
        for example in train:
            if example["public"]["id"] == query["id"]:
                raise ValueError("Train/test monument overlap")
            distance, mapping = align(example["public"], query)
            if distance > MAX_DISTANCE:
                continue
            for frame in example["frames"]:
                copied = transfer(frame, mapping)
                if copied is None:
                    continue
                value = math.exp(-6.0 * distance)
                existing = scores.get(copied)
                if existing is None or value > existing["merit"]:
                    scores[copied] = {"merit":value, "source":example["public"]["id"],
                                      "distance":distance, "mapping":mapping}
        candidates[query["id"]] = scores
    family_results, predictions = {}, {}
    relations = sorted(ARITY)
    for family, ids in families.items():
        logs = {relation:0.0 for relation in relations}
        evidence, seen_templates, pooled_ids = {}, set(), []
        for tid in ids:
            evidence[tid] = {relation:max((v["merit"] for f,v in candidates[tid].items() if f[0] == relation),default=0.0)
                             for relation in relations}
            fingerprint = public_fingerprint(views[tid])
            if fingerprint in seen_templates:
                continue
            seen_templates.add(fingerprint)
            pooled_ids.append(tid)
            for relation in relations:
                # Fixed small floor: a noisy record cannot give an infinite veto.
                logs[relation] += math.log(0.02 + evidence[tid][relation])
        top = max(logs.values())
        weights = {r:math.exp(v-top) for r,v in logs.items()}
        total = sum(weights.values())
        support = {r:v/total for r,v in weights.items()}
        winner = min(relations,key=lambda r:(-support[r],r))
        family_results[family] = {"records":ids,"pooled_template_representatives":pooled_ids,
                                  "support":support,"winner":winner,"evidence":evidence}
        for tid in ids:
            matching = {f:v for f,v in candidates[tid].items() if f[0] == winner}
            ranked = sorted(matching,key=lambda f:(-matching[f]["merit"],f))
            accepted, endpoint_support = [], 0.0
            if ranked:
                best = ranked[0]
                endpoint_support = matching[best]["merit"] / sum(v["merit"] for v in matching.values())
                if support[winner] >= MIN_SUPPORT and endpoint_support >= MIN_SUPPORT:
                    accepted=[list(best)]
            predictions[tid] = {"accepted":accepted, "family":family,
                                "family_support":support[winner],"endpoint_support":endpoint_support,
                                "winning_relation":winner,
                                "candidates":[dict(v,frame=list(f)) for f,v in sorted(candidates[tid].items())]}
    return predictions, family_results


def score(records):
    counts = collections.Counter()
    per_target = collections.defaultdict(lambda: [0, 0])
    for r in records:
        predicted = {frame_key(f) for f in r["prediction"]["accepted"]}
        gold = {frame_key(f) for f in r["gold"]}
        required = {frame_key(f) for f in r["targets"]}
        counts["documents"] += 1
        counts["documents_with_calls"] += bool(predicted)
        counts["correct_edges"] += len(predicted & gold)
        counts["called_edges"] += len(predicted)
        counts["gold_edges"] += len(gold)
        counts["target_hits"] += len(predicted & required)
        counts["targets"] += len(required)
        counts["exact_graphs"] += predicted == gold
        per_target[r["target_family"]][0] += len(predicted & required)
        per_target[r["target_family"]][1] += len(required)
    return {**counts,
            "target_recall": counts["target_hits"] / counts["targets"] if counts["targets"] else 0.0,
            "edge_precision": counts["correct_edges"] / counts["called_edges"] if counts["called_edges"] else None,
            "edge_recall": counts["correct_edges"] / counts["gold_edges"] if counts["gold_edges"] else 0.0,
            "coverage": counts["documents_with_calls"] / counts["documents"] if counts["documents"] else 0.0,
            "by_target": {k: {"hits": v[0], "items": v[1], "recall": v[0]/v[1]} for k, v in sorted(per_target.items())}}
