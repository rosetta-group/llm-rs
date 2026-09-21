"""Prepare, solve, and separately grade a controlled Italian recovery benchmark."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import ALPHABET, edit_distance, language_model, normalize, recover, restore_spaces

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts/decipherment"
PUBLIC = STATE / "public"
PRIVATE = STATE / "evaluator-only"
OUT = ROOT / "experiments/decipherment"
PLAN = dict(passages=6, keys_per_passage=2, minimum_letters=1200, maximum_letters=2000,
            source_language="Italian", prior="UD_Italian-ISDT training sentences",
            challenge="UD_Italian-Old (Dante), disjoint from the prior corpus",
            restarts=6, steps_per_restart=12000, seed=42,
            families=["substitution-spaces", "substitution-no-spaces", "naibbe-known-structure"],
            scope="Cipher family and source language known; Naibbe codebook structure supplied; letter key hidden")


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def source(repository):
    sources = read(ROOT / "experiments/language-sources.json")["sources"]
    s = next(s for s in sources if s["repository"] == repository)
    f = next(f for f in s["files"] if f["path"].endswith("-train.conllu"))
    if digest(ROOT / f["path"]) != f["sha256"]:
        raise ValueError("Source hash mismatch")
    return list(conllu_sentences(ROOT / f["path"])), f


def structure():
    rows = list(csv.DictReader((STATE / "vendor/references/naibbe_tables.csv").open(encoding="utf-8-sig")))
    maps = {state: {r["glyphs"]: r["code"].split("_")[-1] for r in rows
                   if r["code"].startswith(state + "_")} for state in ("unigram", "prefix", "suffix")}
    options = defaultdict(set)
    for a, x in maps["prefix"].items():
        for b, y in maps["suffix"].items():
            options[a + b].add(x + y)
    # Published UNAMBIGUOUS=True avoids bigrams that collide with unigram spellings.
    for glyph, letter in maps["unigram"].items():
        options[glyph] = {letter}
    return {glyph: sorted(values) for glyph, values in options.items()}


def prepare():
    if PUBLIC.exists() or PRIVATE.exists():
        raise FileExistsError("Benchmark already prepared; preserve its split and hidden keys")
    vendor_sources = read(ROOT / "experiments/decipherment-sources.json")
    for f in vendor_sources["files"]:
        if digest(ROOT / f["path"]) != f["sha256"]:
            raise ValueError("Naibbe source drift")
    prior, prior_file = source("UD_Italian-ISDT")
    challenge, challenge_file = source("UD_Italian-Old")
    prior_texts = [normalize(" ".join(s["words"])) for s in prior]
    prior_set = set(prior_texts)
    PUBLIC.mkdir(); PRIVATE.mkdir()
    os.chmod(PRIVATE, 0o700)
    word_counts = Counter(w for s in prior_texts for w in s.split())
    write(PUBLIC / "word-counts.json", word_counts)
    for spaces in (True, False):
        probabilities, counts = language_model(prior_texts, spaces)
        np.savez(PUBLIC / f"lm-{int(spaces)}.npz", log_probabilities=probabilities, letter_counts=counts)
    spec = importlib.util.spec_from_file_location("published_naibbe", STATE / "vendor/naibbe.py")
    vendor = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(STATE / "vendor"); spec.loader.exec_module(vendor)
    finally:
        os.chdir(cwd)
    mappings = structure()
    write(PUBLIC / "naibbe-structure.json", mappings)
    # Six nonoverlapping source regions. No passage is chosen based on recovery scores.
    passages = []
    for index in range(PLAN["passages"]):
        start = index * len(challenge) // PLAN["passages"]
        chosen, texts = [], []
        for sentence in challenge[start:]:
            text = normalize(" ".join(sentence["words"]))
            if text in prior_set:
                raise ValueError("Challenge/prior sentence overlap")
            if sum(len(t.replace(" ", "")) for t in texts) + len(text.replace(" ", "")) > PLAN["maximum_letters"]:
                break
            chosen.append(sentence["id"]); texts.append(text)
            if sum(len(t.replace(" ", "")) for t in texts) >= PLAN["minimum_letters"]:
                break
        if sum(len(t.replace(" ", "")) for t in texts) < PLAN["minimum_letters"]:
            raise ValueError("Insufficient challenge text")
        passages.append(dict(id=f"passage-{index:02d}", sentences=chosen, plaintext=" ".join(texts)))
    all_ids = [s for p in passages for s in p["sentences"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Challenge passages overlap")
    cases, gold = [], []
    for p in passages:
        for key_index in range(PLAN["keys_per_passage"]):
            seed = random.SystemRandom().randrange(2**63)
            alphabet = list(ALPHABET); random.Random(seed).shuffle(alphabet)
            key = dict(zip(ALPHABET, alphabet))
            inverse = {v: k for k, v in key.items()}
            spaced = "".join(key.get(c, c) for c in p["plaintext"])
            dense = spaced.replace(" ", "")
            random.seed(seed)
            trace = io.StringIO()
            tokens = vendor.encrypt_naibbe(dense, vendor.naibbe_tables, vendor.placeholder_to_glyph,
                                          use_78=False, pre_plaintext_file=trace)
            chunks = trace.getvalue().split()
            if len(tokens) != len(chunks) or any(chunk not in mappings[token] for token, chunk in zip(tokens, chunks)):
                raise ValueError("Published encoding is inconsistent with its candidate decoding")
            roundtrip = "".join(inverse[c] for c in "".join(chunks))
            if roundtrip != p["plaintext"].replace(" ", ""):
                raise ValueError("Keyed normalization roundtrip failed")
            for family, ciphertext in zip(PLAN["families"], (spaced, dense, " ".join(tokens))):
                ident = f"{p['id']}-key-{key_index}-{family}"
                cases.append(dict(id=ident, passage=p["id"], family=family, ciphertext=ciphertext))
                gold.append(dict(id=ident, plaintext=p["plaintext"], encryption_key=key,
                                 source_sentences=p["sentences"], generator_seed=seed,
                                 keyed_roundtrip=True, ambiguous_naibbe_tokens=sum(len(mappings[t]) > 1 for t in tokens)))
    write(PUBLIC / "cases.json", cases)
    write(PRIVATE / "answers.json", gold)
    write(STATE / "manifest.json", dict(plan=PLAN, prior_source=prior_file, challenge_source=challenge_file,
          private_answers_sha256=digest(PRIVATE / "answers.json"),
          public_sha256={str(p.relative_to(STATE)): digest(p) for p in PUBLIC.iterdir()},
          source_sentence_overlap=0, test_scored=False,
          note="Benchmark test plaintext is separate from the untouched Voynich final test"))
    OUT.mkdir(exist_ok=True)
    write(OUT / "plan.json", PLAN)
    print(f"Prepared {len(cases)} blinded cases from {len(passages)} disjoint passages", flush=True)


def solve():
    """Only the public directory and fixed plan are inputs; answers are not opened."""
    destination = STATE / "predictions.json"
    if destination.exists():
        raise FileExistsError("Predictions already frozen")
    started = time.time()
    models = {spaces: np.load(PUBLIC / f"lm-{int(spaces)}.npz") for spaces in (True, False)}
    counts = read(PUBLIC / "word-counts.json")
    codebook = read(PUBLIC / "naibbe-structure.json")
    results = []
    for i, case in enumerate(read(PUBLIC / "cases.json")):
        spaced = case["family"] == "substitution-spaces"
        cipher = case["ciphertext"]
        ambiguity = 0
        if case["family"] == "naibbe-known-structure":
            tokens = cipher.split()
            options = [codebook[t] for t in tokens]
            ambiguity = sum(len(o) > 1 for o in options)
            # Fixed lexical tie-break uses no true key or original content.
            cipher = "".join(o[0] for o in options)
        model = models[spaced]
        result = recover(cipher, model["log_probabilities"], model["letter_counts"], spaces=spaced,
                         seed=PLAN["seed"] + i, restarts=PLAN["restarts"], steps=PLAN["steps_per_restart"])
        result.update(id=case["id"], passage=case["passage"], family=case["family"],
                      ambiguous_tokens=ambiguity, supplied_original_spaces=spaced)
        if not spaced:
            result["recovered"] = restore_spaces(result["recovered"], counts)
            result["frequency_baseline"] = restore_spaces(result["frequency_baseline"], counts)
        results.append(result)
        write(STATE / "progress.json", dict(completed=len(results), total=36, seconds=time.time() - started,
                                           latest=case["id"], stage="blind recovery"))
        print(f"Recovered {len(results)}/36: {case['id']}", flush=True)
    write(destination, dict(results=results, seconds=time.time() - started,
          solver_sha256=digest(ROOT / "voynich/decipher.py"), driver_sha256=digest(__file__),
          public_sha256={str(p.relative_to(STATE)): digest(p) for p in PUBLIC.iterdir()},
          access="Public ciphertext, unpaired Italian prior, known cipher structure; no answer/key file"))


def evaluate():
    predictions = read(STATE / "predictions.json")
    manifest = read(STATE / "manifest.json")
    if (predictions["solver_sha256"] != digest(ROOT / "voynich/decipher.py")
            or predictions["driver_sha256"] != digest(__file__)):
        raise ValueError("Solver changed after blind predictions were frozen")
    if predictions["public_sha256"] != manifest["public_sha256"]:
        raise ValueError("Public inputs changed after benchmark preparation")
    if digest(PRIVATE / "answers.json") != manifest["private_answers_sha256"]:
        raise ValueError("Evaluator answers changed")
    gold = {r["id"]: r for r in read(PRIVATE / "answers.json")}
    results = []
    for r in predictions["results"]:
        target = gold[r["id"]]["plaintext"]
        dense = target.replace(" ", "")
        row = dict(r, reference=target)
        for label in ("recovered", "frequency_baseline"):
            candidate = r[label].replace(" ", "")
            if len(candidate) != len(dense):
                raise ValueError("Unexpected substitution length change")
            row[label + "_character_accuracy"] = sum(a == b for a, b in zip(candidate, dense)) / len(dense)
            row[label + "_word_error_rate"] = edit_distance(r[label].split(), target.split()) / len(target.split())
        row.update(characters=len(dense), words=len(target.split()), keyed_roundtrip=gold[r["id"]]["keyed_roundtrip"])
        results.append(row)
    if len(results) != len(gold):
        raise ValueError("Incomplete benchmark")
    summaries = {}
    for family in PLAN["families"]:
        rows = [r for r in results if r["family"] == family]
        summaries[family] = {metric: sum(r[metric] * r[unit] for r in rows) / sum(r[unit] for r in rows)
                            for metric, unit in [("recovered_character_accuracy", "characters"),
                                                ("frequency_baseline_character_accuracy", "characters"),
                                                ("recovered_word_error_rate", "words")]}
    write(OUT / "results.json", dict(plan=PLAN, manifest=manifest, summaries=summaries, results=results,
          predictions_sha256=digest(STATE / "predictions.json"), solver_sha256=predictions["solver_sha256"],
          seconds=predictions["seconds"], voynich_test_scored=False,
          limitation="Naibbe uses the published structural codebook plus a hidden global letter permutation; it is not unknown-cipher discovery"))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "solve", "evaluate"])
    args = parser.parse_args()
    globals()[args.command]()
