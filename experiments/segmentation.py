"""Fit, freeze, solve and grade new segmentation cases in separate stages."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from experiments.boundaries import segment as old_segment
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import normalize, edit_distance
from voynich.segmentation import fit, Segmenter, boundaries

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts/segmentation"
OUT = ROOT / "experiments/segmentation"
CODE = ["voynich/segmentation.py", "experiments/segmentation.py", "voynich/decipher.py", "voynich/corpora.py"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def hashes():
    return {p: digest(ROOT / p) for p in CODE}


def corpus(repository, split):
    source = next(s for s in read(ROOT / "experiments/language-sources.json")["sources"] if s["repository"] == repository)
    file = next(f for f in source["files"] if f["path"].endswith(f"-{split}.conllu"))
    if digest(ROOT / file["path"]) != file["sha256"]:
        raise ValueError("Corpus source drift")
    return list(conllu_sentences(ROOT / file["path"])), file


def tune():
    if (STATE / "frozen.json").exists():
        raise FileExistsError("Segmentation configuration already frozen")
    lex = read(ROOT / "experiments/segmentation-sources.json")["lexicon"]
    if digest(ROOT / lex["path"]) != lex["sha256"]:
        raise ValueError("Lexicon drift")
    forms = {normalize(line.split("\t")[0]) for line in (ROOT / lex["path"]).read_text(encoding="latin-1").splitlines()}
    forms = {w for w in forms if w and " " not in w}
    train, train_file = corpus("UD_Italian-ISDT", "train")
    dev, dev_file = corpus("UD_Italian-ISDT", "dev")
    texts = [normalize(" ".join(s["words"])) for s in train]
    model = fit(texts, forms)
    write(STATE / "model.json", model)
    development = [normalize(" ".join(s["words"])) for s in dev]
    prior_set = set(texts)
    development = [t for t in development if 80 <= len(t.replace(" ", "")) <= 300 and t not in prior_set][:200]
    trials, started = [], time.time()
    for alpha, bigram, unknown in itertools.product([.01, .1, 1.], [0., .5, .9], [2., 3.]):
        segmenter = Segmenter(model, alpha, bigram, unknown)
        errors = sum(edit_distance(segmenter.segment(t.replace(" ", "")).split(), t.split()) for t in development)
        row = dict(alpha=alpha, bigram=bigram, unknown=unknown, word_error_rate=errors / sum(len(t.split()) for t in development))
        trials.append(row)
        print(row, flush=True)
    winner = min(trials, key=lambda r: (r["word_error_rate"], r["alpha"], r["bigram"], r["unknown"]))
    frozen = dict(parameters={k: winner[k] for k in ("alpha", "bigram", "unknown")}, beam=8, maximum=32,
                  trials=trials, development_sentences=len(development), lexicon_forms=len(model["lexicon"]),
                  train_source=train_file, development_source=dev_file, lexicon=lex, code_sha256=hashes(),
                  model_sha256=digest(STATE / "model.json"), protocol_sha256=digest(ROOT / "experiments/METHOD_BENCHMARK_PLAN.md"),
                  at=time.time(), seconds=time.time()-started, scope="Non-Dante training/development only; evaluation not generated yet")
    write(STATE / "frozen.json", frozen); write(OUT / "freeze.json", frozen)


def check_freeze():
    frozen = read(STATE / "frozen.json")
    if frozen["code_sha256"] != hashes() or frozen["model_sha256"] != digest(STATE / "model.json"):
        raise ValueError("Frozen method changed")
    return frozen


def prepare():
    frozen = check_freeze()
    if (STATE / "public.json").exists():
        raise FileExistsError("Challenge already prepared")
    old = read(ROOT / "artifacts/decipherment/evaluator-only/answers.json")
    excluded_ids = {s for r in old for s in r["source_sentences"]}
    prior = set()
    for split in ("train", "dev"):
        rows, _ = corpus("UD_Italian-ISDT", split)
        prior.update(normalize(" ".join(r["words"])) for r in rows)
    public, private, sources = [], [], []
    for dataset, repo, split in [("modern", "UD_Italian-ISDT", "test"), ("historical", "UD_Italian-Old", "train")]:
        rows, source = corpus(repo, split); sources.append(source)
        used = set()
        for i in range(12):
            chunk, ids = [], []
            start, stop = i * len(rows) // 12, (i + 1) * len(rows) // 12
            for row in rows[start:stop]:
                text = normalize(" ".join(row["words"]))
                if (dataset == "historical" and row["id"] in excluded_ids) or text in prior or not text:
                    if chunk: break
                    continue
                length = sum(len(t.replace(" ", "")) for t in chunk) + len(text.replace(" ", ""))
                if length > 800:
                    if chunk: break
                    continue
                chunk.append(text); ids.append(row["id"])
                if length >= 400: break
            text = " ".join(chunk)
            if len(text.replace(" ", "")) < 400 or any(x in used for x in ids):
                raise ValueError("Insufficient disjoint challenge text")
            used.update(ids)
            ident = hashlib.sha256((repo + ':' + ':'.join(ids)).encode()).hexdigest()[:16]
            public.append(dict(id=ident, text=text.replace(" ", "")))
            private.append(dict(id=ident, dataset=dataset, plaintext=text, source_ids=ids, source=source))
    private_dir = STATE / "evaluator-only"; private_dir.mkdir(mode=0o700, exist_ok=True)
    write(private_dir / "answers.json", private)
    write(STATE / "public.json", sorted(public, key=lambda r: r["id"]))
    write(STATE / "challenge.json", dict(cases=24, passages_per_corpus=12, old_challenge_sentence_overlap=0,
          prior_sentence_overlap=0, sources=sources, code_sha256=hashes(), frozen_sha256=digest(STATE / "frozen.json"),
          public_sha256=digest(STATE / "public.json"), answers_sha256=digest(private_dir / "answers.json"), at=time.time()))
    print("Prepared 24 fresh passages; originals kept evaluator-only")


def solve():
    frozen = check_freeze()
    if (STATE / "predictions.json").exists(): raise FileExistsError("Predictions already frozen")
    model = read(STATE / "model.json")
    segmenter = Segmenter(model, **frozen["parameters"])
    import math
    total = sum(model["counts"].values()); costs = {w: -math.log(n / total) for w, n in model["counts"].items()}
    rows, started = [], time.time()
    for case in read(STATE / "public.json"):
        rows.append(dict(id=case["id"], lexicon=segmenter.segment(case["text"]),
                         baseline=old_segment(case["text"], costs, 2.)))
    write(STATE / "predictions.json", dict(rows=rows, code_sha256=hashes(), public_sha256=digest(STATE / "public.json"),
          frozen_sha256=digest(STATE / "frozen.json"), seconds=time.time()-started, at=time.time()))
    print("Predictions frozen; no originals opened")


def metrics(candidate, reference):
    truth, found = boundaries(reference), boundaries(candidate)
    if candidate.replace(" ", "") != reference.replace(" ", ""):
        raise ValueError("Segmenter changed characters")
    return dict(errors=edit_distance(candidate.split(), reference.split()), words=len(reference.split()),
                tp=len(truth & found), fp=len(found - truth), fn=len(truth - found), exact=candidate == reference)


def evaluate():
    check_freeze(); challenge=read(STATE / "challenge.json"); predictions=read(STATE / "predictions.json")
    for field, path in [("public_sha256", STATE / "public.json"), ("answers_sha256", STATE / "evaluator-only/answers.json"), ("frozen_sha256", STATE / "frozen.json")]:
        if challenge[field] != digest(path): raise ValueError("Challenge drift")
    if predictions["code_sha256"] != hashes() or predictions["public_sha256"] != challenge["public_sha256"]:
        raise ValueError("Prediction provenance mismatch")
    gold = {r["id"]: r for r in read(STATE / "evaluator-only/answers.json")}
    if {r["id"] for r in predictions["rows"]} != set(gold): raise ValueError("Missing challenge predictions")
    rows=[]
    for r in predictions["rows"]:
        truth=gold[r["id"]]
        rows.append(dict(id=r["id"], dataset=truth["dataset"],
                         **{method: metrics(r[method], truth["plaintext"]) for method in ("lexicon", "baseline")}))
    summary={}
    for dataset in ("modern", "historical"):
        cases=[r for r in rows if r["dataset"]==dataset]; result={}
        for method in ("lexicon", "baseline"):
            totals={k:sum(r[method][k] for r in cases) for k in ("errors","words","tp","fp","fn","exact")}
            result[method]=dict(word_error_rate=totals["errors"]/totals["words"],
                               boundary_f1=2*totals["tp"]/(2*totals["tp"]+totals["fp"]+totals["fn"]),
                               precision=totals["tp"]/(totals["tp"]+totals["fp"]),
                               recall=totals["tp"]/(totals["tp"]+totals["fn"]), **totals)
        values=np.array([[r['baseline']['errors']-r['lexicon']['errors'],r['lexicon']['words']] for r in cases])
        sampled=values[np.random.default_rng(42).integers(0,len(cases),(2000,len(cases)))].sum(axis=1)
        result['absolute_wer_improvement_interval_95']=np.quantile(sampled[:,0]/sampled[:,1],[.025,.975]).tolist()
        result['relative_wer_reduction']=1-result['lexicon']['word_error_rate']/result['baseline']['word_error_rate']
        summary[dataset]=result
    write(OUT / "results.json",dict(summary=summary, cases=rows, challenge=challenge,
          predictions_sha256=digest(STATE / "predictions.json"), test_scored=False,
          limitation="Fresh passages, not independent works; Dante shares author/work with earlier benchmark. No tuning on these answers."))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument('command',choices=['tune','prepare','solve','evaluate'])
    globals()[p.parse_args().command]()
