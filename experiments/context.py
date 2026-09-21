"""Evaluate frozen GRUs at exact context lengths; no training or downloads."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from experiments.cloud import write
from voynich.data import digest, load_documents
from voynich.evaluate import paired_interval, summarize, target_signature

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("experiments/context-plan.json")
OUT = Path("artifacts/context-ablation")


def read(path):
    return json.loads(Path(path).read_text())


def check_targets(reference, candidate):
    def keys(rows):
        result = {r["page"]: (r["units"], r["target_sha256"], r["folio"]) for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate page scores")
        return result
    if keys(reference) != keys(candidate):
        raise ValueError("Context scores must use identical targets and folios")


def prepare(case, plan):
    path = Path(f"training_run_outputs/char-gru-{case['dataset']}-s{case['seed']}")
    run = read(path / "run.json")
    config = run["config"]
    if (run["status"] != "complete" or not run["checkpoint_verified"] or run["test_scored"]
            or config["kind"] != "gru" or config["seed"] != case["seed"]
            or Path(config["dataset"]).name != case["dataset"]):
        raise ValueError(f"Unverified source run: {path}")
    if max(plan["contexts"]) > config["context"] or min(plan["contexts"]) < 1:
        raise ValueError("Context outside the trained range")
    data = Path(config["dataset"])
    if digest(data / "documents.json") != run["dataset_sha256"]:
        raise ValueError("Dataset changed since training")
    if digest(path / "best.pt") != run["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    docs = [d for d in load_documents(data) if d["split"] == "validation"]
    if [d["page"] for d in docs] != run["validation_pages"]:
        raise ValueError("Validation split changed")
    old = read(path / "adapted.json")
    targets = [dict(page=d["page"], folio=d["folio"], units=sum(c != "?" for c in d["text"]),
                    target_sha256=target_signature(d["text"])) for d in docs]
    check_targets(old["pages"], targets)
    baseline_path = Path(f"artifacts/results/{case['dataset']}.json")
    for baseline in read(baseline_path)["models"].values():
        check_targets(baseline["pages"], targets)
    files = [path / "run.json", path / "adapted.json", path / "best.pt",
             data / "documents.json", baseline_path]
    source = dict(case, path=str(path), selected_epoch=run["selected_epoch"],
                  input_sha256={str(p): digest(p) for p in files})
    return source, config, docs, targets


def comparisons(results):
    comparisons = []
    for row in results:
        reference = next(r for r in results if r["dataset"] == row["dataset"]
                         and r["seed"] == row["seed"] and r["context"] == 128)
        check_targets(row["pages"], reference["pages"])
        value = dict(dataset=row["dataset"], seed=row["seed"], context=row["context"],
                     gain_to_128_bpc=row["summary"]["overall"]["bits_per_character"]
                     - reference["summary"]["overall"]["bits_per_character"])
        if row["dataset"] in {"gc", "gc-shuffle"}:
            value["paired_folio_interval"] = paired_interval(row["pages"], reference["pages"])
        comparisons.append(value)
    return comparisons


def suite():
    import torch
    from voynich.character import CharacterModel
    from voynich.context import score_context

    os.chdir(ROOT)
    plan = read(PLAN)
    if plan["contexts"] != [8, 16, 32, 64, 128]:
        raise ValueError("Expected the fixed five context lengths")
    if shutil.disk_usage(ROOT).free < plan["min_free_gib"] * 2**30:
        raise RuntimeError("Less than the reserved free disk space")
    if plan["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("This suite requires the local Apple GPU")
    torch.set_num_threads(2)
    with Path("artifacts/replication.lock").open("a+") as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [prepare(case, plan) for case in plan["cases"]]
        OUT.mkdir(exist_ok=False)
        started = time.time()
        code = [PLAN, Path(__file__), Path("voynich/context.py"), Path("voynich/character.py"),
                Path("voynich/evaluate.py"), Path("voynich/data.py")]
        manifest = dict(plan=plan, started_at=started, torch_version=torch.__version__,
                        sources=[s[0] for s in sources],
                        code_sha256={str(p): digest(p) for p in code},
                        protocol="stride 1; reset per target; validation only", test_scored=False)
        write(OUT / "manifest.json", manifest)
        results = []
        try:
            for source, config, docs, targets in sources:
                path = Path(source["path"])
                model = CharacterModel("gru", config["width"], config["layers"], config["context"])
                model.load_state_dict(torch.load(path / "best.pt", map_location="cpu", weights_only=True))
                model.to(plan["device"])
                for context in plan["contexts"]:
                    job = f"{source['dataset']}-s{source['seed']}-c{context}"
                    write(OUT / "status.json", dict(stage="scoring", job=job, completed=len(results),
                                                   total=30, pid=os.getpid(), at=time.time()))
                    tick = time.time()
                    pages = score_context(model, docs, context, plan["batch_size"])
                    check_targets(targets, pages)
                    result = dict(dataset=source["dataset"], seed=source["seed"], context=context,
                                  pages=pages, summary=summarize(pages), seconds=time.time() - tick,
                                  test_scored=False)
                    write(OUT / f"{job}.json", result)
                    results.append(result)
                    print(f"{job}: {result['summary']['overall']['bits_per_character']:.6f} BPC "
                          f"({result['seconds']:.1f}s)", flush=True)
                del model
                torch.mps.empty_cache()
            hashes = dict(manifest["code_sha256"])
            for source in manifest["sources"]:
                hashes.update(source["input_sha256"])
            if any(digest(p) != value for p, value in hashes.items()):
                raise ValueError("An input changed during evaluation")
            write(OUT / "results.json", dict(manifest=manifest, results=results,
                  comparisons=comparisons(results), inputs_verified=True, test_scored=False))
            write(OUT / "status.json", dict(stage="complete", completed=len(results), total=30,
                                           seconds=time.time() - started, at=time.time()))
        except Exception as exc:
            write(OUT / "status.json", dict(stage="failed", error=str(exc), at=time.time()))
            raise


if __name__ == "__main__":
    suite()
