"""Bounded local suite: matched training and evaluation histories."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.cloud import write
from experiments.context import check_targets
from voynich.data import cluster, digest, load_documents
from voynich.evaluate import paired_interval

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("experiments/matched-plan.json")
STATE = Path("artifacts/matched")


def read(path):
    return json.loads(Path(path).read_text())


def jobs(plan):
    keys = ("device", "width", "layers", "batch_size", "eval_batch_size", "learning_rate", "epochs", "min_free_gib")
    return [dict({k: plan[k] for k in keys}, context=context, seed=seed,
                 dataset=f"artifacts/data/{dataset}",
                 output=f"training_run_outputs/matched-{dataset}-c{context}-s{seed}")
            for seed in plan["seeds"] for context in plan["contexts"] for dataset in plan["datasets"]]


def guard_disk(plan):
    if shutil.disk_usage(ROOT).free < plan["min_free_gib"] * 2**30:
        raise RuntimeError("Less than the reserved free disk space")
    used = sum(p.stat().st_size for c in jobs(plan) for p in Path(c["output"]).rglob("*") if p.is_file())
    if used > plan["max_output_gib"] * 2**30:
        raise RuntimeError("Matched suite output cap exceeded")


def interaction(gc_short, gc_long, shuffled_short, shuffled_long, draws=2000):
    """Difference of within-text gains; resample matched source folios together."""
    check_targets(gc_short, gc_long)
    check_targets(shuffled_short, shuffled_long)
    maps = [{r["page"]: r for r in rows} for rows in (gc_short, gc_long, shuffled_short, shuffled_long)]
    if any(m.keys() != maps[0].keys() for m in maps):
        raise ValueError("Interaction needs the same source pages")
    groups = {}
    for page, row in maps[0].items():
        others = [m[page] for m in maps]
        if any(r["folio"] != row["folio"] or r["units"] != row["units"] for r in others):
            raise ValueError("Interaction needs matching folios and character counts")
        delta = others[0]["nll"] - others[1]["nll"] - others[2]["nll"] + others[3]["nll"]
        group = cluster(row["folio"], "folio")
        groups.setdefault(group, np.zeros(2))[:] += (delta, row["units"])
    if len(groups) < 2:
        raise ValueError("Need at least two folios")
    values = np.array(list(groups.values()))
    samples = np.random.default_rng(42).integers(0, len(values), (draws, len(values)))
    sums = values[samples].sum(axis=1)
    distribution = sums[:, 0] / sums[:, 1] / math.log(2)
    return dict(extra_intact_gain_bpc=float(values[:, 0].sum() / values[:, 1].sum() / math.log(2)),
                interval_95=[float(v) for v in np.quantile(distribution, [.025, .975])],
                groups=len(groups), draws=draws, seed=42,
                method="paired source-folio bootstrap of the difference of within-text context gains")


def training_seed(result, seed):
    result = dict(result)
    result["bootstrap_seed"] = result.pop("seed")
    return dict(result, seed=seed)


def report():
    plan = read(STATE / "manifest.json")["plan"]
    results = []
    for config in jobs(plan):
        root = Path(config["output"])
        if not (root / "run.json").exists():
            continue
        run = read(root / "run.json")
        if run["status"] != "complete":
            continue
        if (run["config"] != config or not run["checkpoint_verified"] or run["test_scored"]
                or digest(root / "last.pt") != run["checkpoint_sha256"]):
            raise ValueError(f"Unverified result: {root}")
        scores = read(root / "last.json")
        if (scores["epoch"] != plan["epochs"] or scores["training_characters_seen"]
                != plan["epochs"] * run["training_characters_per_epoch"]):
            raise ValueError("Unequal training exposure")
        references = read(f"artifacts/results/{Path(config['dataset']).name}.json")["models"]
        check_targets(next(iter(references.values()))["pages"], scores["pages"])
        results.append(dict(dataset=Path(config["dataset"]).name, seed=config["seed"],
                            context=config["context"], bpc=scores["summary"]["overall"]["bits_per_character"],
                            scores=scores, path=str(root), initial_sha256=run["initial_sha256"],
                            parameters=run["parameters"], best_epoch=run["best_epoch"],
                            best_bpc=run["best_validation_bpc"], seconds=run["wall_seconds"]))
    if len({r["scores"]["training_characters_seen"] for r in results}) > 1:
        raise ValueError("Results differ in total training exposure")
    for seed in plan["seeds"]:
        if len({r["initial_sha256"] for r in results if r["seed"] == seed}) > 1:
            raise ValueError("Paired initial weights differ")
    by_case = {(r["dataset"], r["seed"], r["context"]): r for r in results}
    gains, contrasts = [], []
    for seed in plan["seeds"]:
        for dataset in plan["datasets"]:
            if all((dataset, seed, c) in by_case for c in (8, 128)):
                a, b = [by_case[dataset, seed, c]["scores"]["pages"] for c in (8, 128)]
                gains.append(dict(dataset=dataset, **training_seed(paired_interval(a, b), seed)))
        keys = [(dataset, seed, c) for dataset in plan["datasets"] for c in (8, 128)]
        if all(k in by_case for k in keys):
            contrasts.append(training_seed(interaction(*(by_case[k]["scores"]["pages"] for k in keys)), seed))
    data = dict(plan=plan, results=results, gains=gains, contrasts=contrasts,
                complete=len(results) == len(jobs(plan)), primary="fixed final epoch; gain from 8 to 128",
                test_scored=False, updated_at=time.time())
    write("experiments/matched-results.json", data)
    lines = ["# Matched-context results", "", f"Completed: {len(results)}/18 runs. Primary score: epoch 20, exact context at every target.", "",
             "[Protocol](MATCHED_PLAN.md) · [Project goal and research record](../RESEARCH_LOG.md)", "",
             "| Data | Seed | Context | Final BPC | Best validation BPC (secondary) | Training characters |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['context']} | {r['bpc']:.4f} | "
                     f"{r['best_bpc']:.4f} | {r['scores']['training_characters_seen']:,} |")
    lines += ["", "## Does longer history help intact text more?", "",
              "Positive extra gain favors intact Voynich. Each gain compares identical targets within its own text. "
              "The interaction resamples corresponding original/shuffled folios together, without pretending their strings are identical.", "",
              "| Seed | Extra intact gain, 8 to 128 (BPC) | 95% folio interval |", "|---:|---:|---|"]
    for r in contrasts:
        low, high = r["interval_95"]
        lines.append(f"| {r['seed']} | {r['extra_intact_gain_bpc']:.4f} | [{low:.4f}, {high:.4f}] |")
    lines += ["", "These are validation results, conditional on the fixed optimization budget. "
              "Twenty passes match exposure, not compute or convergence. Best-epoch scores are secondary and can have unequal exposure. "
              "A positive interaction identifies an effect of preserving order under these controls, not a translation. "
              "A null result does not imply meaningless text. Final-test pages remain sealed.", ""]
    Path("experiments/MATCHED.md").write_text("\n".join(lines))
    return data


def suite(resume=False):
    import torch
    plan = read(PLAN)
    if not torch.backends.mps.is_available() or plan["device"] != "mps":
        raise RuntimeError("This suite requires the local Apple GPU")
    configs = jobs(plan)
    if len(configs) != 18 or plan["primary"] != "final_epoch":
        raise ValueError("Expected the fixed 18-run final-epoch design")
    with Path("artifacts/replication.lock").open("a+") as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume and any(Path(c["output"]).exists() for c in configs):
            raise FileExistsError("A matched run already exists; review before restarting")
        guard_disk(plan)
        counts = []
        for dataset in plan["datasets"]:
            docs = load_documents(f"artifacts/data/{dataset}")
            counts.append(sum(c != "?" for d in docs if d["split"] == "train" for c in d["text"]))
        if len(set(counts)) != 1:
            raise ValueError("Training character counts must match across datasets")
        if not resume:
            STATE.mkdir(exist_ok=False)
        files = [PLAN, Path(__file__), Path("voynich/matched.py"), Path("voynich/character.py"),
                 Path("voynich/context.py"), Path("voynich/data.py"), Path("voynich/evaluate.py"),
                 Path("experiments/context.py"), Path("experiments/cloud.py")]
        files += [Path(f"artifacts/data/{d}/documents.json") for d in plan["datasets"]]
        files += [Path(f"artifacts/results/{d}.json") for d in plan["datasets"]]
        hashes = {str(p): digest(p) for p in files}
        if resume:
            manifest = read(STATE / "manifest.json")
            if manifest["plan"] != plan or manifest["input_sha256"] != hashes:
                raise ValueError("Resume requires the reviewed frozen inputs and unchanged plan")
            if read(STATE / "status.json")["stage"] != "failed":
                raise ValueError("Only a stopped, failed suite may be resumed")
            for config in configs:
                path = Path(config["output"])
                if path.exists() and read(path / "run.json")["status"] != "complete":
                    raise ValueError("An incomplete training run needs separate review")
            started, deadline = manifest["started_at"], manifest["deadline"]
        else:
            started = time.time()
            deadline = started + plan["max_suite_hours"] * 3600
            write(STATE / "manifest.json", dict(plan=plan, started_at=started, deadline=deadline,
                  input_sha256=hashes, training_characters_per_epoch=counts[0], test_scored=False))
        try:
            report()
            for index, config in enumerate(configs):
                if resume and Path(config["output"]).exists():
                    continue  # report() has verified each completed configuration and checkpoint.
                guard_disk(plan)
                if any(digest(p) != value for p, value in hashes.items()):
                    raise ValueError("Frozen code or data changed during the suite")
                limit = min(deadline, time.time() + plan["max_job_hours"] * 3600)
                if limit <= time.time():
                    raise TimeoutError("Suite budget exhausted")
                name = Path(config["output"]).name
                config_path = STATE / f"{name}.json"
                write(config_path, config)
                write(STATE / "status.json", dict(stage="training", job=name, completed=index, total=len(configs),
                      pid=os.getpid(), deadline=deadline, job_deadline=limit, at=time.time()))
                with (STATE / f"{name}.log").open("w") as log:
                    subprocess.run([sys.executable, "-u", "-m", "experiments.matched", "worker", str(config_path),
                                    "--deadline", str(limit)], stdout=log, stderr=subprocess.STDOUT,
                                   timeout=max(1, limit - time.time()), check=True)
                report()
                print(f"Completed {index + 1}/18: {name}", flush=True)
            if any(digest(p) != value for p, value in hashes.items()):
                raise ValueError("Frozen inputs changed")
            write(STATE / "status.json", dict(stage="complete", completed=len(configs), total=len(configs),
                                             seconds=time.time() - started, inputs_verified=True, at=time.time()))
        except Exception as exc:
            write(STATE / "status.json", dict(stage="failed", error=str(exc), at=time.time()))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["suite", "resume", "worker", "report"])
    parser.add_argument("config", nargs="?")
    parser.add_argument("--deadline", type=float, default=math.inf)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "worker":
        from voynich.matched import train
        train(read(args.config), args.deadline)
    elif args.command == "report":
        report()
    else:
        suite(resume=args.command == "resume")
