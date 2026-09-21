"""Queue local character baselines after the existing Qwen replication suite."""

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

from voynich.evaluate import paired_interval
from experiments.cloud import write

ROOT = Path(__file__).resolve().parents[1]
STATE = Path("artifacts/characters")
PLAN = Path("experiments/character-plan.json")


def read(path):
    return json.loads(Path(path).read_text())


def jobs(plan):
    result = []
    cases = [("gc", seed) for seed in plan["gc_seeds"]]
    cases += [(dataset, plan["control_seed"]) for dataset in plan["controls"]]
    for dataset, seed in cases:
        for kind, sizes in plan["models"].items():
            name = f"char-{kind}-{dataset}-s{seed}"
            config = {k: plan[k] for k in ("device", "context", "stride", "batch_size",
                       "learning_rate", "epochs", "patience", "max_seconds")}
            result.append(dict(config, **sizes, kind=kind, seed=seed,
                               dataset=f"artifacts/data/{dataset}", output=f"training_run_outputs/{name}"))
    return result


def guard_disk(plan):
    if shutil.disk_usage(ROOT).free < plan["min_free_gib"] * 2**30:
        raise RuntimeError("Less than the reserved free disk space; character suite stopped")
    used = sum(p.stat().st_size for config in jobs(plan)
               for p in Path(config["output"]).rglob("*") if p.is_file())
    if used >= plan["max_output_gib"] * 2**30:
        raise RuntimeError("Character output storage cap reached")


def report(plan):
    results = []
    qwen = read("experiments/replication-results.json")["results"]
    for config in jobs(plan):
        root = Path(config["output"])
        if not (root / "run.json").exists():
            continue
        run = read(root / "run.json")
        if run["status"] != "complete":
            continue
        scores = read(root / "adapted.json")
        dataset = Path(config["dataset"]).name
        references = read(f"artifacts/results/{dataset}.json")["models"]
        name = min(references, key=lambda k: references[k]["summary"]["overall"]["bits_per_character"])
        baseline = references[name]
        # Fingerprints must agree even for synthetic controls where folio CIs are not meaningful.
        target = {r["page"]: (r["units"], r["target_sha256"]) for r in scores["pages"]}
        if target != {r["page"]: (r["units"], r["target_sha256"]) for r in baseline["pages"]}:
            raise ValueError("Character/baseline target mismatch")
        row = dict(model=config["kind"], dataset=dataset, seed=config["seed"],
                   bpc=scores["summary"]["overall"]["bits_per_character"], parameters=run["parameters"],
                   selected_epoch=run["selected_epoch"], epochs=run["epoch"], stop_reason=run["stop_reason"],
                   baseline=name, baseline_bpc=baseline["summary"]["overall"]["bits_per_character"],
                   selected_training_characters=scores["training_characters_seen"], path=str(root))
        two = root / "validation/epoch-002.json"
        row["epoch_2_bpc"] = read(two)["summary"]["overall"]["bits_per_character"] if two.exists() else None
        match = next((r for r in qwen if r["dataset"] == dataset and r["seed"] == config["seed"]), None)
        if match:
            if target != {r["page"]: (r["units"], r["target_sha256"]) for r in match["scores"]["pages"]}:
                raise ValueError("Character/Qwen target mismatch")
            row["qwen_bpc"] = match["bits_per_character"]
            if dataset in {"gc", "gc-shuffle"}:
                row["vs_qwen"] = paired_interval(match["scores"]["pages"], scores["pages"])
        if dataset in {"gc", "gc-shuffle"}:
            row["vs_baseline"] = paired_interval(baseline["pages"], scores["pages"])
        results.append(row)
    write("experiments/character-results.json", dict(results=results, test_scored=False))
    lines = ["# Character-model results", "", "Local models trained from random weights. Lower BPC is better.", "",
             "| Model | Data | Seed | Parameters | Selected epoch | BPC | Epoch 2 BPC | Qwen BPC |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        early = f"{r['epoch_2_bpc']:.4f}" if r["epoch_2_bpc"] is not None else "pending"
        q = f"{r['qwen_bpc']:.4f}" if "qwen_bpc" in r else "pending"
        lines.append(f"| {r['model']} | {r['dataset']} | {r['seed']} | {r['parameters']:,} | "
                     f"{r['selected_epoch']} | {r['bpc']:.4f} | {early} | {q} |")
    lines += ["", "Exact scored characters match the references. Training exposure, context, architecture,",
              "and full-model versus adapter training differ. These are practical baselines, not a causal",
              "estimate of pretraining's effect. Epoch 2 is an early reference, not exact Qwen exposure matching.",
              "Each model selects its best validation epoch; uncertainty is exploratory. Final test remains sealed.",
              "Timm and Naibbe are single synthetic samples, not independent generator replications.", ""]
    Path("experiments/CHARACTERS.md").write_text("\n".join(lines))


def suite():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "suite.lock").open("a+") as own, Path("artifacts/replication.lock").open("a+") as gpu:
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Character suite already queued or running.", flush=True)
            return
        if (STATE / "status.json").exists():
            raise ValueError("Character suite already started; review existing state")
        plan = read(PLAN)
        write(STATE / "plan.json", plan)
        write(STATE / "status.json", dict(stage="waiting_for_local_replications", pid=os.getpid()))
        print("Waiting for the existing local Qwen suite; no GPU allocated yet.", flush=True)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        try:
            for job in read("experiments/replication-plan.json")["jobs"]:
                run = read(Path(job["output"]) / "run.json")
                if run["status"] != "complete" or run["config"] != read(job["config"]):
                    raise ValueError(f"Existing Qwen suite needs review: {job['name']}")
            if read(PLAN) != plan:
                raise ValueError("Character plan changed while queued; review before starting")
            deadline = time.time() + plan["suite_max_hours"] * 3600
            for config in jobs(plan):
                if time.time() >= deadline:
                    write(STATE / "status.json", dict(stage="budget_exhausted", pid=os.getpid()))
                    return
                guard_disk(plan)
                name = Path(config["output"]).name
                if Path(config["output"]).exists():
                    raise ValueError(f"Existing character run needs review: {name}")
                write(STATE / f"{name}.json", config)
                write(STATE / "status.json", dict(stage="training", job=name, pid=os.getpid(), deadline=deadline))
                with (STATE / f"{name}.log").open("w") as log:
                    subprocess.run([sys.executable, "-u", "-m", "experiments.characters", "worker",
                                    str(STATE / f"{name}.json"), "--deadline", str(deadline)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                report(plan)
                print(f"Completed {name}", flush=True)
            write(STATE / "status.json", dict(stage="complete", at=time.time()))
        except Exception as exc:
            write(STATE / "status.json", dict(stage="failed", error=str(exc), at=time.time()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["suite", "worker", "report"])
    parser.add_argument("config", nargs="?")
    parser.add_argument("--deadline", type=float, default=math.inf)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "worker":
        from voynich.character import train
        train(read(args.config), args.deadline)
    elif args.command == "report":
        report(read(PLAN))
    else:
        suite()


if __name__ == "__main__":
    main()
