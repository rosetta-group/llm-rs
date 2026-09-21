"""Replicate a promising learning-curve result, then test published text controls."""

import json
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from voynich.data import write_json
from voynich.evaluate import paired_interval


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    baseline = read("artifacts/results/gc.json")["models"]["copy"]
    candidates = []
    for selection in ("random", "outer"):
        path = Path(f"training_run_outputs/qwen-{selection}-c64-s42-n3000")
        run = read(path/"run.json")
        if run["status"] != "complete":
            raise ValueError(f"Primary run is not complete: {path}")
        scores = read(path/"adapted.json")
        candidates.append((scores["summary"]["overall"]["bits_per_character"], path, run, scores))
    _, path, run, scores = min(candidates, key=lambda item: item[0])
    gain = paired_interval(baseline["pages"], scores["pages"])
    if gain["interval_95"][0] <= 0:
        return dict(status="gate_not_met", gain=gain, jobs=[])
    base = run["config"]
    jobs = []
    # Hold the chosen layer subset fixed; these seeds vary adapter initialization and data order.
    cases = [("gc", seed) for seed in (43, 44, 45, 46)]
    cases += [(dataset, 42) for dataset in ("gc-shuffle", "timm", "naibbe")]
    for dataset, seed in cases:
        name = f"qwen-replica-{dataset}-s{seed}-n3000"
        config = dict(base, run_name=name, output_dir=f"training_run_outputs/{name}",
                      dataset=f"artifacts/data/{dataset}", seed=seed, trainable_layers=run["layers"],
                      frozen_reference=str(path) if dataset == "gc" else None)
        config_path = Path("experiments/generated/replications")/f"{name}.json"
        write_json(config_path, config)
        jobs.append(dict(name=name, dataset=dataset, seed=seed, config=str(config_path),
                         output=config["output_dir"], log=f"artifacts/{name}.log"))
    return dict(status="ready", primary=str(path), layers=run["layers"], gain=gain,
                seeds=[42, 43, 44, 45, 46], layer_subset="fixed across seeds", jobs=jobs)


def report(plan):
    results = []
    paths = [dict(name="primary", dataset="gc", seed=42, output=plan["primary"]), *plan["jobs"]]
    for job in paths:
        path = Path(job["output"])
        if not (path/"run.json").exists() or read(path/"run.json").get("status") != "complete":
            continue
        run, scores = read(path/"run.json"), read(path/"adapted.json")
        baselines = read(f"artifacts/results/{job['dataset']}.json")["models"]
        name = min(baselines, key=lambda key: baselines[key]["summary"]["overall"]["bits_per_character"])
        reference = baselines[name]
        # Synthetic blocks are not independent generator samples; report their point gains only.
        bpc = scores["summary"]["overall"]["bits_per_character"]
        row = dict(dataset=job["dataset"], seed=job["seed"], selected_step=run["selected_step"],
                   bits_per_character=bpc, baseline=name,
                   baseline_bpc=reference["summary"]["overall"]["bits_per_character"],
                   scores=scores, layers=run["layers"], path=str(path))
        if job["dataset"] in {"gc", "gc-shuffle"}:
            row["vs_baseline"] = paired_interval(reference["pages"], scores["pages"])
        results.append(row)
    write_json("experiments/replication-results.json", dict(plan=plan, results=results, test_scored=False))
    lines = ["# Replication results", "", f"Selected layer subset: {plan['layers']}. Held fixed across all runs.",
             "Five seeds vary initialization and data order. Each run selects its best validation checkpoint within 3,000 updates.",
             "Controls use seed 42 only. The final test set remains sealed.", "",
             "| Dataset | Seed | Selected update | Qwen bits/character | Best baseline | Baseline bits/character |",
             "|---|---:|---:|---:|---|---:|"]
    for row in results:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['selected_step']} | "
                     f"{row['bits_per_character']:.4f} | {row['baseline']} | {row['baseline_bpc']:.4f} |")
    lines += ["", "Compare Qwen with the baseline within each row; different datasets have different target strings.",
              "Timm and Naibbe are single published samples, not independent generator replications.",
              "These remain validation experiments: checkpoint and layer selection used these pages.",
              "This tests stability of the selected configuration, not a general claim about which layers encode language.", ""]
    Path("experiments/REPLICATIONS.md").write_text("\n".join(lines))


def run_suite():
    plan = prepare()
    write_json("experiments/replication-plan.json", plan)
    if not plan["jobs"]:
        print("The validation gain did not pass the replication gate.", flush=True)
        return
    report(plan)
    for job in plan["jobs"]:
        run_file = Path(job["output"])/"run.json"
        if run_file.exists():
            previous = read(run_file)
            expected = read(job["config"])
            if previous["status"] == "complete" and previous["config"] == expected:
                report(plan)
                continue
            raise ValueError(f"Existing run needs review: {job['output']}")
        print(f"Starting {job['name']}", flush=True)
        with Path(job["log"]).open("w") as log:
            subprocess.run([sys.executable, "run_fine_tuning.py", job["config"]],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report(plan)
        print(f"Completed {job['name']}", flush=True)


def main():
    # Prevent overlapping progress checks from launching two GPU jobs.
    with Path("artifacts/replication.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Replication suite is already running.", flush=True)
            return
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        run_suite()


if __name__ == "__main__":
    main()
