"""Bounded Runpod comparison. No credentials or final-test text enter the bundle."""

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = Path("artifacts/cloud")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def bundle():
    target = ROOT / "artifacts/cloud-upload.tar.gz"
    files = ["fine_tuning.py", "run_fine_tuning.py", "configs/config_model.py",
             "experiments/cloud.py", "experiments/cloud-plan.json", "experiments/cloud-requirements.txt"]
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"voynich").glob("*.py"))]
    payload = {name: (ROOT/name).read_bytes() for name in files}
    for name in ("gc", "gc-shuffle"):
        prefix = f"artifacts/data/{name}"
        docs = read(ROOT/prefix/"documents.json")
        docs = [d for d in docs if d["split"] in {"train", "validation"}]
        payload[f"{prefix}/documents.json"] = (json.dumps(docs, indent=2)+"\n").encode()
        payload[f"{prefix}/manifest.json"] = (ROOT/prefix/"manifest.json").read_bytes()
        path = f"artifacts/results/{name}.json"
        assert read(ROOT/path)["split"] == "validation"
        payload[path] = (ROOT/path).read_bytes()
    manifest = {name: hashlib.sha256(body).hexdigest() for name, body in payload.items()}
    payload["bundle-manifest.json"] = json.dumps(manifest, indent=2).encode()
    with tarfile.open(target, "w:gz") as archive:
        for name, body in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(body))
    print(target, target.stat().st_size, "bytes; training/validation only")


def arm(started_at, hourly):
    plan = read("experiments/cloud-plan.json")
    if not math.isfinite(hourly) or not 0 < hourly <= plan["max_total_hourly_usd"]:
        raise ValueError("Hourly quote exceeds the approved envelope")
    if not math.isfinite(started_at) or not 0 <= time.time()-started_at < 4*3600:
        raise ValueError("Supply the real provisioning timestamp, no future timestamps")
    if (STATE/"budget.json").exists():
        raise ValueError("Budget already armed; do not reset its deadline")
    seconds = min(plan["max_hours"]*3600,
                  (plan["budget_usd"]-plan["reserved_usd"])/hourly*3600)
    value = dict(started_at=started_at, deadline=started_at+seconds, hourly_usd=hourly,
                 max_compute_usd=seconds/3600*hourly, reserved_usd=plan["reserved_usd"])
    write(STATE/"budget.json", value)
    return value


def time_left(reserve=600):
    return read(STATE/"budget.json")["deadline"] - time.time() - reserve


def stop_command():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        raise ValueError("This command must run inside the provisioned Runpod pod")
    for parts in (["runpodctl", "pod", "stop"], ["runpodctl", "stop", "pod"]):
        result = subprocess.run(parts+["--help"], capture_output=True, text=True, timeout=10)
        # Older CLIs return success and root help for an unknown nested command.
        usage = rf"(?m)^\s*{re.escape(' '.join(parts))}(?:\s|$)"
        if result.returncode == 0 and re.search(usage, result.stdout):
            return parts+[pod_id]
    raise RuntimeError("No supported pod-stop command; do not start paid training")


def watchdog():
    command = stop_command()
    write(STATE/"watchdog.json", dict(pid=os.getpid(), deadline=read(STATE/"budget.json")["deadline"]))
    while time_left(reserve=0) > 0:
        time.sleep(max(0, min(20, time_left(reserve=0))))
    for attempt in range(10):
        result = subprocess.run(command, capture_output=True, timeout=30)
        write(STATE/"stop-result.json", dict(attempt=attempt, returncode=result.returncode, at=time.time()))
        if result.returncode == 0:
            return
        time.sleep(10)
    raise RuntimeError("Pod stop failed; inspect immediately")


def config_for(size, dataset, steps, benchmark=False):
    plan = read("experiments/cloud-plan.json")
    item = plan["models"][size]
    name = f"cloud-qwen-{size}-base-{dataset}-s42-n{steps}" + ("-benchmark" if benchmark else "")
    return dict(model_path=item["model_path"], revision=item["revision"], local_files_only=True,
                lora_rank=item["lora_rank"], lora_alpha=item["lora_alpha"], device="cuda", dtype="bfloat16",
                dataset=f"artifacts/data/{dataset}", run_name=name, output_dir=f"training_run_outputs/{name}",
                layer_selection="outer", layer_count=4, context=plan["context"], stride=plan["stride"],
                max_steps=steps, train_batch_size=1, gradient_accumulation_steps=2, eval_batch_size=1,
                learning_rate=0.0001, eval_steps=steps if benchmark else plan["eval_steps"],
                save_steps=steps if benchmark else plan["eval_steps"], logging_steps=10 if benchmark else 50,
                max_eval_pages=1 if benchmark else None, seed=plan["seed"], load_best_model_at_end=True)


def worker(config_path):
    import torch
    from transformers import TrainerCallback
    import fine_tuning
    from configs.config_model import FineTuningConfig
    config = FineTuningConfig.model_validate_json(Path(config_path).read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    measurements = dict(step_seconds=[], evaluation_seconds=[], gpu=torch.cuda.get_device_name())

    class Timer(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            measurements["step_seconds"].append(time.monotonic()-self.start)

    original = fine_tuning.WeightedTrainer

    class TimedTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(Timer())

        def evaluate(self, *args, **kwargs):
            start = time.monotonic()
            value = super().evaluate(*args, **kwargs)
            measurements["evaluation_seconds"].append(time.monotonic()-start)
            return value

    fine_tuning.WeightedTrainer = TimedTrainer
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    fine_tuning.run_fine_tuning(config)
    measurements.update(wall_seconds=time.monotonic()-start,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    write(Path(config.output_dir)/"timing.json", measurements)


def execute(config, limit=None):
    config_path = STATE/f"{config['run_name']}.json"
    write(config_path, config)
    seconds = min(time_left(), limit) if limit is not None else time_left()
    if seconds <= 0:
        raise TimeoutError("Retrieval reserve reached")
    with (STATE/f"{config['run_name']}.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.cloud", "worker", str(config_path)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if process.wait(timeout=seconds):
                raise RuntimeError(f"Run failed; inspect {log.name}")
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"Time budget exhausted: {config['run_name']}")
    output = Path(config["output_dir"])
    run = read(output/"run.json")
    expected = next(m["expected_trainable_parameters"] for m in read("experiments/cloud-plan.json")["models"].values()
                    if m["model_path"] == config["model_path"])
    assert run["trainable_parameters"] == expected
    selected = output/f"checkpoint-{run['selected_step']}"/"adapter_model.safetensors"
    assert hashlib.sha256(selected.read_bytes()).digest() == hashlib.sha256((output/"adapter_model.safetensors").read_bytes()).digest()
    return output


def estimate_pair(benchmarks, steps):
    estimate = 0
    for row in benchmarks:
        durations = sorted(row["step_seconds"][2:] or row["step_seconds"])
        step = durations[min(len(durations)-1, int(len(durations)*.9))]
        # One-page benchmark extrapolated conservatively to all 29 validation pages.
        evaluation = max(row["evaluation_seconds"])*29
        estimate += steps*step + (steps//500+2)*evaluation + 120
    return 1.5*estimate


def report(outputs):
    from voynich.evaluate import paired_interval
    rows = []
    for path in outputs:
        run, adapted = read(path/"run.json"), read(path/"adapted.json")
        dataset = Path(run["config"]["dataset"]).name
        baseline = read(f"artifacts/results/{dataset}.json")["models"]["copy"]
        rows.append(dict(path=str(path), model=run["config"]["model_path"], dataset=dataset,
                         selected_step=run["selected_step"], updates=run["optimizer_steps"],
                         trainable_parameters=run["trainable_parameters"],
                         bpc=adapted["summary"]["overall"]["bits_per_character"],
                         vs_copy=paired_interval(baseline["pages"], adapted["pages"])))
    comparisons = {}
    for dataset in {row["dataset"] for row in rows}:
        pair = [r for r in rows if r["dataset"] == dataset]
        if len(pair) == 2:
            comparisons[dataset] = paired_interval(read(Path(pair[0]["path"])/"adapted.json")["pages"],
                                                   read(Path(pair[1]["path"])/"adapted.json")["pages"])
    write("experiments/cloud-results.json", dict(results=rows, larger_vs_smaller=comparisons, test_scored=False))
    lines = ["# Cloud scaling results", "", "Validation only. Positive paired differences favor the larger model.", "",
             "| Model | Dataset | Updates | Selected | BPC |", "|---|---|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['dataset']} | {r['updates']} | {r['selected_step']} | {r['bpc']:.4f} |" for r in rows]
    for dataset, comparison in sorted(comparisons.items()):
        low, high = comparison["interval_95"]
        lines += ["", f"{dataset}: larger-model gain {comparison['delta_bits_per_character']:.4f} bits/character; "
                  f"paired 95% folio interval [{low:.4f}, {high:.4f}]."]
    lines += ["", "Adapters use 1,245,184 parameters (1.7B) and 1,212,416 (8B).",
              "Single seed, validation-selected checkpoints: exploratory evidence, not translation."]
    Path("experiments/CLOUD_RESULTS.md").write_text("\n".join(lines)+"\n")
    return comparisons


def suite():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from voynich.windows import encode_document
    plan = read("experiments/cloud-plan.json")
    if (STATE/"status.json").exists():
        raise ValueError("Cloud suite already started; review saved results instead of overwriting them")
    watch = read(STATE/"watchdog.json")
    os.kill(watch["pid"], 0)
    if watch["deadline"] != read(STATE/"budget.json")["deadline"]:
        raise ValueError("Watchdog deadline mismatch")
    outputs = []
    try:
        tokenizations = []
        for size, item in plan["models"].items():
            write(STATE/"status.json", dict(stage="download", model=size, at=time.time()))
            snapshot_download(item["model_path"], revision=item["revision"],
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            tokenizer = AutoTokenizer.from_pretrained(item["model_path"], revision=item["revision"], local_files_only=True)
            tokenizations.append([[encode_document(tokenizer, d["text"]) for d in read(f"artifacts/data/{ds}/documents.json")]
                                  for ds in ("gc", "gc-shuffle")])
        if tokenizations[0] != tokenizations[1]:
            raise ValueError("Tokenizers differ; redesign matched exposure before training")
        write(STATE/"tokenizer-check.json", dict(identical_ids_masks_units=True, test_included=False))
        benchmarks = []
        deadline = time.time()+600
        for size in plan["models"]:
            write(STATE/"status.json", dict(stage="benchmark", model=size, at=time.time()))
            path = execute(config_for(size, "gc", 30, benchmark=True), deadline-time.time())
            benchmarks.append(read(path/"timing.json"))
        candidates = [n for n in range(500,plan["max_steps"]+1,500) if estimate_pair(benchmarks,n) < time_left()]
        if not candidates:
            raise TimeoutError("A matched 500-update pair will not fit")
        steps = max(candidates)
        write(STATE/"schedule.json", dict(steps=steps, projected_pair_seconds=estimate_pair(benchmarks,steps),
                                          benchmarks=benchmarks, selected_before_primary_scores=True))
        for size in plan["models"]:
            write(STATE/"status.json", dict(stage="primary", model=size, steps=steps, at=time.time()))
            outputs.append(execute(config_for(size,"gc",steps)))
            comparisons = report(outputs)
        if comparisons["gc"]["interval_95"][0] > 0 and estimate_pair(benchmarks,steps) < time_left():
            for size in plan["models"]:
                write(STATE/"status.json", dict(stage="shuffle", model=size, steps=steps, at=time.time()))
                outputs.append(execute(config_for(size,"gc-shuffle",steps)))
                report(outputs)
        write(STATE/"status.json", dict(stage="complete", outputs=list(map(str,outputs)), at=time.time(),
                                        control_complete=len(outputs)==4))
    except Exception as exc:
        write(STATE/"status.json", dict(stage="stopped", error=str(exc), at=time.time()))
        raise
    finally:
        with tarfile.open("artifacts/cloud-results.tar.gz", "w:gz") as archive:
            for folder in (STATE, Path("training_run_outputs")):
                if folder.exists():
                    archive.add(folder, arcname=str(folder))
            for name in ("cloud-results.json", "CLOUD_RESULTS.md"):
                path = Path("experiments")/name
                if path.exists():
                    archive.add(path, arcname=str(path))
        print("Results ready in artifacts/cloud-results.tar.gz. Retrieve, then stop the pod.", flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("bundle", "watchdog", "suite"):
        commands.add_parser(name)
    arm_parser = commands.add_parser("arm")
    arm_parser.add_argument("--started-at", type=float, required=True)
    arm_parser.add_argument("--hourly", type=float, required=True)
    commands.add_parser("worker").add_argument("config")
    args = parser.parse_args()
    if args.command == "arm":
        print(json.dumps(arm(args.started_at,args.hourly),indent=2))
    elif args.command == "worker":
        worker(args.config)
    elif args.command == "suite":
        STATE.mkdir(parents=True, exist_ok=True)
        with (STATE/"suite.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            suite()
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
