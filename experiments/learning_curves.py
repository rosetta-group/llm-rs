"""Report completed validation checkpoints without opening the final test set."""

import argparse
import json
from pathlib import Path

from voynich.data import write_json
from voynich.evaluate import paired_interval


def read(path):
    return json.loads(Path(path).read_text())


def report(paths, output="experiments"):
    baseline = read("artifacts/results/gc.json")["models"]["copy"]
    results = {}
    for path in paths:
        path = Path(path)
        if not (path/"run.json").exists():
            results[path.name] = {"status": "queued", "checkpoints": []}
            continue
        run = read(path/"run.json")
        checkpoints = []
        for file in sorted((path/"validation").glob("step-*.json")):
            score = read(file)
            checkpoints.append(dict(step=score["step"], **score["summary"]["overall"],
                                    vs_copy=paired_interval(baseline["pages"], score["pages"])))
        entry = dict(status=run["status"], layers=run["layers"], seed=run["config"]["seed"],
                     parameters=run["trainable_parameters"], dataset_sha256=run["dataset_sha256"],
                     code_sha256=run["code_sha256"], checkpoints=checkpoints,
                     selected_step=run.get("selected_step"))
        if checkpoints:
            entry["best_so_far"] = min(checkpoints, key=lambda row: row["bits_per_character"])
        if run["status"] == "complete":
            entry["selected_scores"] = read(path/"adapted.json")
        results[path.name] = entry
    output = Path(output)
    result = dict(test_scored=False, baseline=baseline["summary"]["overall"], runs=results)
    write_json(output/"learning-curve-results.json", result)
    lines = ["# Qwen learning curves", "", "Local hardware only. Validation pages only. Final test remains sealed.", "",
             "Fresh adapters; seed 42; context 64; stride 32; rank 8; batch 1; accumulation 2.",
             "Each run has a 3,000-update budget and its own cosine learning-rate schedule starting at 0.0001.",
             "Validation and saving occur every 500 updates. The adapter exported at the run root is the best validation checkpoint.", "",
             f"Copy baseline: **{baseline['summary']['overall']['bits_per_character']:.4f} bits/character**. Lower is better.", "",
             "| Run | Status | Update | Bits/character | Gain over copy, 95% interval |",
             "|---|---|---:|---:|---|"]
    for name, run in results.items():
        for checkpoint in run["checkpoints"]:
            delta = checkpoint["vs_copy"]
            lo, hi = delta["interval_95"]
            marker = " **selected**" if checkpoint["step"] == run.get("selected_step") else ""
            lines.append(f"| {name} | {run['status']} | {checkpoint['step']}{marker} | "
                         f"{checkpoint['bits_per_character']:.4f} | "
                         f"{delta['delta_bits_per_character']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        if not run["checkpoints"]:
            lines.append(f"| {name} | {run['status']} | — | — | — |")
    lines += ["", "Positive gain favors Qwen. Intervals resample 15 validation folio groups, not training seeds.",
              "Selecting the best checkpoint on these same pages makes the intervals exploratory; they do not establish a final-test gain.",
              "The older 400-update pilots used a shorter learning-rate schedule, so their endpoints are separate experiments.", "",
              "Regenerate this report with `python -m experiments.learning_curves`.", ""]
    Path(output/"LEARNING_CURVES.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=[
        "training_run_outputs/qwen-random-c64-s42-n3000",
        "training_run_outputs/qwen-outer-c64-s42-n3000"])
    args = parser.parse_args()
    result = report(args.runs)
    for name, run in result["runs"].items():
        best = run.get("best_so_far", {})
        print(name, run["status"], "best step", best.get("step"), "BPC", best.get("bits_per_character"))


if __name__ == "__main__":
    main()
