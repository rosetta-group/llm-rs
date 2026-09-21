"""Save a compact, reviewable record of the local validation experiments."""

import json
from pathlib import Path

from voynich.data import digest, write_json
from voynich.evaluate import paired_interval


def read(path):
    return json.loads(Path(path).read_text())


def main():
    names = ("gc", "gc-shuffle", "gc-merged", "gc-separate", "gc-quire", "zl", "timm", "naibbe")
    baselines = {name: read(f"artifacts/results/{name}.json") for name in names}
    runs = {}
    for selection in ("outer", "middle", "random"):
        path = Path(f"training_run_outputs/qwen-{selection}-c64-s42-n400")
        run = read(path/"run.json")
        if run.get("status") != "complete":
            raise ValueError(f"Incomplete experiment: {path}")
        frozen, adapted = read(path/"frozen.json"), read(path/"adapted.json")
        config = {k: v for k, v in run["config"].items()
                  if k not in {"model_path", "frozen_reference", "logging_dir"}}
        runs[selection] = dict(config=config, layers=run["layers"], versions=run["versions"],
            trainable_parameters=run["trainable_parameters"], optimizer_steps=run["optimizer_steps"],
            dataset_sha256=run["dataset_sha256"], code_sha256=run.get("code_sha256"),
            frozen=frozen["summary"], adapted=adapted["summary"], pages=adapted["pages"],
            vs_frozen=paired_interval(frozen["pages"], adapted["pages"]),
            vs_copy=paired_interval(baselines["gc"]["models"]["copy"]["pages"], adapted["pages"]))
    outer = runs["outer"]
    for selection in ("middle", "random"):
        runs[selection]["vs_outer"] = paired_interval(outer["pages"], runs[selection]["pages"])
    result = dict(date="2026-09-20", split="validation", test_scored=False,
        model="mlx-community/Qwen3-1.7B-bf16", revision="9cd6692855d3e06772228e9a962b2606359b2d24",
        hardware="Apple Silicon, 36 GiB unified memory, MPS", seeds_run=[42],
        source_manifest_sha256=digest("experiments/sources.json"),
        baselines={name: dict(dataset_sha256=r["dataset_sha256"],
                             models={m: v["summary"] for m, v in r["models"].items()})
                   for name, r in baselines.items()}, qwen=runs,
        copy_vs_frequency=paired_interval(baselines["gc"]["models"]["frequency"]["pages"],
                                         baselines["gc"]["models"]["copy"]["pages"]))
    write_json("experiments/results.json", result)
    lines = ["# Local validation results", "", "2026-09-20. All experiments used local hardware. The final test set remains sealed.", "",
        "BPC is bits per normalized transcription character; lower is better. These are prediction results, not translations.", "",
        "## Baselines", "", "| Dataset | Frequency | 3-char context | 5-char context | Layout | Copy |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, data in baselines.items():
        values = [data["models"][m]["summary"]["overall"]["bits_per_character"]
                  for m in ("frequency", "ngram3", "ngram5", "layout", "copy")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    gc = baselines["gc"]["models"]
    lines += ["", "Compare models within a row. Transcriptions, merged boundaries, and synthetic controls have different target strings.", "",
        f"GC copy accuracy: {gc['copy']['summary']['overall']['accuracy']:.2%}; frequency accuracy: "
        f"{gc['frequency']['summary']['overall']['accuracy']:.2%}. Local spelling and copying suffice for this gain over the frequency baseline.", "",
        "## Qwen3-1.7B", "",
        "400 optimizer steps per selection; seed 42; rank 8; four layers; 1,245,184 trainable parameters each.",
        "Training batch 1, accumulation 2, learning rate 0.0001, context 64, stride 32, BF16 on MPS.",
        "Each run sees about 0.27 epochs. The frozen model is shared; all 29 validation pages are scored.", "",
        "| Layers | Frozen BPC | Adapted BPC | Gain over copy, 95% folio interval |",
        "|---|---:|---:|---|" ]
    for selection, run in runs.items():
        delta = run["vs_copy"]
        lo, hi = delta["interval_95"]
        lines.append(f"| {selection} {run['layers']} | {run['frozen']['overall']['bits_per_character']:.4f} | "
            f"{run['adapted']['overall']['bits_per_character']:.4f} | "
            f"{delta['delta_bits_per_character']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    best = min(runs, key=lambda name: runs[name]["adapted"]["overall"]["bits_per_character"])
    lines += ["", f"The best Qwen selection in this pilot is **{best}**. One seed cannot establish a layer-location effect."]
    lines += ["", "Positive gain favors Qwen. Intervals use 2,000 paired draws over 15 validation folio groups.",
        "They do not include variation across training seeds. Qwen token accuracy uses subwords and cannot be compared with character accuracy.", "",
        "## Limits and next experiments", "",
        "```text", "Compare five seeds at matched training budgets", "Evaluate the same adapters at contexts 16, 64, and 256",
        "Repeat neural comparisons on shuffled text, other transcriptions, and published controls",
        "If a gain survives those checks: freeze choices and release the test set",
        "Otherwise: revise the representation or model before testing meaning", "```", "",
        "- GC uses 148 training, 29 validation, and 30 test paragraph pages. Validation has 22 A and 7 B pages; the mix is uneven.",
        "- The quire split has only three validation quires. Its uncertainty will be weak.",
        "- Baseline robustness is measured above. Neural robustness and the five-seed matrix are not yet run.",
        "- Naibbe and Timm each use one published sample with chronological block splits and one omitted block between splits. This is not replication over generator seeds.",
        "- ZL3b comes from a pinned mirror because the publisher returned HTTP 406; it was not byte-verified against the publisher.",
        "- BF16 evaluation can vary slightly with batch shape. Outer evaluation used batch 2; middle/random use batch 1 and reuse the same frozen scores.",
        "- Learned BPE was fitted on training pages and passed a two-step random-model smoke run. It has not been attached to pretrained embeddings.",
        "- No image association, plaintext recovery, SAE experiment, or claimed Voynich translation has been run.", "",
        "## Reproduce", "", "```sh", ".venv/bin/python -m experiments.baselines",
        ".venv/bin/python -m experiments.results", "```", "",
        "The second command requires the three completed `qwen-*-c64-s42-n400` runs. Their settings and per-page scores are preserved in `results.json`.",
        "Generate fresh training configs with `python -m voynich matrix MODEL --contexts 64 --steps 400`; set accumulation to 2 to match these pilots.",
        "See `../README.md` for training and comparison commands and source attribution.", ""]
    Path("experiments/RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
