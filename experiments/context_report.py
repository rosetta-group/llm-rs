"""Plot the frozen context experiment; report dependencies include matplotlib."""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "experiments/context-figures"
LABELS = {"gc": "Voynich", "gc-shuffle": "Shuffled Voynich",
          "timm": "Timm sample", "naibbe": "Naibbe ciphertext"}
COLORS = {"gc": "#176B87", "gc-shuffle": "#C76423", "timm": "#6D5C91", "naibbe": "#32947C"}


def bpc(row):
    return row["summary"]["overall"]["bits_per_character"]


def curves(data):
    for case in data["manifest"]["plan"]["cases"]:
        rows = [r for r in data["results"] if r["dataset"] == case["dataset"]
                and r["seed"] == case["seed"]]
        yield case, sorted(rows, key=lambda r: r["context"])


def plot(data):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    for gain in (False, True):
        fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
        for case, rows in curves(data):
            name, seed = case["dataset"], case["seed"]
            label = LABELS[name] + (f" (seed {seed})" if name == "gc" else " (seed 42)")
            ys = [bpc(rows[0]) - bpc(r) if gain else bpc(r) for r in rows]
            ax.plot([r["context"] for r in rows], ys, marker="o", markersize=4,
                    color=COLORS[name], linestyle={42: "-", 43: "--", 44: ":"}[seed],
                    linewidth=1.6, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128], [8, 16, 32, 64, 128])
        ax.set_xlabel("Maximum preceding transcription characters per prediction")
        ax.set_ylabel("BPC saved versus 8 characters (higher is better)" if gain
                      else "Validation bits per character (lower is better)")
        ax.set_title("How much does extra context help?" if gain else "Prediction with shorter memory")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.23), frameon=False)
        name = "context-gain" if gain else "context-loss"
        for extension in ("png", "svg"):
            fig.savefig(FIG / f"{name}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def report(data):
    mean = {c: statistics.mean(bpc(r) for r in data["results"]
                              if r["dataset"] == "gc" and r["context"] == c)
            for c in data["manifest"]["plan"]["contexts"]}
    full_gain = mean[8] - mean[128]
    fraction32 = (mean[8] - mean[32]) / full_gain
    fraction64 = (mean[8] - mean[64]) / full_gain
    shuffled_gain = next(r["gain_to_128_bpc"] for r in data["comparisons"]
                         if r["dataset"] == "gc-shuffle" and r["context"] == 8)
    lines = ["# Voynich memory test", "", "Pattern: **context ablation**. Shorten the history of a fixed model to measure how its predictions change.", "",
             f"Across three Voynich GRUs, mean loss changes from {mean[8]:.4f} BPC at 8 characters "
             f"to {mean[128]:.4f} at 128; 64 characters recover {fraction64:.1%} of that measured improvement.", "",
             "**Context:** preceding normalized transcription characters, including spaces and line breaks.",
             "**BPC:** bits per scored character; lower means better prediction.",
             "**GRU:** the small recurrent predictor already trained on each dataset.", "",
             "```text", "Load each saved GRU; verify weights and dataset hashes",
             "For each context in 8, 16, 32, 64, 128:",
             "    Reset state before every validation target",
             "    Read at most that many preceding symbols from its own page",
             "    Predict the same next character; save loss by page",
             "Compare context gains within each dataset", "Keep final-test pages sealed", "```", "",
             "## What was done", "",
             "- Evaluated six frozen GRUs at five context lengths: 30 evaluations on the local Apple GPU.",
             "- Used three training seeds for intact Voynich and one each for shuffled Voynich, Timm, and Naibbe.",
             "- Verified source weights and inputs before and after scoring; checked exact target hashes against previous models and baselines.",
             "- Saved per-page scores, folio bootstrap intervals, and two graphs. No retraining, downloads, or paid compute.", "",
             "## Why it was done", "",
             "The earlier GRU result showed strong prediction without language pretraining. "
             "This experiment asks how much of that prediction depends on the nearby text versus a longer history.", "",
             "## Results", "",
             "![Validation loss by context](context-figures/context-loss.png)", "",
             "The curves show each model on its own dataset. Absolute scores across the four different texts "
             "are not a model ranking or a test of meaning.", "",
             "| Data | Seed | 8 | 16 | 32 | 64 | 128 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for case, rows in curves(data):
        lines.append(f"| {LABELS[case['dataset']]} | {case['seed']} | "
                     + " | ".join(f"{bpc(r):.4f}" for r in rows) + " |")
    lines += ["", "![Improvement from extra context](context-figures/context-gain.png)", "",
              "Each curve subtracts its own 8-character loss. This is a descriptive comparison: "
              "the texts, selected checkpoints, and training histories differ.", "",
              "1. **Amount of useful history.** On intact Voynich, extending 32 to 128 characters saves "
              f"{mean[32] - mean[128]:.4f} BPC on average; extending 64 to 128 saves {mean[64] - mean[128]:.4f}. "
              f"The 8-to-32 change recovers {fraction32:.1%} of the measured 8-to-128 improvement; "
              f"8-to-64 recovers {fraction64:.1%}. This fraction is not a share of all manuscript structure.",
              "2. **Control comparison.** Extending 8 to 128 characters saves "
              f"{full_gain:.4f} BPC for Voynich (three-seed mean) and {shuffled_gain:.4f} for shuffled Voynich "
              "(one seed). Much of the context benefit survives word shuffling within lines. "
              "The Timm sample gains only 0.0182 BPC. None of these differences identifies meaning.",
              "3. **Training mismatch.** Naibbe worsens from 1.8280 BPC at 8 characters to 2.3321 at 32, "
              "then improves to 1.7361 at 128. The original evaluator independently reproduces this curve. "
              "These GRUs were trained with 128-character windows and stride 64; most scored training targets "
              "had 65–128 preceding symbols. A state reset after only 8–32 symbols changes that setting. "
              "The curve cannot separate distant information from the effect of giving the recurrent state more warm-up.", "",
              "## Uncertainty", "",
              "Positive gain means the 128-character condition predicts better. Paired 95% percentile bootstrap "
              "intervals resample 15 folio groups, with 2,000 draws and seed 42. They describe page sampling, "
              "not training-seed variation or uncertainty from choosing checkpoints on validation.", "",
              "| Data | Seed | Context change | Gain (BPC) | 95% folio interval |",
              "|---|---:|---|---:|---|"]
    for r in data["comparisons"]:
        if r["context"] not in (32, 64) or "paired_folio_interval" not in r:
            continue
        low, high = r["paired_folio_interval"]["interval_95"]
        lines.append(f"| {LABELS[r['dataset']]} | {r['seed']} | {r['context']} → 128 | "
                     f"{r['gain_to_128_bpc']:.4f} | [{low:.4f}, {high:.4f}] |")
    lines += ["", "Timm and Naibbe each use one published sample split into chronological blocks; "
              "those blocks are not independent generator replications. No folio intervals are attached to them. "
              "Cross-text differences are not paired as if they were identical prediction targets.", "",
              "## Exact evaluation protocol", "",
              "- Every readable validation character is scored once (stride 1). `?` stays in context but is never a scored target.",
              "- Page starts have less available history. A beginning-of-page symbol occupies one context slot where present. No state crosses a page boundary.",
              "- GRU weights, checkpoints, alphabet, text, and target masks stay fixed within each curve. Dropout is disabled.",
              "- The new 128-character score is recomputed at stride 1. Earlier reports used stride 64, which gave targets varying history lengths; their 2.332 mean is a different evaluation setting.",
              "- GC2a uses the v101 transcription. Results concern this representation and validation split; they do not establish a translation.",
              "- All choices remain exploratory. Final-test pages have not been scored.", "",
              "## Next experiment", "",
              "Before treating the context curve as a property of the manuscript, test the training mismatch. "
              "Train small GRUs with histories matched to the evaluated 8, 32, and 128 characters, "
              "equal scored-character exposure, and repeated seeds on intact and shuffled Voynich. "
              "This is proposed work; it has not been run. It tests whether shorter histories remain limiting "
              "when the model has learned to use them.", "",
              "## Reproduce", "", "```sh", ".venv/bin/python -m unittest tests.test_context -v",
              ".venv/bin/python -m experiments.context", ".venv/bin/python -m experiments.context_verify",
              "python -m experiments.context_report", "```", "",
              "Run from the repository root. The evaluator requires local PyTorch/MPS and existing saved checkpoints; "
              "it refuses to overwrite `artifacts/context-ablation`. The report command needs matplotlib and reads "
              "the existing results without evaluating models.", "",
              "- [Fixed plan](context-plan.json)", "- [Frozen results and input hashes](context-results.json)",
              "- [Independent evaluator cross-check](context-verification.json)",
              "- Raw run: `artifacts/context-ablation/manifest.json`, `results.json`, and one JSON per condition.",
              "- [Previous completed experiments](report-completed/REPORT.md)", ""]
    (ROOT / "experiments/CONTEXT.md").write_text("\n".join(lines))


def main():
    raw = ROOT / "artifacts/context-ablation/results.json"
    data = json.loads(raw.read_text())
    if not data["inputs_verified"] or data["test_scored"] or len(data["results"]) != 30:
        raise ValueError("Need the complete verified validation experiment")
    verification = ROOT / "artifacts/context-ablation/reference-verification.json"
    checks = json.loads(verification.read_text())
    if (checks["test_scored"] or len(checks["checks"]) != 6
            or any(c["absolute_bpc_error"] >= 1e-5 or not c["targets_match"] for c in checks["checks"])):
        raise ValueError("Independent evaluator cross-check failed")
    (ROOT / "experiments/context-verification.json").write_bytes(verification.read_bytes())
    (ROOT / "experiments/context-results.json").write_bytes(raw.read_bytes())
    plot(data)
    report(data)


if __name__ == "__main__":
    main()
