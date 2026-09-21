"""Illustrated progress/final report for the matched-context suite."""

import json
import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from voynich.data import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/report-matched"
COLORS = {42: "#176B87", 43: "#C76423", 44: "#32947C"}


def read(path):
    return json.loads((ROOT / path).read_text())


def capture():
    data = read("experiments/matched-results.json")
    manifest = read("artifacts/matched/manifest.json")
    if data["test_scored"] or manifest["test_scored"]:
        raise ValueError("Only validation reports are allowed")
    if any(digest(ROOT / p) != sha for p, sha in manifest["input_sha256"].items()):
        raise ValueError("Frozen experiment input changed")
    curves, sources = {}, {}
    for seed in data["plan"]["seeds"]:
        for context in data["plan"]["contexts"]:
            for dataset in data["plan"]["datasets"]:
                run = f"training_run_outputs/matched-{dataset}-c{context}-s{seed}"
                path = ROOT / run / "learning_curve.json"
                if path.exists():
                    raw = path.read_bytes()
                    curves[run] = json.loads(raw)
                    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    for row in data["results"]:
        run = read(row["path"] + "/run.json")
        if (run["status"] != "complete" or not run["checkpoint_verified"]
                or digest(ROOT / row["path"] / "last.pt") != run["checkpoint_sha256"]):
            raise ValueError("Unverified final checkpoint")
        if run["training_characters_seen"] != 2511320:
            raise ValueError("Training exposure mismatch")
    data.update(captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                curves=curves, curve_sha256=sources, manifest=manifest)
    return data


def figures(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"{name}.{suffix}", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), layout="constrained", sharex=True)
    for i, dataset in enumerate(data["plan"]["datasets"]):
        for j, context in enumerate(data["plan"]["contexts"]):
            ax = axes[i, j]
            found = False
            for seed in data["plan"]["seeds"]:
                curve = data["curves"].get(f"training_run_outputs/matched-{dataset}-c{context}-s{seed}", [])
                if not curve:
                    continue
                found = True
                for key, style in [("training_bpc", "--"), ("validation_bpc", "-")]:
                    ax.plot([r["epoch"] for r in curve], [r[key] for r in curve],
                            color=COLORS[seed], linestyle=style, linewidth=1.5,
                            label=f"Seed {seed}" if style == "-" else None)
            if not found:
                ax.text(.5, .5, "Awaiting training", ha="center", va="center", transform=ax.transAxes)
            ax.set(title=f"{'Intact' if dataset == 'gc' else 'Shuffled'} · {context} characters",
                   xlim=(1, 20), xticks=[1, 5, 10, 15, 20])
            if i == 1:
                ax.set_xlabel("Complete passes over training targets")
            if j == 0:
                ax.set_ylabel("Bits per character")
            if found:
                ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("Learning curves: validation solid, training dashed\n"
                 "Training includes dropout; validation does not. Panel scales differ.", fontsize=12)
    save(fig, "learning")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained", sharey=True)
    for ax, dataset in zip(axes, data["plan"]["datasets"]):
        for seed in data["plan"]["seeds"]:
            rows = sorted((r for r in data["results"] if r["dataset"] == dataset and r["seed"] == seed),
                          key=lambda r: r["context"])
            if rows:
                ax.plot([r["context"] for r in rows], [r["bpc"] for r in rows], "o-",
                        color=COLORS[seed], label=f"Seed {seed}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 128], [8, 32, 128])
        ax.set_xlim(6, 160)
        ax.set(title="Intact Voynich" if dataset == "gc" else "Shuffled Voynich",
               xlabel="Trained and evaluated history (characters)")
        ax.grid(axis="y", alpha=.2)
        if ax.lines:
            ax.legend(fontsize=9)
    axes[0].set_ylabel("Epoch-20 validation BPC (lower is better)")
    fig.suptitle(f"Matched exposure: {len(data['results'])}/18 completed runs")
    save(fig, "final-scores")

    if data["contrasts"]:
        fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
        for i, row in enumerate(data["contrasts"]):
            value = row["extra_intact_gain_bpc"]
            lo, hi = row["interval_95"]
            ax.errorbar(value, i, xerr=[[value - lo], [hi - value]], fmt="o",
                        color=COLORS[row["seed"]], capsize=5)
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set(yticks=range(len(data["contrasts"])),
               yticklabels=[f"Seed {r['seed']}" for r in data["contrasts"]],
               xlabel="Extra intact 8-to-128 gain (BPC); positive favors intact text",
               title="Order-related context gain: 95% source-folio intervals")
        ax.set_ylim(-.6, len(data["contrasts"]) - .4)
        ax.invert_yaxis()
        save(fig, "interaction")


def report(data):
    count = len(data["results"])
    state = "Complete" if data["complete"] else "In progress"
    lines = ["# Matched-context experiment", "", f"{state}: **{count}/18 runs**. Snapshot: {data['captured_at']}.", "",
             "The goal is validated meaning recovery and translation. This experiment checks whether a model's "
             "apparent dependence on long history survives training it specifically for shorter histories.", "",
             "**BPC:** bits per scored character; lower is better. **Matched exposure:** every final model has "
             "seen 2,511,320 training targets. **Overfitting:** training prediction improves while validation prediction worsens.", "",
             "## What was done", "", "- Launched 18 fresh character GRU runs with exact 8-, 32-, or 128-character histories; completion is shown above.",
             "- Fixed the comparison to intact and within-line shuffled text across three seeds.",
             "- Preserved the fixed epoch-20 primary comparison, best-epoch diagnostics, and full learning curves.",
             "- Used local hardware and kept final-test pages sealed.", "", "## Why it was done", "",
             "The previous experiment shortened the history of a model trained on longer windows. "
             "This experiment removes that training/evaluation mismatch, but does not guarantee equal convergence or overfitting.", "",
             "## Results", "", "![Fixed final-epoch scores](figures/final-scores.png)", "",
             "Only completed, checkpoint-verified runs appear in this graph. Pending conditions are not inferred.", "",
             "| Data | Seed | Characters | Epoch-20 BPC | Best BPC (secondary) | Best epoch |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in data["results"]:
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['context']} | {r['bpc']:.4f} | "
                     f"{r['best_bpc']:.4f} | {r['best_epoch']} |")
    lines += ["", "![Learning curves](figures/learning.png)", "",
              "Training loss is accumulated as weights change with dropout enabled. Validation uses the epoch-end "
              "weights with dropout disabled. Their absolute levels are not a perfectly controlled comparison; "
              "a sustained rise in validation loss after its minimum is the relevant overtraining diagnostic.", ""]
    overfit = [r for r in data["results"] if r["bpc"] - r["best_bpc"] > .02]
    if overfit:
        lines += [f"**Observed:** {len(overfit)} completed runs finish more than 0.02 BPC worse than their best "
                  "validation epoch. This descriptive threshold flags curves for inspection; it is not a selection rule. "
                  "Do not interpret the final-score difference as a pure measure of how much history the text needs.", ""]
    lines += ["## Primary interaction", "",
              "Extra intact gain = (intact BPC at 8 − intact BPC at 128) − "
              "(shuffled BPC at 8 − shuffled BPC at 128). Positive means longer history helps intact text more.", "",
              "| Seed | Extra intact gain (BPC) | 95% source-folio interval |", "|---:|---:|---|"]
    for r in data["contrasts"]:
        lo, hi = r["interval_95"]
        lines.append(f"| {r['seed']} | {r['extra_intact_gain_bpc']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    if len(data["contrasts"]) == 3:
        xs = [r["extra_intact_gain_bpc"] for r in data["contrasts"]]
        lines += ["", f"Across seeds: mean {statistics.mean(xs):.4f} BPC; range [{min(xs):.4f}, {max(xs):.4f}]."]
        means = {dataset: statistics.mean(r["delta_bits_per_character"] for r in data["gains"]
                                         if r["dataset"] == dataset) for dataset in ("gc", "gc-shuffle")}
        lines += ["", "![Context-gain interaction](figures/interaction.png)", "",
                  f"Mean 8-to-128 gain is **{means['gc']:.4f} BPC for intact text** and "
                  f"**{means['gc-shuffle']:.4f} for shuffled text**. One seed favors shuffled text; "
                  "two have intervals spanning zero. This experiment did not establish an extra long-history "
                  "advantage for intact text. These three seeds share the same corpus and split; they are "
                  "training replications, not independent manuscript samples.", "",
                  "All 18 models reach their best validation epoch between passes 5 and 9, then worsen "
                  "before the fixed pass-20 comparison. The fixed budget is auditable, but it does not "
                  "equalize convergence or overfitting. We retain the primary result and its limitation; "
                  "we do not replace it with a more favorable checkpoint selection."]
    else:
        lines += ["", "The three-seed interaction is not complete. No overall finding is declared yet."]
    lines += ["", "Each within-text contrast checks identical targets. The interaction resamples matching source "
              "folios together across original and shuffled text; their strings are not treated as identical. "
              "Intervals use 2,000 draws across 15 folio groups. They do not capture every source of model or design uncertainty.", "",
              "## Implication for translation", "",
              "A stable positive interaction would identify an order-related prediction effect, not a word meaning. "
              "A null interaction would not rule out meaningful text. Overfitting or incomplete training limits either interpretation. "
              "The [first controlled recovery benchmark](../decipherment/REPORT.md) is now complete: spaced substitution "
              "recovered all normalized Italian words under declared cipher assumptions. Space-free word recovery remains poor. "
              "The proposed next semantic work is better segmentation tested on fresh passages, then fewer supplied cipher hints. "
              "We should not keep extending the prediction track indefinitely.", "",
              "## Records", "", "- [Overall goal and complete research record](../../RESEARCH_LOG.md)",
              "- [Fixed protocol](../MATCHED_PLAN.md)", "- [Current numerical results](../MATCHED.md)",
              "- [Snapshot used for these figures](snapshot.json)",
              "- [Final checkpoint, exposure, input-hash, and resource verification](verification.json)",
              "- Regenerate with `python -m experiments.matched_report` in an environment with matplotlib and NumPy.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    data = capture()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshot.json").write_text(json.dumps(data, indent=2) + "\n")
    figures(data)
    report(data)
