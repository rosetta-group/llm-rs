"""Frozen report for completed Qwen, cloud, and character experiments."""

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate

from experiments import research_report as layout

ROOT = layout.ROOT
OUT = ROOT / "experiments/report-completed"
FIG = OUT / "figures"
PDF = ROOT / "output/pdf/voynich-completed-experiments.pdf"
BLUE, TEAL, ORANGE, GRAY = layout.BLUE, layout.TEAL, layout.ORANGE, layout.GRAY


def capture():
    paths = ["experiments/replication-results.json", "experiments/character-results.json",
             "experiments/cloud-results.json", "experiments/character-plan.json",
             "experiments/cloud-plan.json", "artifacts/final-verification.json"]
    qwen, chars, cloud = [layout.read(p) for p in paths[:3]]
    assert not any(d["test_scored"] for d in (qwen, chars, cloud))
    curves, runs = {}, {}
    for row in qwen["results"] + chars["results"] + cloud["results"]:
        root = row["path"]
        run = layout.read(f"{root}/run.json")
        assert run["status"] == "complete"
        paths += [f"{root}/run.json", f"{root}/adapted.json"]
        if root.startswith("training_run_outputs/char-"):
            paths += [f"{root}/learning_curve.json", f"{root}/best.pt"]
            curves[root] = layout.read(f"{root}/learning_curve.json")
        runs[root] = {k: run[k] for k in ("status", "selected_step", "selected_epoch", "epoch",
                     "wall_seconds", "parameters", "trainable_parameters", "training_characters_seen",
                     "training_characters_per_epoch", "training_windows", "optimizer_steps") if k in run}
    return dict(captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                qwen=qwen["results"], characters=chars["results"], cloud=cloud,
                curves=curves, runs=runs, input_sha256={p: layout.sha(p) for p in paths})


def values(data, model):
    if model == "qwen":
        return [r["bits_per_character"] for r in data["qwen"] if r["dataset"] == "gc"]
    return [r["bpc"] for r in data["characters"] if r["dataset"] == "gc" and r["model"] == model]


def control(data, dataset):
    q = next(r for r in data["qwen"] if r["dataset"] == dataset and r["seed"] == 42)
    c = {r["model"]: r for r in data["characters"] if r["dataset"] == dataset and r["seed"] == 42}
    return q, c


def plots(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#273548", "text.color": "#273548",
                         "axes.edgecolor": "#B9C3CF", "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    cloud = data["cloud"]["results"]
    labels = ["Spelling + copy", "Character transformer (3 seeds)", "Local Qwen 1.7B (5 seeds)",
              "Qwen 1.7B-Base (1 seed)", "Qwen 8B-Base (1 seed)", "Character GRU (3 seeds)"]
    vals = [control(data, "gc")[0]["baseline_bpc"], statistics.mean(values(data, "transformer")),
            statistics.mean(values(data, "qwen")), cloud[0]["bpc"], cloud[1]["bpc"],
            statistics.mean(values(data, "gru"))]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.barh(labels, vals, color=[GRAY, ORANGE, BLUE, BLUE, BLUE, TEAL], height=.62)
    ax.invert_yaxis()
    ax.set(xlabel="Validation bits per character (lower is better)", xlim=(0, 3))
    for i, value in enumerate(vals):
        ax.text(value + .035, i, f"{value:.3f}", va="center")
    save(fig, "comparison")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3), layout="constrained", sharey=True)
    for ax, kind, label, color in zip(axes, ("qwen", "transformer", "gru"),
                                     ("Local Qwen", "Character transformer", "Character GRU"),
                                     (BLUE, ORANGE, TEAL)):
        ys = values(data, kind)
        ax.scatter(range(42, 42 + len(ys)), ys, color=color, s=55)
        ax.axhline(statistics.mean(ys), color=color, linestyle="--", linewidth=1)
        ax.set(title=label, xlabel="Training seed", xticks=list(range(42, 42 + len(ys))), ylim=(2.30, 2.61))
    axes[0].set_ylabel("Validation bits/character")
    save(fig, "seeds")

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for kind, color, name in (("transformer", ORANGE, "Character transformer"), ("gru", TEAL, "Character GRU")):
        for row in [r for r in data["characters"] if r["dataset"] == "gc" and r["model"] == kind]:
            curve = data["curves"][row["path"]]
            ax.plot([v["epoch"] for v in curve], [v["validation_bpc"] for v in curve], color=color,
                    alpha=.8 if row["seed"] == 42 else .4, label=name if row["seed"] == 42 else None)
            selected = next(v for v in curve if v["epoch"] == row["selected_epoch"])
            ax.scatter(selected["epoch"], selected["validation_bpc"], color=color, s=20)
    ax.axhline(statistics.mean(values(data, "qwen")), color=BLUE, linestyle="--", label="Selected local Qwen mean")
    ax.axvline(2, color=GRAY, linestyle=":", linewidth=1)
    ax.set(xlabel="Full passes over character training targets (epochs)", ylabel="Validation bits/character",
           xlim=(1, 20), ylim=(2.30, 2.96), xticks=[1, 2, 5, 10, 15, 20])
    ax.legend(frameon=False, fontsize=9)
    save(fig, "learning")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), layout="constrained", sharey=True)
    for ax, dataset, title in zip(axes, ("gc-shuffle", "timm", "naibbe"),
                                   ("Shuffled Voynich", "Timm generated sample", "Naibbe ciphertext")):
        q, c = control(data, dataset)
        ys = [q["baseline_bpc"], q["bits_per_character"], c["transformer"]["bpc"], c["gru"]["bpc"]]
        ax.bar(range(4), ys, color=[GRAY, BLUE, ORANGE, TEAL], width=.7)
        for x, y in enumerate(ys):
            ax.text(x, y + .04, f"{y:.2f}", ha="center", fontsize=9)
        ax.set(title=title, xticks=range(4), xticklabels=["Base", "Qwen", "Trf", "GRU"], ylim=(0, 3.2))
    axes[0].set_ylabel("Validation bits/character")
    save(fig, "controls")

    item = data["cloud"]["larger_vs_smaller"]["gc"]
    point = item["delta_bits_per_character"]
    low, high = item["interval_95"]
    fig, ax = plt.subplots(figsize=(9, 1.7), layout="constrained")
    ax.errorbar(point, 0, xerr=[[point - low], [high - point]], fmt="o", color=BLUE, capsize=6)
    ax.axvline(0, color=GRAY, linestyle="--")
    ax.set(xlabel="1.7B loss - 8B loss (positive favors 8B)", yticks=[], xlim=(-.005, .022))
    ax.text(point, .055, f"{point:.4f}   [95% interval: {low:.4f}, {high:.4f}]", ha="center", fontsize=10)
    ax.set_ylim(-.09, .12)
    save(fig, "scaling")


class Report(layout.Report):
    def finish(self):
        def footer(canvas, doc):
            canvas.setStrokeColor(colors.HexColor("#DDE4EA"))
            canvas.line(48, 42, 547, 42)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor(GRAY))
            canvas.drawString(48, 29, f"VOYNICH / COMPLETED EXPERIMENTS / {self.stamp[:10]}")
            canvas.drawRightString(547, 29, str(doc.page))
        SimpleDocTemplate(str(PDF), pagesize=(595, 842), leftMargin=48, rightMargin=48,
                          topMargin=44, bottomMargin=58, title="Voynich: completed prediction experiments",
                          author="llm-rs research project").build(self.story, onFirstPage=footer, onLaterPages=footer)
        (OUT / "REPORT.md").write_text("\n".join(self.md))


def build(data):
    layout.FIG = FIG
    plots(data)
    r = Report(data["captured_at"])
    qmean = statistics.mean(values(data, "qwen"))
    gmean = statistics.mean(values(data, "gru"))
    tmean = statistics.mean(values(data, "transformer"))
    baseline = control(data, "gc")[0]["baseline_bpc"]
    r.heading("Voynich: what the completed runs tell us", True)
    r.p(f"Completed prediction experiments | Snapshot {data['captured_at']}", "Caption")
    r.p("This study compares prediction of Voynich transcription characters on fixed validation pages.")
    r.p(f"**Main finding:** a 1.32-million-parameter recurrent model trained from scratch reaches **{gmean:.3f} bits/character**, ahead of adapted Qwen and the implemented copying baseline.")
    r.p("**Bits/character (BPC):** negative log probability per scored transcription character; lower is better.")
    r.p("**Validation:** pages excluded from weight fitting but reused for selecting settings and checkpoints.")
    r.figure("comparison", "Figure 1. Same 29 GC validation pages and 19,752 scored characters. Bars show seed means where available. Training exposure and model architecture differ; these are practical scores, not translation accuracy.")
    r.p("1. **Completed work.** Five local Qwen seeds, three Qwen controls, two cloud base models, and twelve character-model runs. All selected checkpoints were verified. Final-test scores remain sealed.")
    r.p("2. **Implication.** A model without prior language training can exceed our present Qwen scores. These results establish predictive structure, not word meanings or an English/Italian translation.")

    r.heading("01 / The result repeats across seeds")
    r.p("Repeated training measures sensitivity to initialization and data order.")
    r.p("**Thesis:** Qwen's gain and the GRU's stronger score are stable across the seeds tested.")
    r.p("**Seed:** controls randomness in initialization and training order. **GRU:** a gated recurrent neural network that updates a hidden state as it reads characters.")
    r.figure("seeds", "Figure 2. Each dot is a separately trained model's selected checkpoint; dashed lines are means. The common vertical scale starts at 2.30 to make differences visible. These are training repeats on the same validation split.")
    table = [["Model", "Seeds", "Mean BPC", "Seed SD", "Range"]]
    for kind, label in (("qwen", "Local Qwen"), ("transformer", "Character transformer"), ("gru", "Character GRU")):
        ys = values(data, kind)
        table.append([label, len(ys), f"{statistics.mean(ys):.4f}", f"{statistics.stdev(ys):.4f}", f"{min(ys):.4f}-{max(ys):.4f}"])
    r.table(table, [153, 45, 80, 75, 146])
    r.p(f"1. **Baseline gain.** Local Qwen averages {qmean:.4f} against copy at {baseline:.4f}: {(1-qmean/baseline):.1%} lower loss. This is not an accuracy percentage.")
    r.p("2. **Separate uncertainties.** Seed spread measures training randomness. A folio bootstrap measures variation across manuscript leaves. Neither corrects for repeated validation-based selection.")
    r.p("3. **Layer claims remain unsupported.** Seed-42 random and outer subsets scored 2.4828 and 2.4819. The later repeats hold the outer subset fixed; they do not establish language-specific layers or preserved English/Italian semantics.")

    r.heading("02 / Training exposure changes the comparison")
    r.p("The character models fit all their weights; Qwen fits only small adapters.")
    r.p("**Thesis:** the GRU wins the practical comparison after more passes, so the result does not isolate the effect of pretraining.")
    r.p("**Epoch:** one pass over all training targets. **Adapter:** trainable changes added to a much larger frozen model.")
    r.figure("learning", "Figure 3. Character-model validation curves for three seeds. Dots mark selected checkpoints. The Qwen line is its selected five-seed mean, not an epoch-matched trajectory.")
    early = [row["epoch_2_bpc"] for row in data["characters"] if row["model"] == "gru" and row["dataset"] == "gc"]
    r.p(f"1. **Early GRU results.** At epoch 2, GRU loss ranges from {min(early):.3f} to {max(early):.3f}. Selected epochs 14, 16 and 17 yield {gmean:.3f} on average. Qwen saw about 2.04 passes over its token-window dataset; epoch 2 is only an approximate exposure reference.")
    r.p("2. **Different inputs.** Character models use context 128 and stride 64 characters. Qwen uses 64 and 32 subword tokens. Both score identical underlying target positions, but their available context is not identical.")
    r.p("3. **Similar trainable counts, different capacity.** GRU has 1,316,608 weights; the character transformer has 1,272,448. Local Qwen trains 1,245,184 adapter parameters on top of about 1.7 billion pretrained weights.")
    r.p("4. **Inference limit.** Prior language training is unnecessary to achieve the observed GRU score on this split. It may still improve data efficiency or other tasks. Neither prediction result establishes decipherment.")

    r.heading("03 / Controls weaken the meaning claim")
    r.p("Altered and generated texts test alternatives to semantic interpretation.")
    r.p("**Thesis:** neural prediction gains also occur on shuffled or synthetic text, so beating a simple baseline is not a meaning test.")
    r.p("**Control:** a text with known changes or a known construction process, evaluated against its own baseline.")
    r.figure("controls", "Figure 4. Compare models within each panel, not absolute scores across texts. Base is copy for shuffled/Timm and layout for Naibbe. Trf is the small character transformer. Every control uses seed 42.")
    table = [["Text", "Best baseline", "Qwen", "GRU", "Character transformer"]]
    for dataset, label in (("gc-shuffle", "Shuffled GC"), ("timm", "Timm sample"), ("naibbe", "Naibbe sample")):
        q, c = control(data, dataset)
        table.append([label, f"{q['baseline_bpc']:.4f}", f"{q['bits_per_character']:.4f}", f"{c['gru']['bpc']:.4f}", f"{c['transformer']['bpc']:.4f}"])
    r.table(table, [110, 100, 80, 80, 129])
    r.p("1. **Order is not required for the Qwen gain.** Qwen improves over copy by 0.2166 BPC on intact GC and 0.2442 on within-line shuffled GC. The original word order is unnecessary for a gain of this kind. This does not show that order carries no information.")
    r.p("2. **Architecture matters.** GRU wins on GC, shuffled GC and Naibbe; Qwen wins on the Timm sample. There is no single winner across every dataset in this experiment.")
    r.p("3. **Controls are limited.** Timm and Naibbe each use one published sample with chronological block splits. They are not repeated generator draws. Predicting Naibbe ciphertext is not the same as recovering its known plaintext.")

    r.heading("04 / Larger Qwen added little")
    r.p("One rented L40S compared Qwen3-1.7B-Base and Qwen3-8B-Base.")
    r.p("**Thesis:** the observed size gain is too small and uncertain to justify another larger-model run on this evidence alone.")
    r.p("**Paired folio interval:** resample the same manuscript leaves for both models and recompute the difference.")
    r.figure("scaling", "Figure 5. The 95% interval over 15 folio groups includes zero. It describes page-sampling uncertainty for one seed and validation-selected checkpoints, not a confirmatory final-test result.")
    r.table([["Setting", "1.7B-Base", "8B-Base"], ["Validation BPC", "2.4771", "2.4689"],
             ["Optimizer updates", "3,000", "3,000"], ["Trainable adapter parameters", "1,245,184", "1,212,416"],
             ["Outer layers", "0, 1, 26, 27", "0, 1, 34, 35"]], [219, 140, 140])
    r.p("1. **Matched exposure.** Both cloud runs use the same token IDs, target masks, seed, context, update count and schedule. Adapter counts differ by 2.6%; model depth and pretraining also differ.")
    r.p("2. **Predeclared gate.** The shuffled cloud follow-up required a positive lower interval bound. It did not qualify and was skipped. Both selected adapters and the downloaded archive were verified.")
    r.p("3. **Cost.** GPU stopped after retrieval. Runpod showed $9.18 remaining from $10 at stop, about $0.82 used. The retained 50 GB volume costs $0.014/hour until deletion is confirmed; this is not a current balance quote.")
    r.p("4. **Operational correction.** The initial watchdog command probe accepted generic CLI help for an unsupported command. Manual shutdown used the verified legacy syntax. Detection was corrected and regression-tested; no GPU remains running from this experiment.")

    r.heading("05 / What is justified next")
    r.p("Use a **predict-then-test-meaning** research sequence.")
    r.p("**Thesis:** the next informative experiments should explain the GRU's prediction advantage and validate a recovery method on known text.")
    r.p("**Context ablation:** shorten the past text available to a saved model while retaining the same scored targets.")
    r.code("Freeze the completed results and keep final-test scores sealed\nCompare GRU context lengths on identical character targets\nRepeat robust comparisons on another transcription or quire split\nValidate plaintext recovery on held-out synthetic texts and keys\nOnly then test constrained Voynich mappings on unseen evidence")
    r.p("1. **Explain the signal.** Use the selected GRU at shorter contexts, with matched scoring windows. Test whether its advantage is explained by spelling and nearby repetitions. Do not retrain or choose new checkpoints during that ablation.")
    r.p("2. **Check transfer.** A second transcription, boundary rules and held-out quires test robustness beyond this one split. Compare each representation on its own matched target characters.")
    r.p("3. **Make recovery falsifiable.** Use newly generated Naibbe examples with held-out plaintexts and keys. Measure character or word recovery, not how fluent a proposed English or Italian rendering sounds.")
    r.sub("Evidence and reproducibility")
    r.p("Main data: GC2a-n in v101 transcription, 148 training pages and 29 validation pages; unreadable targets excluded. These characters are not necessarily individual manuscript glyphs. No accepted Voynich plaintext is used.")
    r.p("The frozen snapshot stores numerical results, learning curves and source-file SHA256 hashes. Twenty-two selected runs were verified: eight local Qwen, two cloud Qwen, and twelve character models. Character checkpoints reproduce selected scores after reload. The earlier interim report remains unchanged.")
    r.p("Builder: `experiments/completed_report.py`. Snapshot: `experiments/report-completed/snapshot.json`. Run with `--capture` only to create a reviewed newer snapshot. Character outputs use about 62 MiB; there were no new pretrained downloads.")
    r.p("Control sources: [Timm-Schinner generator](https://github.com/TorstenTimm/SelfCitationTextgenerator) and [Greshko's Naibbe implementation](https://github.com/greshko/naibbe-cipher). These experiments test the supplied samples, not all claims in those projects.", "Caption")
    r.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    path = OUT / "snapshot.json"
    if args.capture:
        path.write_text(json.dumps(capture(), indent=2) + "\n")
    build(json.loads(path.read_text()))
    print(PDF)


if __name__ == "__main__":
    main()
