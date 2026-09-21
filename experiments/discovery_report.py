"""Illustrated corpus-statistics and controlled decipherment reports."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, TEAL, ORANGE, GRAY = "#176B87", "#32947C", "#C76423", "#777A86"


def read(path):
    return json.loads((ROOT / path).read_text())


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def language_report():
    data = read("experiments/language-comparison/results.json")
    folder = ROOT / "experiments/language-comparison"
    figures = folder / "figures"
    by_id = {r["id"]: r for r in data["rows"]}
    selected = [by_id[k] for k in ("gc", "zl")] + [r for r in data["rows"] if r["period"] in {"historical", "modern"}]
    selected += [by_id[k] for k in ("gc-shuffle", "timm", "naibbe")]
    colors = [BLUE if r["id"] == "gc" else TEAL if r["id"] == "zl" else ORANGE if r["period"] == "historical" else GRAY for r in selected]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    for ax, period in zip(axes, ("historical", "modern")):
        comparison = [r for r in data["rows"] if r["period"] == period]
        for r in comparison + [by_id["gc"], by_id["zl"]]:
            bins = np.zeros(15)
            for size, n in r["metrics"]["length_histogram"].items():
                bins[min(int(size), 15) - 1] += n
            color = BLUE if r["id"] == "gc" else TEAL if r["id"] == "zl" else None
            ax.plot(range(1, 16), bins / bins.sum(), label=r["label"], color=color,
                    linewidth=2.8 if color else 1.1, alpha=1 if color else .65)
        ax.set(title=f"Voynich and {period} samples", xlabel="Written-form length (15 includes longer forms)",
               xticks=[1, 3, 5, 7, 9, 11, 13, 15])
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Fraction of word tokens")
    save(fig, figures, "word-lengths")

    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    for i, r in enumerate(selected):
        m = r["metrics"]
        null = [n["length_correlation"] for n in m["shuffle_null"]]
        ax.plot([min(null), max(null)], [i, i], color="#B7B7B7", linewidth=5)
        ax.scatter(m["length_correlation_shuffle_mean"], i, color=GRAY, marker="|", s=60)
        ax.scatter(m["length_correlation"], i, color=colors[i], s=35, zorder=3)
    ax.set(yticks=range(len(selected)), yticklabels=[r["label"] for r in selected],
           xlabel="Adjacent word-length correlation (positive = similar lengths cluster)",
           title="Points: observed text · gray ranges: 24 within-segment shuffles")
    ax.invert_yaxis(); ax.axvline(0, color="#AAAAAA", linewidth=.8)
    save(fig, figures, "length-adjacency")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    for ax, key in zip(axes.flat, ("gc", "zl", "UD_Italian-Old", "UD_Italian-ISDT")):
        row = by_id[key]; counts = np.array(row["metrics"]["length_pair_counts"], dtype=float)
        totals = counts.sum(axis=1, keepdims=True)
        values = np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals >= 20)
        plot = ax.imshow(values, origin="lower", vmin=0, vmax=.6, cmap="Blues", extent=(.5, 12.5, .5, 12.5))
        ax.set(title=row["label"], xlabel="Next word length", ylabel="Current word length", xticks=[1,4,8,12], yticks=[1,4,8,12])
    fig.colorbar(plot, ax=axes, label="Conditional fraction; length 12 includes longer forms", shrink=.8)
    save(fig, figures, "adjacency-matrices")

    fig, axes = plt.subplots(1, 3, figsize=(11, 7), layout="constrained", sharey=True)
    for ax, key, label in zip(axes, ("length_cv", "types_per_1000_mean", "repeat_rate"),
                             ("Length variation: SD / mean", "Distinct forms per 1,000 tokens", "Adjacent identical words (%)")):
        values = [r["metrics"][key] * (100 if key == "repeat_rate" else 1) for r in selected]
        ax.barh(range(len(selected)), values, color=colors)
        ax.set(yticks=range(len(selected)), yticklabels=[r["label"] for r in selected], xlabel=label)
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    save(fig, figures, "other-features")

    lines = ["# Voynich versus historical and modern language samples", "",
             "This study compares written-form statistics. It does not identify Voynich's language.", "",
             "Voynich's length distribution is relatively narrow, and neighboring lengths cluster more strongly "
             "than in these eleven language samples. Transcription, boundaries, layout, genre, and sample selection "
             "all affect the comparison.", "",
             "**Word:** a written form under the stated boundary rule, not a proven linguistic word in Voynich.",
             "**Length:** transcription symbols for Voynich; Unicode letters, excluding diacritics and internal punctuation, for other texts.",
             "**Correlation:** positive values mean long forms tend to neighbor long forms and short forms short forms.", "",
             "## Scope and method", "", "```text", "Use Voynich training pages only",
             "Read pinned UD 2.18 training corpora as surface text", "Count written forms and their lengths",
             "Compare adjacency within manuscript lines or corpus sentences",
             "Shuffle forms within those same segments to check the effect of order",
             "Compare vocabulary diversity in equal 1,000-token windows", "```", "",
             "The main Voynich counts describe **148 training paragraph pages**, not the whole manuscript. "
             "Final-test pages are excluded. GC/v101 contains 25,723 usable forms; 45 forms with uncertain glyphs "
             "are excluded and break adjacency. The EVA version has 24,147 usable forms on the same count of training pages, "
             "with 520 uncertain forms excluded. The transcriptions differ in readings and boundaries.", "",
             "Known-language words are extracted from surface text, preserving internal apostrophes and hyphens. "
             "Letter counts omit those punctuation marks and combining diacritics. Case is lowered. "
             "CoNLL-U surface-token ranges and spacing are respected when reconstructing missing text; "
             "syntactic clitic splits are not treated as written spaces. "
             "[UD format documentation](https://universaldependencies.org/format.html)", "",
             "## Word counts and lengths", "", "![Word-length distributions](figures/word-lengths.png)", "",
             "| Sample | Word tokens | Distinct forms | Mean length | SD | Adjacent-length correlation |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in selected:
        m=r["metrics"]
        lines.append(f"| {r['label']} | {m['words']:,} | {m['types']:,} | {m['length_mean']:.3f} | {m['length_sd']:.3f} | {m['length_correlation']:+.3f} |")
    lines += ["", "Counts are corpus sizes, not estimates of the size or vocabulary of a language. "
              "Raw distinct-form counts depend on sample size; use the equal-size window measure below.", "",
              "1. **Transcription changes the answer.** Mean Voynich length is 3.865 in v101 and 4.975 in EVA. "
              "Old Italian is 4.029 and modern Italian 5.307 in these samples. Selecting a language by the nearest "
              "average would therefore be misleading; encoding units and genre already change the apparent resemblance.",
              "2. **Length is comparatively concentrated.** The v101 SD is 1.479, EVA 1.784, and modern Arabic 1.998; "
              "the other language samples are broader in absolute character units. The synthetic controls are also narrow "
              "(Timm 1.548; Naibbe 1.626), so narrowness does not determine whether a text has meaning.", "",
              "## Word-length adjacency", "", "![Length adjacency](figures/length-adjacency.png)", "",
              "![Conditional adjacent-length distributions](figures/adjacency-matrices.png)", "",
              "Heatmap rows with fewer than 20 observed pairs are blank; this avoids highlighting single rare words. "
              "The full count matrices remain in the numerical results.", "",
              "1. **Similar lengths cluster.** v101 has adjacent-length correlation +0.218; EVA +0.169. "
              "Old Italian is −0.146 and modern Italian −0.183; Ancient Greek, Arabic, and Finnish have small positive values.",
              "2. **Line composition explains part of it.** Shuffling within Voynich lines gives mean correlations "
              "around +0.088 (v101) and +0.072 (EVA). Observed excesses are +0.130 and +0.097. "
              "These measure order beyond the segment's collection of word lengths, not syntax. "
              "Line-position rules, copying, and transcription choices remain possible explanations.",
              "3. **Segment definitions matter.** Manuscript lines are not known sentences. A sensitivity check uses "
              "nonoverlapping 10-word blocks within longer segments: v101 +0.225 and EVA +0.187. "
              "It retains only sufficiently long segments and does not make genres or layouts equivalent. "
              "Shuffle ranges describe these 24 permutations, not population confidence intervals.", "",
              "## Vocabulary, repetition, and predictability", "", "![Additional features](figures/other-features.png)", "",
              "| Sample | Types / 1,000 tokens | Immediate repeat % | One-edit neighbor % | Within-word next-character entropy (bits) |",
              "|---|---:|---:|---:|---:|"]
    for r in selected:
        m=r["metrics"]
        lines.append(f"| {r['label']} | {m['types_per_1000_mean']:.1f} | {100*m['repeat_rate']:.3f} | {100*m['one_edit_rate']:.3f} | {m['next_character_entropy']:.3f} |")
    lines += ["", "Vocabulary diversity averages 32 seeded, contiguous 1,000-token windows. Windows may overlap; "
              "they are a size control, not independent replicates. One-edit neighbors differ by exactly one symbol "
              "insertion, deletion, or substitution; identical pairs are counted separately.", "",
              "Voynich repeats and near-copies adjacent forms more often than the language samples, but the shuffled "
              "and Timm controls also do so. The v101 immediate-repeat rate is 0.701%, versus 0.741% in the fixed "
              "within-line shuffle. Repetition within lines does not itself establish meaningful word order.", "",
              "Character entropy is a descriptive frequency calculation within words, not held-out model performance. "
              "It depends strongly on the transcription alphabet and word boundaries: EVA gives 2.08 bits, v101 2.62. "
              "Do not compare it directly with the model BPC in earlier reports.", "",
              "## Voynich sensitivity checks", "",
              "| Variant | Tokens | Mean length | Adjacent-length correlation | Excess above shuffled segment |",
              "|---|---:|---:|---:|---:|"]
    for key in ("gc", "gc-A", "gc-B", "gc-merged", "zl"):
        r=by_id[key];m=r["metrics"]
        lines.append(f"| {r['label']} | {m['words']:,} | {m['length_mean']:.3f} | {m['length_correlation']:+.3f} | {m['length_correlation_excess']:+.3f} |")
    lines += ["", "The A/B rows exclude other or unknown variety labels and need not sum to the whole sample. "
              "Merging uncertain spaces changes the mean but preserves a positive adjacency excess in this sample.", "",
              "## What this means for translation", "",
              "A candidate encoding must explain boundaries, narrow written-form lengths, local length clustering, "
              "and frequent related forms together. These constraints can reject an overly simple account, but they "
              "cannot select a source language from a resemblance ranking. A letter-for-letter substitution preserving "
              "spaces preserves the source word lengths; Naibbe's artificial spacing does not. "
              "Our [controlled decipherment benchmark](../decipherment/REPORT.md) tests recoverable content separately.", "",
              "## Sources and limitations", "",
              "Pinned [UD 2.18](https://universaldependencies.org/download.html) corpora are samples, not language-wide "
              "distributions. Ancient literature, medieval poetry/theology/legal text, and modern news/web text are "
              "not genre matched. Corpus/topic effects, orthography, and morphological structure are confounded. "
              "No source-language probabilities, universal rankings, or causal historical changes are inferred.", "",
              "| Sample | Repository | Recorded genre | License |", "|---|---|---|---|"]
    for s in data["source_manifest"]["sources"]:
        lines.append(f"| {s['label']} | [{s['repository']}](https://github.com/UniversalDependencies/{s['repository']}/tree/{s['revision']}) | {s['genre']} | {s.get('license','See retained license')} |")
    lines += ["", "Individual source licenses and attribution files are retained under `artifacts/language-sources/`. "
              "Latin and Old French sources include noncommercial licenses. The manifest records every downloaded "
              "file, immutable revision, hash, and size. Downloads total about 210 MiB, with no model downloads.", "",
              "- [Full numerical results, histograms, and hashes](results.json)",
              "- [Pinned source manifest](../language-sources.json)", "- [Overall research record](../../RESEARCH_LOG.md)", "",
              "```sh", ".venv/bin/python -m experiments.languages", "python -m experiments.discovery_report", "```", ""]
    (folder / "REPORT.md").write_text("\n".join(lines))


def decipherment_report():
    data = read("experiments/decipherment/results.json")
    secondary = read("experiments/decipherment/boundary-diagnostic.json")
    folder = ROOT / "experiments/decipherment"
    families = data["plan"]["families"]
    labels = ["Substitution\nspaces supplied", "Substitution\nspaces removed", "Naibbe\nstructure supplied"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    xs = np.arange(3)
    for offset, metric, label, color in [(-.18, "frequency_baseline_character_accuracy", "Frequency baseline", GRAY),
                                        (.18, "recovered_character_accuracy", "Language-model key search", BLUE)]:
        axes[0].bar(xs + offset, [100*data["summaries"][f][metric] for f in families], width=.34, label=label, color=color)
    axes[0].set(ylabel="Correct normalized letters (%)", ylim=(0, 110), xticks=xs, xticklabels=labels, title="Recovering letters")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.20))
    axes[1].bar(xs - .18, [100*data["summaries"][f]["recovered_word_error_rate"] for f in families], width=.34, color=ORANGE, label="Original boundary baseline")
    axes[1].bar(xs + .18, [100*secondary["summary_word_error_rate"][f] for f in families], width=.34, color=TEAL, label="Calibrated boundary diagnostic")
    axes[1].set(ylabel="Word error rate (%) — lower is better", xticks=xs, xticklabels=labels, title="Recovering word boundaries")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.20))
    save(fig, folder / "figures", "recovery")
    lines = ["# Controlled decipherment: recovering Italian content", "",
             "A blind solver recovered normalized Italian from synthetic ciphertext without receiving the true letter key or paired original passages.", "",
             "It recovered every letter and word when original spaces were supplied. Removing the spaces made "
             "word recovery much harder, even when the letter sequence was correct. This is a controlled positive "
             "result for decipherment, not a Voynich translation.", "",
             "**Plaintext:** the original readable passage. **Blind:** the solver receives no answer/key file.",
             "**Word error rate:** substitutions, deletions, and insertions divided by reference word count; lower is better.", "",
             "```text", "Fit character frequencies on unrelated modern Italian training text",
             "Encrypt six Dante passages under two hidden random letter keys each",
             "Attempt each passage/key with spaces, without spaces, and through Naibbe",
             "Freeze predictions before opening the evaluator's originals", "Score recovered letters and words",
             "Keep the original failure record when testing a later boundary repair", "```", "",
             "## What was done", "",
             "- Ran 36 recovery cases: six nonoverlapping Dante passages, two keys, and three conditions.",
             "- Trained a four-character frequency model on modern Italian ISDT training text; no LLM or paired cipher/original training examples were used.",
             "- Used seeded simulated annealing to search a substitution key: six restarts × 12,000 proposals per case. Frequency-only decoding is the baseline.",
             "- Generated hidden keys independently with system randomness and stored them only in the evaluator directory. Solver inputs and predictions are hashed.",
             "- Used local CPU computation; the Voynich model suite retained exclusive use of the Apple GPU.", "",
             "## Results", "", "![Letter and word recovery](figures/recovery.png)", "",
             "| Condition | Letter recovery | Frequency-only letters | Original word error rate | Calibrated boundary error rate (secondary) |",
             "|---|---:|---:|---:|---:|"]
    for family,label in zip(families, ["Substitution, spaces supplied", "Substitution, spaces removed", "Naibbe, structure supplied"]):
        m=data["summaries"][family]
        lines.append(f"| {label} | {m['recovered_character_accuracy']:.2%} | {m['frequency_baseline_character_accuracy']:.2%} | {m['recovered_word_error_rate']:.2%} | {secondary['summary_word_error_rate'][family]:.2%} |")
    lines += ["", "Scores pool characters or words across cases. The independent source material is six passages "
              "from one work; keys and conditions reuse them. Thirty-six cases are not thirty-six independent texts.", "",
              "1. **A verified positive control.** With original spaces, the search recovered all normalized letters "
              "and words on these passages. The unrelated Italian character model supplies constraints strong enough "
              "to infer the hidden letter mapping for this known cipher family.",
              "2. **Letters are not complete text recovery.** Without spaces, the letters were still recovered exactly, "
              "but the original word splitter produced roughly 99% word error. Its unknown-word penalty favored "
              "long merged strings. That failure remains in the frozen primary results.",
              "3. **A transparent secondary repair.** After observing that failure, a separate diagnostic selected "
              "an unknown-word penalty on 100 modern Italian development sentences, then froze new segmentations "
              "before grading the Dante passages. It reduced error to about 60–62%, still poor. "
              "This is exploratory reuse of the benchmark, not a fresh confirmatory result. Vocabulary, spelling, "
              "and the simple boundary model are possible sources of the remaining errors; their effects are not isolated.", "",
              "## What the Naibbe condition supplies", "",
              "The [published Naibbe implementation](https://github.com/greshko/naibbe-cipher/tree/f2675ec5dd275268bc64dd48ea64fc0e0e9827a2) "
              "encodes one- or two-letter groups into longer glyph strings and introduces artificial token boundaries. "
              "We used its 52-card configuration and ambiguity-avoidance setting, with no extra deletion of ciphertext spaces. "
              "Original word spaces are removed. A hidden global permutation changes the plaintext-letter assignments.", "",
              "The solver receives the published structural codebook and cipher family. It first converts glyph groups "
              "to latent cipher letters, choosing a fixed lexical tie-break for ambiguous parses, then searches the unknown "
              "global letter key. This is substantially easier than discovering an unknown homophonic cipher from scratch. "
              "The remaining letter errors are consistent with this unresolved parse ambiguity. We do not infer that Voynich uses Naibbe.", "",
              "The encoder records a private trace. Each generated chunk was verified to be among the public decoder's "
              "candidates, and reversing the true letter key reproduced the normalized source letters. "
              "That check does not assert every glyph parse is unique or restore original word spaces.", "",
              "## Example", "",
              "First normalized original / correctly recovered spaced case:", "",
              "> nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura", "",
              "The original space-free boundary baseline instead began:", "",
              "> nelmezzodelcammindinostr avitamiritrovaiperunasel vaoscuracheladirittaviae", "",
              "The symbols can be correct while the proposed words are wrong. Evaluation must keep those outcomes separate.", "",
              "## Meaning, supervision, and limitations", "",
              "- Source language (Italian), normalization, alphabet, and cipher family are supplied. These advantages are not established for Voynich.",
              "- Accents, case, punctuation, and some spelling distinctions are removed: j→i, k→c, w→uu. Scores concern this declared normalized representation.",
              "- Dante is held out from the modern Italian frequency-model corpus. The solver is a frequency model fitted here, not a pretrained LLM that might have memorized the work.",
              "- No direct semantic question answering, source-language discovery, English rendering, or Voynich word mapping has been demonstrated.",
              "- The next useful recovery test should improve word segmentation with independent validation and fresh evaluation passages, then reduce the supplied cipher-structure assumptions.", "",
              "## Reproduction and provenance", "", "```sh", ".venv/bin/python -m experiments.decipherment prepare",
              ".venv/bin/python -m experiments.decipherment solve", ".venv/bin/python -m experiments.decipherment evaluate",
              ".venv/bin/python -m experiments.boundaries", "python -m experiments.discovery_report", "```", "",
              "Preparation and solving refuse existing benchmark outputs. The evaluator-only directory is separated "
              "from solver inputs; this is audited software input separation, not an OS sandbox. The blind solver "
              "does not open evaluator answers. The Voynich final test remains untouched.", "",
              "- [Fixed benchmark plan](plan.json)", "- [Primary results and recovered passages](results.json)",
              "- [Secondary boundary calibration and errors](boundary-diagnostic.json)",
              "- [Pinned Naibbe sources](../decipherment-sources.json)", "- [Pinned UD sources](../language-sources.json)",
              "- Original/hidden keys: `artifacts/decipherment/evaluator-only/answers.json`.",
              "- Frozen public solver inputs and predictions: `artifacts/decipherment/public/` and `predictions.json`.",
              "- [Overall goal and research history](../../RESEARCH_LOG.md)", "",
              "Naibbe attribution: Michael A. Greshko (2025), [The Naibbe cipher](https://doi.org/10.1080/01611194.2025.2566408); "
              "modified MIT license retained with the pinned source. Italian-Old/Dante and Italian-ISDT sources "
              "are credited in the source manifest. Italian-Old is CC BY-SA 4.0; Italian-ISDT is CC BY-NC-SA 3.0. "
              "Preserve the respective source terms and attribution when redistributing derived material. "
              "[Italian-Old corpus description](https://universaldependencies.org/treebanks/it_old/index.html)", ""]
    (folder / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    language_report()
    decipherment_report()
