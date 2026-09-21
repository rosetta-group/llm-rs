"""Descriptive written-form comparisons across pinned historical/modern corpora."""

import json
import math
from collections import Counter
from pathlib import Path
import time

import numpy as np

from voynich.corpora import conllu_sentences, letter_form, voynich_segments
from voynich.data import digest, load_documents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/language-comparison"


def entropy(counts):
    n = sum(counts.values())
    return -sum(v / n * math.log2(v / n) for v in counts.values()) if n else None


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else None


def one_edit(a, b):
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(a)):
        if a[i] != b[i]:
            return a[i:] == b[i + 1:]
    return True


def runs(segments):
    for segment in segments:
        current = []
        for word in segment["words"]:
            if word is None:
                if current:
                    yield current
                current = []
            else:
                current.append(word)
        if current:
            yield current


def profile(segments, glyphs=False, shuffles=24):
    sequences = list(runs(segments))
    words = [w for row in sequences for w in row]
    types = Counter(words)
    forms = {w: w if glyphs else letter_form(w) for w in types}
    lengths = np.array([len(forms[w]) for w in words])
    if not len(words):
        raise ValueError("Empty corpus")
    pairs = [(a, b) for row in sequences for a, b in zip(row, row[1:])]
    a = np.array([len(forms[x]) for x, _ in pairs])
    b = np.array([len(forms[y]) for _, y in pairs])
    lag = {}
    for distance in range(1, 6):
        x, y = [], []
        for row in sequences:
            values = [len(forms[w]) for w in row]
            x.extend(values[:-distance]); y.extend(values[distance:])
        lag[distance] = correlation(x, y)
    chars = Counter(c for w, n in types.items() for c in forms[w] for _ in range(n))
    bigrams = Counter()
    for w, n in types.items():
        for pair in zip(forms[w], forms[w][1:]):
            bigrams[pair] += n
    starts = Counter()
    for (x, _), n in bigrams.items():
        starts[x] += n
    n_pairs = sum(bigrams.values())
    h1 = -sum(n / n_pairs * math.log2(n / starts[x]) for (x, _), n in bigrams.items()) if n_pairs else None
    ids = {w: i for i, w in enumerate(types)}
    sizes = np.array([len(forms[w]) for w in types])
    encoded = [np.array([ids[w] for w in row]) for row in sequences if len(row) >= 2]
    rng = np.random.default_rng(42)
    null = []
    for _ in range(shuffles):
        permuted = [rng.permutation(row) for row in encoded]
        left = np.concatenate([row[:-1] for row in permuted])
        right = np.concatenate([row[1:] for row in permuted])
        null.append(dict(length_correlation=correlation(sizes[left], sizes[right]),
                         repeat_rate=float(np.mean(left == right))))
    windows = []
    if len(words) >= 1000:
        for start in rng.integers(0, len(words) - 999, size=32):
            counts = Counter(words[start:start + 1000])
            windows.append(dict(types=len(counts), hapax_token_fraction=sum(n == 1 for n in counts.values()) / 1000))
    # Fixed 10-word blocks reduce the sentence-versus-manuscript-line length mismatch.
    blocks = [row[start:start + 10] for row in sequences for start in range(0, len(row) - 9, 10)]
    block_pairs = [(len(forms[a]), len(forms[b])) for row in blocks for a, b in zip(row, row[1:])]
    block_corr = correlation(*zip(*block_pairs)) if block_pairs else None
    hist = Counter(map(int, lengths))
    joint = np.bincount((np.minimum(a, 12) - 1) * 12 + np.minimum(b, 12) - 1, minlength=144).reshape(12, 12)
    result = dict(words=len(words), characters=int(lengths.sum()),
                  excluded_uncertain_forms=sum(w is None for s in segments for w in s["words"]),
                  segments=len(segments), types=len(types), alphabet=len(chars),
                  length_mean=float(np.mean(lengths)), length_median=float(np.median(lengths)),
                  length_sd=float(np.std(lengths)), length_cv=float(np.std(lengths) / np.mean(lengths)),
                  length_quantiles=[float(v) for v in np.quantile(lengths, [.05, .25, .75, .95])],
                  length_histogram=dict(sorted(hist.items())), short_1_2_fraction=float(np.mean(lengths <= 2)),
                  length_pair_counts=joint.tolist(),
                  long_10_plus_fraction=float(np.mean(lengths >= 10)), adjacency_pairs=len(pairs),
                  length_lag_correlations=lag, length_correlation=correlation(a, b),
                  ten_word_block_correlation=block_corr, ten_word_blocks=len(blocks),
                  repeat_rate=sum(x == y for x, y in pairs) / len(pairs),
                  one_edit_rate=sum(one_edit(forms[x], forms[y]) for x, y in pairs) / len(pairs),
                  shuffle_null=null, unigram_entropy=entropy(chars), next_character_entropy=h1,
                  normalized_next_character_entropy=h1 / math.log2(len(chars)),
                  word_entropy=entropy(types), top_10_fraction=sum(n for _, n in types.most_common(10)) / len(words),
                  matched_1000_windows=windows,
                  types_per_1000_mean=float(np.mean([w["types"] for w in windows])) if windows else None,
                  top_words=types.most_common(20))
    valid = [v["length_correlation"] for v in null if v["length_correlation"] is not None]
    result["length_correlation_shuffle_mean"] = float(np.mean(valid)) if valid else None
    result["length_correlation_excess"] = result["length_correlation"] - float(np.mean(valid)) if valid else None
    result["repeat_shuffle_mean"] = float(np.mean([r["repeat_rate"] for r in null]))
    return result


def main():
    sources = json.loads((ROOT / "experiments/language-sources.json").read_text())
    rows, inputs = [], {}
    for source in sources["sources"]:
        for file in source["files"]:
            path = ROOT / file["path"]
            if digest(path) != file["sha256"]:
                raise ValueError(f"Source changed: {path}")
        file = next(f for f in source["files"] if f["path"].endswith("-train.conllu"))
        segments = list(conllu_sentences(ROOT / file["path"]))
        rows.append(dict(id=source["repository"], label=source["label"], period=source["period"],
                         genre=source["genre"], source=file, metrics=profile(segments)))
        inputs[file["path"]] = file["sha256"]
        print(source["label"], rows[-1]["metrics"]["words"], flush=True)
    for dataset in ["gc", "zl", "gc-shuffle", "timm", "naibbe"]:
        path = ROOT / f"artifacts/data/{dataset}/documents.json"
        docs = load_documents(path.parent)
        inputs[str(path.relative_to(ROOT))] = digest(path)
        segments = list(voynich_segments(docs))
        label = {"gc": "Voynich v101", "zl": "Voynich EVA", "gc-shuffle": "Shuffled Voynich",
                 "timm": "Timm synthetic", "naibbe": "Naibbe ciphertext"}[dataset]
        rows.append(dict(id=dataset, label=label, period="Voynich" if dataset in {"gc", "zl"} else "control",
                         genre="training paragraph text" if dataset in {"gc", "zl", "gc-shuffle"} else "synthetic sample",
                         training_pages_or_blocks=len([d for d in docs if d["split"] == "train"]),
                         metrics=profile(segments, glyphs=True)))
        print(label, rows[-1]["metrics"]["words"], flush=True)
        if dataset == "gc":
            for variety in ["A", "B"]:
                subset = [r for r in segments if r["currier"] == variety]
                rows.append(dict(id=f"gc-{variety}", label=f"Voynich v101 {variety}", period="subgroup",
                                 genre="Currier variety", metrics=profile(subset, glyphs=True)))
            merged = list(voynich_segments(docs, uncertain="merge"))
            rows.append(dict(id="gc-merged", label="Voynich, uncertain spaces merged", period="sensitivity",
                             genre="boundary sensitivity", metrics=profile(merged, glyphs=True)))
    OUT.mkdir(parents=True, exist_ok=True)
    inputs["voynich/corpora.py"] = digest(ROOT / "voynich/corpora.py")
    inputs["experiments/languages.py"] = digest(__file__)
    inputs["experiments/language-sources.json"] = digest(ROOT / "experiments/language-sources.json")
    result = dict(rows=rows, source_manifest=sources, input_sha256=inputs,
                  test_scored=False, voynich_split="train", at=time.time(), shuffle_seed=42,
                  scope="Descriptive corpus samples; neither language identity nor translation")
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
