"""Secondary boundary diagnostic: calibrate on modern Italian, never on Dante answers."""

import math
from pathlib import Path
import time

from experiments.decipherment import OUT, PUBLIC, STATE, ROOT, read, write
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import edit_distance, normalize


def segment(text, costs, penalty, maximum=24):
    best = [0.] + [math.inf] * len(text)
    previous = [0] * (len(text) + 1)
    for end in range(1, len(text) + 1):
        for start in range(max(0, end - maximum), end):
            value = best[start] + costs.get(text[start:end], 15 + penalty * (end - start))
            if value < best[end]:
                best[end], previous[end] = value, start
    words, end = [], len(text)
    while end:
        start = previous[end]; words.append(text[start:end]); end = start
    return " ".join(reversed(words))


def main():
    destination = STATE / "boundary-predictions.json"
    if destination.exists():
        raise FileExistsError("Secondary predictions already frozen")
    counts = read(PUBLIC / "word-counts.json")
    total = sum(counts.values())
    costs = {w: -math.log(n / total) for w, n in counts.items()}
    source = next(s for s in read(ROOT / "experiments/language-sources.json")["sources"]
                  if s["repository"] == "UD_Italian-ISDT")
    dev = next(f for f in source["files"] if f["path"].endswith("-dev.conllu"))
    if digest(ROOT / dev["path"]) != dev["sha256"]:
        raise ValueError("Calibration source drift")
    texts = [normalize(" ".join(s["words"])) for s in conllu_sentences(ROOT / dev["path"])]
    calibration = [t for t in texts if 80 <= len(t) <= 300][:100]
    scores = []
    for penalty in [.5, 1., 1.5, 2., 2.5, 3.]:
        errors = sum(edit_distance(segment(t.replace(" ", ""), costs, penalty).split(), t.split()) for t in calibration)
        scores.append(dict(penalty=penalty, word_error_rate=errors / sum(len(t.split()) for t in calibration)))
    selected = min(scores, key=lambda r: r["word_error_rate"])["penalty"]
    predictions = read(STATE / "predictions.json")
    results = []
    for row in predictions["results"]:
        recovered = row["recovered"] if row["family"] == "substitution-spaces" else segment(row["recovered"].replace(" ", ""), costs, selected)
        results.append(dict(id=row["id"], family=row["family"], recovered=recovered))
    write(destination, dict(results=results, calibration_source=dev, calibration_sentences=len(calibration),
                           grid=scores, selected_penalty=selected, code_sha256=digest(__file__),
                           original_predictions_sha256=digest(STATE / "predictions.json"), at=time.time()))
    # Original passages become accessible only after secondary predictions are fixed.
    gold = {r["id"]: r["plaintext"] for r in read(STATE / "evaluator-only/answers.json")}
    for row in results:
        reference = gold[row["id"]]
        row["word_errors"] = edit_distance(row["recovered"].split(), reference.split())
        row["reference_words"] = len(reference.split())
        row["word_error_rate"] = row["word_errors"] / row["reference_words"]
    summary = {}
    for family in sorted({r["family"] for r in results}):
        rows = [r for r in results if r["family"] == family]
        summary[family] = sum(r["word_errors"] for r in rows) / sum(r["reference_words"] for r in rows)
    write(OUT / "boundary-diagnostic.json", dict(results=results, summary_word_error_rate=summary,
          calibration=read(destination), predictions_sha256=digest(destination),
          limitation="Secondary exploratory repair after observing the original boundary failure; calibrated on separate modern Italian development sentences, not Dante answers"))
    print(dict(selected_penalty=selected, calibration=scores, word_error_rates=summary))


if __name__ == "__main__":
    main()
