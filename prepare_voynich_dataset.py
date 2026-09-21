"""Compatibility entry point for the new IVTFF pipeline."""

from pathlib import Path

from voynich.data import build, load_documents
from voynich.windows import document_windows

ROOT = Path(__file__).resolve().parent


def generate_datasets(manuscript_file, tokenizer, scramble=False, seed=42, context=256, stride=128):
    from datasets import Dataset
    source = Path(manuscript_file)
    if not source.exists():
        source = ROOT / "voynich_transliterations" / manuscript_file
    control = "shuffle" if scramble else "intact"
    destination = ROOT / "artifacts" / "data" / f"{source.stem}-{control}"
    build(source, destination, ROOT / "experiments" / "splits" / f"{source.stem}-folio-{seed}.json",
          seed=seed, control=control)
    docs = load_documents(destination)
    datasets = []
    for split in ("train", "validation"):
        rows = [{key: window[key] for key in ("input_ids", "attention_mask", "labels")}
                for doc in docs if doc["split"] == split
                for window in document_windows(tokenizer, doc, context, stride)]
        datasets.append(Dataset.from_list(rows))
    return tuple(datasets)
