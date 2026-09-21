"""Learn reversible symbol groups using training pages only."""

from pathlib import Path

from .data import digest, load_documents, write_json


def train_tokenizer(dataset, output, merges=64):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    if merges < 0:
        raise ValueError("merges must be non-negative")
    documents = load_documents(dataset)
    train = [d for d in documents if d["split"] == "train"]
    if not train:
        raise ValueError("Tokenizer needs training pages")
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex("[.,\n\x1c\x1d]"), behavior="isolated")
    backend.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(vocab_size=259+merges, min_frequency=2, show_progress=False,
        initial_alphabet=[chr(i) for i in range(256)], special_tokens=["<bos>", "<pad>", "<unk>"])
    backend.train_from_iterator((d["text"] for d in train), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<bos>",
                                       pad_token="<pad>", unk_token="<unk>")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    audit = dict(dataset_sha256=digest(Path(dataset)/"documents.json"),
                 training_pages=[d["page"] for d in train], requested_merges=merges,
                 vocabulary=len(tokenizer), decoder="lossless normalized transcription")
    for doc in documents:
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != doc["text"]:
            raise ValueError(f"Tokenizer round-trip failed on {doc['page']}")
    write_json(output/"training_manifest.json", audit)
    return audit
