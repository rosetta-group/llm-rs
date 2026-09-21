"""Token windows with explicit target masks and no decode/encode round trip."""

from dataclasses import dataclass


def encode_document(tokenizer, text, return_positions=False):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None:
        raise ValueError("Tokenizer needs a BOS or EOS token for page starts")
    valid, units, covered = [False], [0], 0
    positions = []
    for start, end in offsets:
        valid.append(end > start and "?" not in text[start:end])
        units.append(max(0, end-max(start, covered)) if valid[-1] else 0)
        if valid[-1]:
            positions.extend(range(max(start, covered), end))
        covered = max(covered, end)
    result = ([bos] + ids, valid, units)
    return (*result, positions) if return_positions else result


def token_windows(ids, valid, units, context=256, stride=128):
    if context < 1 or not 1 <= stride <= context:
        raise ValueError("Require 1 <= stride <= context")
    if not len(ids) == len(valid) == len(units):
        raise ValueError("IDs, masks, and unit counts must align")
    for target_start in range(1, len(ids), stride):
        end = min(len(ids), target_start + stride)
        start = max(0, end-context-1)
        labels = [token if i >= target_start and valid[i] else -100
                  for i, token in enumerate(ids[start:end], start)]
        if not any(x != -100 for x in labels):
            continue
        yield dict(input_ids=ids[start:end], attention_mask=[1]*(end-start), labels=labels,
                   units=sum(units[i] for i in range(target_start, end) if valid[i]),
                   start=start, target_start=target_start, end=end)


def document_windows(tokenizer, doc, context=256, stride=128):
    ids, valid, units = encode_document(tokenizer, doc["text"])
    return list(token_windows(ids, valid, units, context, stride))


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, examples):
        import torch
        size = max(len(row["input_ids"]) for row in examples)
        batch = {}
        for key, padding in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key] = torch.tensor([list(row[key]) + [padding]*(size-len(row[key])) for row in examples])
        return batch
