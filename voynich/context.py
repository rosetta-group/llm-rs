"""Exact rolling-context evaluation of saved character GRUs."""

import torch
from torch.nn import functional as F

from .character import BOS, PAD, FIELDS
from .evaluate import target_signature


def histories(text, context):
    """Yield prior symbols and one readable target; never include future symbols."""
    if context < 1 or any(ord(c) > 255 for c in text):
        raise ValueError("Need positive context and normalized characters in 0..255")
    ids = [BOS] + [ord(c) for c in text]
    for i, char in enumerate(text):
        if char != "?":
            yield ids[max(0, i + 1 - context):i + 1], ord(char)


def score_context(model, docs, context, batch_size=64):
    """Reset state for every target. Right padding cannot affect the selected state."""
    if model.kind != "gru" or batch_size < 1:
        raise ValueError("Need a character GRU and positive batch size")
    if not docs or any(d["split"] != "validation" for d in docs):
        raise ValueError("Context experiments accept validation pages only")
    if len({d["page"] for d in docs}) != len(docs):
        raise ValueError("Duplicate validation page")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.inference_mode():
            for doc in docs:
                examples = list(histories(doc["text"], context))
                nll = count = correct = 0
                for offset in range(0, len(examples), batch_size):
                    selected = examples[offset:offset + batch_size]
                    lengths = torch.tensor([len(h) for h, _ in selected], device=device)
                    width = max(len(h) for h, _ in selected)
                    ids = torch.tensor([h + [PAD] * (width - len(h)) for h, _ in selected],
                                       device=device)
                    targets = torch.tensor([t for _, t in selected], device=device)
                    states, _ = model.recurrent(model.embedding(ids))
                    last = states[torch.arange(len(selected), device=device), lengths - 1]
                    logits = model.head(model.norm(last)).float()
                    losses = F.cross_entropy(logits, targets, reduction="none")
                    if not torch.isfinite(losses).all():
                        raise ValueError("Non-finite context loss")
                    # Sum in float64 on CPU: MPS does not support float64 tensors.
                    nll += losses.cpu().double().sum().item()
                    count += len(selected)
                    correct += (logits.argmax(-1) == targets).sum().item()
                if count != sum(c != "?" for c in doc["text"]):
                    raise ValueError("Every readable character must be scored once")
                rows.append({**{k: doc[k] for k in FIELDS}, "nll": nll, "tokens": count,
                             "units": count, "correct": correct,
                             "target_sha256": target_signature(doc["text"])})
    finally:
        model.train(was_training)
    return rows
