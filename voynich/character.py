"""Small causal character models. Fixed alphabet; no pretrained weights."""

import math
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from .data import digest, load_documents, write_json
from .evaluate import paired_interval, summarize, target_signature
from .windows import CausalCollator, token_windows

BOS, PAD = 256, 257
FIELDS = ("page", "folio", "quire", "currier", "section", "hand")


def windows(text, context=128, stride=64):
    if any(ord(c) > 255 for c in text):
        raise ValueError("Expected normalized IVTFF characters in range 0..255")
    return list(token_windows([BOS] + [ord(c) for c in text],
                [False] + [c != "?" for c in text],
                [0] + [int(c != "?") for c in text], context, stride))


class CharacterModel(nn.Module):
    def __init__(self, kind="transformer", width=128, layers=6, context=128):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(258, width, padding_idx=PAD)
        if kind == "transformer":
            self.position = nn.Embedding(context + 1, width)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(width, 4, width * 4, dropout=.1,
                    activation="gelu", batch_first=True, norm_first=True)
                for _ in range(layers)])
        elif kind == "gru":
            self.recurrent = nn.GRU(width, width, layers, batch_first=True,
                                    dropout=.1 if layers > 1 else 0)
        else:
            raise ValueError(f"Unknown character model: {kind}")
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        if self.kind == "transformer":
            x = x + self.position(torch.arange(x.shape[1], device=x.device))
            mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
            for block in self.blocks:
                x = block(x, src_mask=mask)
        else:
            # Reset hidden state at each window; never carry it between pages.
            x, _ = self.recurrent(x)
        return self.head(self.norm(x))


def batch(rows, device):
    data = CausalCollator(PAD)(rows)
    return data["input_ids"].to(device), data["labels"][:, 1:].to(device)


def score(model, docs, context=128, stride=64, batch_size=16):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows = []
    with torch.inference_mode():
        for doc in docs:
            examples = windows(doc["text"], context, stride)
            nll = count = correct = 0
            for start in range(0, len(examples), batch_size):
                ids, labels = batch(examples[start:start + batch_size], device)
                logits = model(ids)[:, :-1].float()
                mask = labels != -100
                nll += nn.functional.cross_entropy(logits.reshape(-1, 256), labels.reshape(-1),
                                                   ignore_index=-100, reduction="sum").item()
                count += mask.sum().item()
                correct += ((logits.argmax(-1) == labels) & mask).sum().item()
            if count != sum(c != "?" for c in doc["text"]):
                raise ValueError("Each readable character must be scored exactly once")
            rows.append({**{k: doc[k] for k in FIELDS}, "nll": nll, "tokens": count,
                         "units": count, "correct": correct,
                         "target_sha256": target_signature(doc["text"])})
    model.train(was_training)
    return rows


def split_documents(docs):
    train = [d for d in docs if d["split"] == "train"]
    validation = [d for d in docs if d["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Need training and validation pages")
    return train, validation


def train(config, deadline=None):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    run = dict(status="running", config=config, started_at=started, test_scored=False)
    write_json(output / "run.json", run)
    try:
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(2)
        if config["device"] == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; review before changing hardware")
        device = torch.device(config["device"])
        data = Path(config["dataset"])
        train_docs, val_docs = split_documents(load_documents(data))
        run.update(dataset_sha256=digest(data / "documents.json"),
                   code_sha256=digest(__file__), torch_version=torch.__version__,
                   train_pages=[d["page"] for d in train_docs],
                   validation_pages=[d["page"] for d in val_docs])
        model = CharacterModel(config["kind"], config["width"], config["layers"],
                               config["context"]).to(device)
        run["parameters"] = sum(p.numel() for p in model.parameters())
        examples = [w for d in train_docs for w in windows(d["text"], config["context"], config["stride"])]
        run["training_characters_per_epoch"] = sum(w["units"] for w in examples)
        write_json(output / "run.json", run)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=.01)
        order_rng = random.Random(config["seed"])
        best, best_epoch, stale, updates, seen = math.inf, 0, 0, 0, 0
        curve = []
        stop_reason = "max_epochs"
        deadline = min(deadline or math.inf, started + config["max_seconds"])
        for epoch in range(1, config["epochs"] + 1):
            order = list(range(len(examples)))
            order_rng.shuffle(order)
            train_nll = train_count = 0
            model.train()
            for offset in range(0, len(order), config["batch_size"]):
                if time.time() >= deadline:
                    stop_reason = "time_limit"
                    break
                selected = [examples[i] for i in order[offset:offset + config["batch_size"]]]
                ids, labels = batch(selected, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(ids)[:, :-1]
                loss = nn.functional.cross_entropy(logits.reshape(-1, 256), labels.reshape(-1), ignore_index=-100)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                count = sum(w["units"] for w in selected)
                train_nll += loss.item() * count
                train_count += count
                updates += 1
                seen += count
            if stop_reason == "time_limit":
                break
            rows = score(model, val_docs, config["context"], config["stride"], config["batch_size"])
            scores = dict(epoch=epoch, updates=updates, training_characters_seen=seen,
                          summary=summarize(rows), pages=rows)
            bpc = scores["summary"]["overall"]["bits_per_character"]
            write_json(output / "validation" / f"epoch-{epoch:03d}.json", scores)
            curve.append(dict(epoch=epoch, updates=updates, training_characters_seen=seen,
                              training_bpc=train_nll / train_count / math.log(2),
                              validation_bpc=bpc, seconds=time.time() - started))
            write_json(output / "learning_curve.json", curve)
            if bpc < best:
                best, best_epoch, stale = bpc, epoch, 0
                temp = output / "best.tmp"
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temp)
                temp.replace(output / "best.pt")
                write_json(output / "adapted.json", scores)
            else:
                stale += 1
            run.update(epoch=epoch, updates=updates, selected_epoch=best_epoch,
                       training_characters_seen=seen, validation_bpc=bpc)
            write_json(output / "run.json", run)
            print(f"{output.name}: epoch {epoch}, validation {bpc:.4f} BPC, best {best:.4f}", flush=True)
            if epoch >= 2 and stale >= config["patience"]:
                stop_reason = "early_stopping"
                break
        if not best_epoch:
            raise TimeoutError("Budget ended before one full epoch and validation")
        # Reload the exported weights and reproduce their recorded scores.
        model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True))
        verified = score(model, val_docs, config["context"], config["stride"], config["batch_size"])
        selected = json.loads((output / "adapted.json").read_text())
        paired_interval(selected["pages"], verified)
        error = abs(summarize(verified)["overall"]["bits_per_character"] - best)
        if error > 1e-5:
            raise ValueError(f"Exported model score differs by {error}")
        run.update(status="complete", stop_reason=stop_reason, selected_epoch=best_epoch,
                   updates=updates, training_characters_seen=seen, checkpoint_verified=True,
                   checkpoint_sha256=digest(output / "best.pt"), wall_seconds=time.time() - started)
        write_json(output / "run.json", run)
    except Exception as exc:
        run.update(status="failed", error=str(exc))
        write_json(output / "run.json", run)
        raise
    return output
