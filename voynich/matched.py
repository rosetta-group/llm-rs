"""Train a character GRU on one target per exact-length history."""

import math
import random
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.cloud import write
from .character import CharacterModel, PAD, split_documents
from .context import histories, score_context
from .data import digest, load_documents
from .evaluate import summarize, target_signature


def pack_training(docs, context, device):
    if not docs or any(d["split"] != "train" for d in docs):
        raise ValueError("Training examples must come from training pages only")
    examples = [row for doc in docs for row in histories(doc["text"], context)]
    if not examples:
        raise ValueError("No readable training targets")
    ids = torch.tensor([h + [PAD] * (context - len(h)) for h, _ in examples], device=device)
    lengths = torch.tensor([len(h) for h, _ in examples], device=device)
    targets = torch.tensor([t for _, t in examples], device=device)
    return ids, lengths, targets


def last_logits(model, ids, lengths):
    states, _ = model.recurrent(model.embedding(ids))
    last = states[torch.arange(len(ids), device=ids.device), lengths - 1]
    return model.head(model.norm(last)).float()


def save_weights(path, model):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temp)
    temp.replace(path)


def train(config, deadline=math.inf):
    """Primary result is the fixed final epoch, never the best validation epoch."""
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    run = dict(status="training", config=config, started_at=started, test_scored=False,
               primary="final_epoch", torch_version=torch.__version__)
    write(output / "run.json", run)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(config["seed"])
        rng = random.Random(config["seed"])
        data = Path(config["dataset"])
        train_docs, val_docs = split_documents(load_documents(data))
        model = CharacterModel("gru", config["width"], config["layers"], config["context"])
        save_weights(output / "initial.pt", model)
        run.update(dataset_sha256=digest(data / "documents.json"), code_sha256=digest(__file__),
                   initial_sha256=digest(output / "initial.pt"),
                   train_pages=[d["page"] for d in train_docs],
                   validation_pages=[d["page"] for d in val_docs],
                   train_targets={d["page"]: target_signature(d["text"]) for d in train_docs},
                   parameters=sum(p.numel() for p in model.parameters()))
        model.to(config["device"])
        ids, lengths, targets = pack_training(train_docs, config["context"], config["device"])
        per_epoch = len(targets)
        run["training_characters_per_epoch"] = per_epoch
        write(output / "run.json", run)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=.01)
        curves, seen, updates, best = [], 0, 0, math.inf
        for epoch in range(1, config["epochs"] + 1):
            if shutil.disk_usage(output).free < config.get("min_free_gib", 0) * 2**30:
                raise RuntimeError("Less than the reserved free disk space")
            order = list(range(per_epoch))
            rng.shuffle(order)
            model.train()
            nll = count = 0
            for offset in range(0, per_epoch, config["batch_size"]):
                if time.time() >= deadline:
                    raise TimeoutError("Training budget exhausted; no matched final result")
                indices = torch.tensor(order[offset:offset + config["batch_size"]], device=config["device"])
                optimizer.zero_grad(set_to_none=True)
                logits = last_logits(model, ids[indices], lengths[indices])
                loss = F.cross_entropy(logits, targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                size = len(indices)
                nll += loss.item() * size
                count += size
                seen += size
                updates += 1
            if count != per_epoch:
                raise ValueError("Each training character must be scored once per epoch")
            pages = score_context(model, val_docs, config["context"], config["eval_batch_size"])
            scores = dict(epoch=epoch, training_characters_seen=seen, updates=updates,
                          pages=pages, summary=summarize(pages), test_scored=False)
            bpc = scores["summary"]["overall"]["bits_per_character"]
            write(output / "validation" / f"epoch-{epoch:03d}.json", scores)
            curves.append(dict(epoch=epoch, updates=updates, training_characters_seen=seen,
                               training_bpc=nll / count / math.log(2), validation_bpc=bpc,
                               seconds=time.time() - started))
            write(output / "learning_curve.json", curves)
            save_weights(output / "last.pt", model)
            write(output / "last.json", scores)
            if bpc < best:
                best = bpc
                save_weights(output / "best.pt", model)
                write(output / "best.json", scores)
                run.update(best_epoch=epoch, best_validation_bpc=bpc)
            run.update(epoch=epoch, updates=updates, training_characters_seen=seen,
                       validation_bpc=bpc, wall_seconds=time.time() - started)
            write(output / "run.json", run)
            print(f"{output.name}: epoch {epoch}/{config['epochs']}, {bpc:.4f} BPC, "
                  f"{seen:,} characters ({run['wall_seconds']:.1f}s)", flush=True)
        model.load_state_dict(torch.load(output / "last.pt", map_location="cpu", weights_only=True))
        verified = score_context(model, val_docs, config["context"], config["eval_batch_size"])
        keys = lambda rows: [(r["page"], r["units"], r["target_sha256"]) for r in rows]
        if keys(pages) != keys(verified):
            raise ValueError("Reloaded checkpoint target mismatch")
        error = abs(summarize(verified)["overall"]["bits_per_character"] - bpc)
        if error > 1e-5 or seen != config["epochs"] * per_epoch:
            raise ValueError("Final checkpoint or matched exposure verification failed")
        if digest(data / "documents.json") != run["dataset_sha256"]:
            raise ValueError("Dataset changed during training")
        run.update(status="complete", checkpoint_verified=True, checkpoint_sha256=digest(output / "last.pt"),
                   wall_seconds=time.time() - started, reload_bpc_error=error)
        write(output / "run.json", run)
    except Exception as exc:
        run.update(status="failed", error=str(exc), wall_seconds=time.time() - started)
        write(output / "run.json", run)
        raise
    return run
