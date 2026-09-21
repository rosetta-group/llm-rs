"""Local causal-LM adaptation with masked windows and streaming evaluation."""

import importlib.metadata
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from voynich.data import digest, load_documents, write_json
from voynich.evaluate import summarize, target_signature
from voynich.windows import CausalCollator, document_windows, encode_document, token_windows


def select_layers(total, count, selection="outer", seed=42):
    if not 1 <= count <= total:
        raise ValueError(f"Select 1..{total} layers")
    if selection == "input":
        return list(range(count))
    if selection == "middle":
        start = (total-count)//2
        return list(range(start, start+count))
    if selection == "random":
        return sorted(random.Random(seed).sample(range(total), count))
    if selection == "outer":
        early = count//2
        return list(range(early)) + list(range(total-(count-early), total))
    raise ValueError(f"Unknown selection: {selection}")


def get_lora_target_modules(trainable_layers):
    projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    return [f"model.layers.{layer}.{projection}" for layer in trainable_layers for projection in projections]


def device_for(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def tiny_model(seed=42, tokenizer_path=None):
    """A random model for plumbing checks, not evidence of language transfer."""
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    set_seed(seed)
    vocab = {chr(i): i for i in range(256)}
    vocab.update({"<bos>": 256, "<pad>": 257, "<unk>": 258})
    backend = Tokenizer(BPE(vocab, [], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>",
                                       eos_token="<bos>", pad_token="<pad>", unk_token="<unk>")
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=32,
                intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=2048,
                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id))
    return model, tokenizer


def load_model(config):
    device = device_for(config.device)
    if config.quantize and device != "cuda":
        raise ValueError("4-bit loading requires CUDA; use unquantized weights on this Mac")
    if config.tiny:
        model, tokenizer = tiny_model(config.seed, config.tokenizer_path)
    else:
        options = dict(revision=config.revision, local_files_only=config.local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, **options)
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for character offsets")
        extra = {}
        if config.quantize:
            from transformers import BitsAndBytesConfig
            extra = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"), device_map="auto")
        model, info = AutoModelForCausalLM.from_pretrained(config.model_path,
            torch_dtype=getattr(torch, config.dtype), low_cpu_mem_usage=True,
            output_loading_info=True, **options, **extra)
        missing = set(info["missing_keys"])
        if getattr(model.config, "tie_word_embeddings", False):
            missing.discard("lm_head.weight")
        if missing or info["mismatched_keys"] or info["unexpected_keys"]:
            raise ValueError(f"Checkpoint mismatch: {info}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not config.quantize:
        model.to(device)
    model.config.use_cache = False
    return model, tokenizer, device


def evaluation_rows(model, tokenizer, docs, context, stride, batch_size=1):
    """Keep per-page sums, never corpus-sized vocabulary logits."""
    device = next(model.parameters()).device
    collator = CausalCollator(tokenizer.pad_token_id)
    training = model.training
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, doc in enumerate(docs):
            ids, valid, char_units, positions = encode_document(tokenizer, doc["text"], return_positions=True)
            windows = list(token_windows(ids, valid, char_units, context, stride))
            loss = correct = count = units = 0
            for start in range(0, len(windows), batch_size):
                examples = windows[start:start+batch_size]
                batch = {k: v.to(device) for k, v in collator(examples).items()}
                labels = batch.pop("labels")[:, 1:]
                logits = model(**batch).logits[:, :-1, :].float()
                mask = labels != -100
                loss += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                            labels.reshape(-1), ignore_index=-100, reduction="sum").item()
                correct += ((logits.argmax(-1) == labels) & mask).sum().item()
                count += mask.sum().item()
                units += sum(row["units"] for row in examples)
                del logits
            rows.append({**{k: doc[k] for k in ("page", "folio", "quire", "currier", "section", "hand")},
                         "nll": loss, "correct": correct, "tokens": count, "units": units,
                         "target_sha256": target_signature(doc["text"], positions)})
            if (index+1) % 10 == 0:
                print(f"Evaluated {index+1}/{len(docs)} pages", flush=True)
    model.train(training)
    return rows


class WeightedTrainer(Trainer):
    """Use token-weighted validation loss."""
    def __init__(self, *args, eval_docs, run_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_docs, self.run_config = eval_docs, run_config

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        c = self.run_config
        rows = evaluation_rows(self.model, self.processing_class, self.eval_docs,
                               c.context, c.stride, c.eval_batch_size)
        self.last_evaluation = (self.state.global_step, rows)
        metrics = summarize(rows)["overall"]
        write_json(Path(c.output_dir)/"validation"/f"step-{self.state.global_step:06d}.json",
                   dict(step=self.state.global_step, summary=summarize(rows), pages=rows))
        curve = getattr(self, "learning_curve", [])
        curve.append(dict(step=self.state.global_step, **metrics))
        self.learning_curve = curve
        write_json(Path(c.output_dir)/"learning_curve.json", curve)
        values = {f"{metric_key_prefix}_loss": metrics["loss_nats"],
                  f"{metric_key_prefix}_accuracy": metrics["accuracy"],
                  f"{metric_key_prefix}_bits_per_character": metrics["bits_per_character"]}
        self.log(values)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, values)
        return values


def run_fine_tuning(config):
    output = Path(config.output_dir)
    if (output / "run.json").exists():
        raise ValueError(f"Run already exists: {output}; choose a new output directory")
    set_seed(config.seed)
    documents = load_documents(config.dataset)
    train_docs = [d for d in documents if d["split"] == "train"]
    val_docs = [d for d in documents if d["split"] == "validation"]
    if config.max_eval_pages:
        val_docs = sorted(val_docs, key=lambda d: d["page"])[:config.max_eval_pages]
    if not train_docs or not val_docs:
        raise ValueError("Empty training or validation split")
    print(f"Loading {config.model_path if not config.tiny else 'random tiny smoke model'}", flush=True)
    model, tokenizer, device = load_model(config)
    layers = config.trainable_layers or select_layers(model.config.num_hidden_layers,
        config.layer_count, config.layer_selection, config.seed)
    if len(set(layers)) != len(layers) or any(i < 0 or i >= model.config.num_hidden_layers for i in layers):
        raise ValueError("Invalid or duplicated layer index")
    targets = get_lora_target_modules(layers)
    if any(name not in dict(model.named_modules()) for name in targets):
        raise ValueError("Model must expose Llama/Qwen attention and MLP projection names")
    if config.frozen_reference:
        reference = Path(config.frozen_reference)
        previous = json.loads((reference/"run.json").read_text())
        keys = ("model_path", "revision", "context", "stride", "dtype", "quantize", "tiny", "max_eval_pages", "tokenizer_path")
        if any(previous["config"].get(k) != config.model_dump()[k] for k in keys):
            raise ValueError("Frozen reference has different model or evaluation settings")
        if config.tiny or previous["dataset_sha256"] != digest(Path(config.dataset)/"documents.json"):
            raise ValueError("Frozen reference requires identical data and pretrained weights")
        before = json.loads((reference/"frozen.json").read_text())["pages"]
        if any("target_sha256" not in row for row in before):
            raise ValueError("Frozen reference predates target checks; evaluate it again")
    else:
        print(f"Evaluating frozen model on {len(val_docs)} validation pages", flush=True)
        before = evaluation_rows(model, tokenizer, val_docs, config.context, config.stride, config.eval_batch_size)
    if config.quantize:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=targets, bias="none"))
    model.enable_input_require_grads()
    windows = [w for d in train_docs for w in document_windows(tokenizer, d, config.context, config.stride)]
    if config.max_train_windows:
        random.Random(config.seed).shuffle(windows)
        windows = windows[:config.max_train_windows]
    if not windows:
        raise ValueError("No scored training windows")
    dataset = Dataset.from_list([{k: w[k] for k in ("input_ids", "attention_mask", "labels")} for w in windows])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = dict(status="running", config=config.model_dump(), layers=layers, trainable_parameters=trainable,
        total_parameters=sum(p.numel() for p in model.parameters()), device=device,
        model_commit=getattr(model.config, "_commit_hash", None),
        dataset_sha256=digest(Path(config.dataset)/"documents.json"),
        manifest_sha256=digest(Path(config.dataset)/"manifest.json"),
        training_windows=len(windows), validation_pages=[d["page"] for d in val_docs],
        smoke_only=bool(config.tiny or config.max_eval_pages or config.max_train_windows),
        versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "datasets")})
    root = Path(__file__).resolve().parent
    run["code_sha256"] = {str(p.relative_to(root)): digest(p) for p in
                          [root/"fine_tuning.py", root/"configs/config_model.py", *sorted((root/"voynich").glob("*.py"))]}
    run["model_config"] = model.config.to_dict()
    write_json(output/"run.json", run)
    write_json(output/"frozen.json", dict(summary=summarize(before), pages=before))
    args = TrainingArguments(output_dir=str(output), run_name=config.run_name,
        num_train_epochs=config.num_train_epochs, max_steps=config.max_steps,
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps, max_grad_norm=config.max_grad_norm,
        optim="adamw_torch", lr_scheduler_type="cosine",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=device == "cuda" and config.dtype == "float16",
        bf16=device == "cuda" and config.dtype == "bfloat16", use_cpu=device == "cpu",
        dataloader_pin_memory=device == "cuda", eval_strategy="steps", eval_steps=config.eval_steps,
        save_strategy="steps", save_steps=config.save_steps, save_total_limit=2,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="bits_per_character", greater_is_better=False,
        logging_steps=config.logging_steps,
        report_to=config.report_to, seed=config.seed, data_seed=config.seed,
        remove_unused_columns=False, label_names=["labels"])
    trainer = WeightedTrainer(model=model, args=args, train_dataset=dataset,
        eval_dataset=dataset.select([]), processing_class=tokenizer,
        data_collator=CausalCollator(tokenizer.pad_token_id), eval_docs=val_docs, run_config=config)
    print(f"Training {trainable:,} parameters in layers {layers}", flush=True)
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    last = getattr(trainer, "last_evaluation", None)
    selected_step = trainer.state.global_step
    if config.load_best_model_at_end:
        if not trainer.state.best_model_checkpoint:
            raise ValueError("Training finished without a best validation checkpoint")
        selected_step = int(Path(trainer.state.best_model_checkpoint).name.rsplit("-", 1)[1])
        after = json.loads((output/"validation"/f"step-{selected_step:06d}.json").read_text())["pages"]
    else:
        after = (last[1] if last and last[0] == selected_step else
                 evaluation_rows(model, tokenizer, val_docs, config.context, config.stride, config.eval_batch_size))
    write_json(output/"adapted.json", dict(step=selected_step, summary=summarize(after), pages=after))
    run.update(status="complete", optimizer_steps=trainer.state.global_step, selected_step=selected_step,
               best_model_checkpoint=trainer.state.best_model_checkpoint)
    write_json(output/"run.json", run)
    return dict(output=str(output), frozen=summarize(before)["overall"], adapted=summarize(after)["overall"])
