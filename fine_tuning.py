import torch
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import structlog
import wandb

from prepare_voynich_dataset import generate_datasets

logger = structlog.get_logger()


def get_lora_target_modules(trainable_layers):
    target_modules = []
    # target_modules.append("model.embed_tokens")
    for layer_idx in trainable_layers:
        # Target attention and MLP modules in trainable layers
        target_modules.extend([
            #f"model.layers.{layer_idx}.input_layernorm",
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj", 
            f"model.layers.{layer_idx}.self_attn.v_proj",
            f"model.layers.{layer_idx}.self_attn.o_proj",
            #f"model.layers.{layer_idx}.post_attention_layernorm",
            f"model.layers.{layer_idx}.mlp.gate_proj",
            f"model.layers.{layer_idx}.mlp.up_proj",
            f"model.layers.{layer_idx}.mlp.down_proj"
        ])
    # target_modules.append("model.norm")
    # target_modules.append("lm_head")
    return []


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Labels are original window sequences 
    # -> first token has no counterpart in predictions to compare with
    shift_labels = labels[..., 1:].contiguous()
    # Predictions are for next token 
    # -> last token has no counterpart in labels to compare with
    shift_logits = predictions[..., :-1, :].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    # Cross-entropy expects inputs of shape (examples, classes) and (examples,)
    # so flatten the batch and sequence dimensions to a single examples dimension.
    flattened_logits = shift_logits.view(-1, shift_logits.size(-1))
    flattened_labels = shift_labels.view(-1)
    losses = loss_fct(flattened_logits, flattened_labels)

    # Mask padding tokens
    mask = (shift_labels != -100).float()
    losses = losses.view_as(shift_labels) * mask
    
    # Compute mean loss and perplexity
    mean_loss = losses.sum() / mask.sum() # Mean loss over non-padding tokens
    perplexity = torch.exp(mean_loss)
    
    return {
        "perplexity": perplexity.item(),
        "eval_loss": mean_loss.item()
    }


def run_fine_tuning(config):

    wandb.init(
        project="llm-rosetta-stone",
        name=config.run_name,
        config=config.model_dump()
    )

    logger.info("Downloading tokenizer and model from huggingface hub")
    tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,  # Use 4-bit quantization to save memory
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    logger.info("Tokenizer and model loaded successfully")

    logger.info("Generate PEFT model for LoRA fine-tuning")
    target_modules = get_lora_target_modules(config.trainable_layers)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=[]
    )
    peft_model = get_peft_model(model, lora_config)
    logger.info("PEFT model generated successfully")

    logger.info("Loading datasets")
    train_dataset, val_dataset = generate_datasets(
        config.manuscript_file,
        tokenizer
    )
    if config.max_steps > 0:
        train_dataset = train_dataset.select(range(config.max_steps * config.train_batch_size))
        val_dataset = val_dataset.select(range(config.max_steps * config.eval_batch_size))
    logger.info("Datasets loaded successfully")

    logger.info("Preparing data collator and training arguments")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,  # For tensor core efficiency
    )

    training_args = TrainingArguments(
        run_name=config.run_name,
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        optim="adamw_torch",  # Use AdamW optimizer from PyTorch
        lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
        max_grad_norm=config.max_grad_norm,  # Gradient clipping
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        fp16=True,  # Use mixed precision training
        dataloader_pin_memory=False,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        report_to="wandb",  # Use Weights & Biases for logging
        remove_unused_columns=False,  # Required for PEFT models
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete")

    logger.info("Saving model and tokenizer")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("Model and tokenizer saved successfully")

    wandb.finish()
    logger.info("Fine-tuning run completed", run_name=config.run_name)