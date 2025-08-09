from .config_model import FineTuningConfig

trial_config = FineTuningConfig(
    run_name="trial_run",
    output_dir="./trial_output",
    logging_dir="./trial_logs",
    trainable_layers=[0],
    lora_rank=8,
    lora_alpha=16,
    num_train_epochs=1,
    train_batch_size=1,
    eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=10,
    eval_steps=5,
    save_steps=5,
    logging_steps=1,
    max_steps=20
)