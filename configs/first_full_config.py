from .config_model import FineTuningConfig

first_full_config = FineTuningConfig(
    run_name="trial_run",
    output_dir="./trial_output",
    logging_dir="./trial_logs",
    trainable_layers=list(range(0,2)) + list(range(24,32)), ##
    lora_rank=16,
    lora_alpha=32,
    num_train_epochs=3, ##
    train_batch_size=2,
    eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4, ##
    weight_decay=0.05,
    warmup_steps=30, ##
    eval_steps=25,
    save_steps=50,
    logging_steps=5
)