
from pydantic import BaseModel

class FineTuningConfig(BaseModel):
    
    # Model and data
    model_path: str = "meta-llama/Llama-2-7b-hf"
    manuscript_file: str = "GC2a-n.txt"
    run_name: str
    output_dir: str
    logging_dir: str

    # LoRA/PEFT
    trainable_layers: list[int] = list(range(0,4)) + list(range(28,32))
    lora_rank: int = 32
    lora_alpha: float = 64
    lora_dropout: float = 0.1

    # Training
    num_train_epochs: int = 3
    train_batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    max_steps: int = -1
