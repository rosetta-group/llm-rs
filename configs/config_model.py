from pydantic import BaseModel, ConfigDict, Field, model_validator


class FineTuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = "meta-llama/Llama-2-7b-hf"
    revision: str = "main"
    local_files_only: bool = True
    manuscript_file: str = "GC2a-n.txt"
    dataset: str = "artifacts/data/gc"
    run_name: str = "trial"
    output_dir: str = "training_run_outputs/trial"
    logging_dir: str = "training_run_outputs/logs"
    trainable_layers: list[int] | None = None
    layer_selection: str = "outer"
    layer_count: int = Field(default=4, ge=1)
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    context: int = Field(default=256, ge=2)
    stride: int = Field(default=128, ge=1)
    num_train_epochs: int = Field(default=1, ge=1)
    train_batch_size: int = Field(default=1, ge=1)
    eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    eval_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=50, ge=1)
    logging_steps: int = Field(default=5, ge=1)
    max_steps: int = -1
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    quantize: bool = False
    report_to: str = "none"
    max_train_windows: int | None = Field(default=None, ge=1)
    max_eval_pages: int | None = Field(default=None, ge=1)
    tiny: bool = False
    frozen_reference: str | None = None
    tokenizer_path: str | None = None
    load_best_model_at_end: bool = False

    @model_validator(mode="after")
    def validate_settings(self):
        if self.stride > self.context:
            raise ValueError("stride cannot exceed context")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.layer_selection not in {"outer", "middle", "random", "input"}:
            raise ValueError("Unknown layer selection")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("Unknown device")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unknown dtype")
        if self.tokenizer_path and not self.tiny:
            raise ValueError("Learned tokenizers are only supported by random tiny models")
        if self.load_best_model_at_end and self.save_steps != self.eval_steps:
            raise ValueError("Best-checkpoint selection requires save_steps == eval_steps")
        if self.load_best_model_at_end and 0 < self.max_steps < self.eval_steps:
            raise ValueError("Best-checkpoint selection needs at least one validation checkpoint")
        return self
