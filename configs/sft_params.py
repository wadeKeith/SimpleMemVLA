from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    backbone_model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")
    trust_remote_code: bool = field(default=True)
    dtype: str = field(default="bfloat16")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    model_max_length: int = field(default=8192)

    dit_hidden_size: int = field(default=2048)
    dit_depth: int = field(default=16)
    dit_num_heads: int = field(default=16)
    dit_mlp_ratio: float = field(default=3.5)
    dit_dropout: float = field(default=0.0)
    dit_rope_theta: Optional[float] = field(default=None)
    dit_mrope_section: Optional[list[int]] = field(default=None)
    dit_partial_rotary_factor: Optional[float] = field(default=None)
    timestep_beta_alpha: float = field(default=1.5)
    timestep_beta_beta: float = field(default=1.0)
    action_loss_weight: float = field(default=1.0)
    vl_loss_weight: float = field(default=1.0)

    freeze_backbone: bool = field(default=False)
    use_proprio: bool = field(default=True)
    state_dropout_prob: float = field(default=0.0)


@dataclass
class DataArguments:

    benchmark: str = field(
        metadata={
            "help": "Which benchmark to train on: "
            "rmbench | robomme | mikasa | robomemarena | libero"
        }
    )

    repo_id: Optional[str] = field(default=None)
    root: Optional[str] = field(default=None)
    revision: Optional[str] = field(default=None)
    episodes: Optional[list[int]] = field(default=None)
    download_videos: bool = field(default=True)
    video_backend: str = field(default="pyav")

    image_keys: Optional[list[str]] = field(default=None)
    history_image_keys: Optional[list[str]] = field(default=None)
    action_key: str = field(default="action")
    state_key: str = field(default="observation.state")
    task_key: str = field(default="task")
    subtask_key: str = field(default="subtask")
    subtask_index_key: Optional[str] = field(default=None)
    skip_video_demo_frames: Optional[bool] = field(default=None)

    history_video_sec: Optional[float] = field(default=None)
    history_video_fps: Optional[float] = field(default=None)
    variable_history: Optional[bool] = field(default=None)
    image_aug: Optional[bool] = field(default=None)
    native_video_fps: Optional[float] = field(default=None)

    robot_tag: Optional[str] = field(default=None)
    control_frequency_hz: Optional[int] = field(default=None)
    action_delta_indices: Optional[list[int]] = field(default=None)


@dataclass
class SFTTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./checkpoints/sft/simplememvla/baseline")
    run_name: Optional[str] = field(default=None)
    report_to: str | list[str] = field(default="wandb")

    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    max_steps: int = field(default=30000)
    save_steps: int = field(default=3000)
    logging_steps: int = field(default=10)
    save_total_limit: int = field(default=3)

    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=32)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_persistent_workers: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)

    backbone_lr: float = field(default=5e-6)
    action_head_lr: float = field(default=5e-5)
    weight_decay: float = field(default=1e-8)
    warmup_steps: int = field(default=1000)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=1e-8)
    optim: str = field(default="adamw_torch")
    max_grad_norm: float = field(default=1.0)
    seed: int = field(default=429)
    resume: str = field(default="false")

    def __post_init__(self):
        super().__post_init__()
        if self.dataloader_num_workers == 0 and self.dataloader_persistent_workers:
            self.dataloader_persistent_workers = False
        if self.lr_scheduler_type == "cosine_with_min_lr":
            kwargs = self.lr_scheduler_kwargs or {}
            if isinstance(kwargs, str):
                import json

                kwargs = json.loads(kwargs) if kwargs.strip() else {}
            if "min_lr" not in kwargs and "min_lr_rate" not in kwargs:
                kwargs["min_lr_rate"] = 0.1
            self.lr_scheduler_kwargs = kwargs
        self.resume = self._normalize_resume(self.resume)

    @staticmethod
    def _normalize_resume(value):
        if isinstance(value, bool):
            return value
        text = str(value).strip()
        if text.lower() in ("", "false", "none", "0"):
            return False
        if text.lower() in ("true", "1"):
            return True
        return text
