import os
import warnings
import torch
from simplememvla.compat import apply_fla_torch_compat

apply_fla_torch_compat()

from transformers import AutoProcessor, HfArgumentParser, set_seed

from configs.sft_params import (
    DataArguments,
    ModelArguments,
    SFTTrainingArguments,
)
from simplememvla.benchmarks import BenchmarkSpec, get_benchmark
from simplememvla.data.collator import SimpleMemVLADataCollator
from simplememvla.data.dataset import SimpleMemVLADataset
from simplememvla.model import SimpleMemVLAConfig, SimpleMemVLAForActionPrediction
from simplememvla.training.sft_runner import TrainRunner

warnings.filterwarnings("ignore", category=FutureWarning)


def _dtype_from_string(value: str):
    value = value.lower()
    if value in ("auto", "none"):
        return "auto" if value == "auto" else None
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def resolve_benchmark_defaults(data_args: DataArguments) -> BenchmarkSpec:
    spec = get_benchmark(data_args.benchmark)
    if data_args.repo_id is None:
        data_args.repo_id = spec.repo_id
    if data_args.root is None:
        data_args.root = spec.root
    if data_args.image_keys is None:
        data_args.image_keys = list(spec.image_keys)
    if data_args.history_image_keys is None:
        data_args.history_image_keys = list(spec.history_image_keys)
    if data_args.history_video_sec is None:
        data_args.history_video_sec = spec.history_video_sec
    if data_args.history_video_fps is None:
        data_args.history_video_fps = spec.history_video_fps
    if data_args.variable_history is None:
        data_args.variable_history = spec.variable_history
    if data_args.image_aug is None:
        data_args.image_aug = spec.image_aug
    if data_args.robot_tag is None:
        data_args.robot_tag = spec.robot_tag
    if data_args.subtask_index_key is None:
        data_args.subtask_index_key = spec.subtask_index_key
    if data_args.skip_video_demo_frames is None:
        data_args.skip_video_demo_frames = spec.skip_video_demo_frames
    if data_args.action_delta_indices is None:
        data_args.action_delta_indices = list(range(spec.num_actions_chunk))
    return spec


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: SFTTrainingArguments,
):
    set_seed(training_args.seed)
    spec = resolve_benchmark_defaults(data_args)

    processor = AutoProcessor.from_pretrained(
        model_args.backbone_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
        model_max_length=model_args.model_max_length,
    )

    train_dataset = SimpleMemVLADataset(data_args, spec)

    config = SimpleMemVLAConfig(
        backbone_model_name_or_path=model_args.backbone_model_name_or_path,
        action_dim=spec.action_dim,
        action_horizon=spec.num_actions_chunk,
        state_dim=spec.state_dim,
        dit_hidden_size=model_args.dit_hidden_size,
        dit_depth=model_args.dit_depth,
        dit_num_heads=model_args.dit_num_heads,
        dit_mlp_ratio=model_args.dit_mlp_ratio,
        dit_dropout=model_args.dit_dropout,
        dit_rope_theta=model_args.dit_rope_theta,
        dit_mrope_section=model_args.dit_mrope_section,
        dit_partial_rotary_factor=model_args.dit_partial_rotary_factor,
        timestep_beta_alpha=model_args.timestep_beta_alpha,
        timestep_beta_beta=model_args.timestep_beta_beta,
        action_loss_weight=model_args.action_loss_weight,
        vl_loss_weight=model_args.vl_loss_weight,
        freeze_backbone=model_args.freeze_backbone,
        use_proprio=model_args.use_proprio,
        state_dropout_prob=model_args.state_dropout_prob,
        robot_tag=data_args.robot_tag,
        control_frequency_hz=train_dataset.control_frequency_hz,
        history_video_sec=data_args.history_video_sec,
        history_video_fps=data_args.history_video_fps,
        native_video_fps=train_dataset.native_fps,
        variable_history=data_args.variable_history,
        image_keys=data_args.image_keys,
        history_image_keys=train_dataset.history_image_keys,
        subtask_key=train_dataset.subtask_key,
        subtask_index_key=train_dataset.subtask_index_key,
    )

    model = SimpleMemVLAForActionPrediction.from_backbone(
        model_args.backbone_model_name_or_path,
        config=config,
        dtype=_dtype_from_string(model_args.dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = SimpleMemVLADataCollator(
        processor,
        max_length=model_args.model_max_length,
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    runner = TrainRunner(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        resume_from_checkpoint=training_args.resume,
        processor=processor,
        data_root=data_args.root,
    )
    runner.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    train(model_args, data_args, training_args)
