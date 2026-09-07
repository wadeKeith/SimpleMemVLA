import functools
import math
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from transformers import Trainer
from transformers.trainer import get_last_checkpoint


def _permit_trusted_torch_load() -> None:
    from transformers.utils import import_utils

    if import_utils.is_torch_greater_or_equal("2.6"):
        return

    def _noop() -> None:
        return

    import transformers.trainer as _trainer_mod

    import_utils.check_torch_load_is_safe = _noop
    _trainer_mod.check_torch_load_is_safe = _noop

    if not getattr(torch.load, "_simplememvla_trusted", False):
        _orig_load = torch.load

        @functools.wraps(_orig_load)
        def _trusted_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        _trusted_load._simplememvla_trusted = True
        torch.load = _trusted_load


class _FrozenScheduler(torch.optim.lr_scheduler.LambdaLR):

    def step(self, *args, **kwargs):
        return

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class EpochAwareSampler(Sampler):
    def __init__(self, data_source: Dataset, shuffle: bool, seed: int):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if not self.shuffle:
            return iter(range(len(self.data_source)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(epoch)


class SimpleMemVLATrainer(Trainer):
    def __init__(self, *args, data_root: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.data_root = data_root
        self._metric_sums = {"action_loss": 0.0, "vl_loss": 0.0}
        self._metric_count = 0

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None:
            return None
        return EpochAwareSampler(train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return EpochAwareSampler(eval_dataset, shuffle=False, seed=self.args.seed)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        decay_names = self.get_decay_parameter_names(opt_model)
        grouped = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if p.requires_grad and "action_head" not in n and n in decay_names
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if p.requires_grad and "action_head" not in n and n not in decay_names
                ],
                "weight_decay": 0.0,
                "lr": self.args.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if p.requires_grad and "action_head" in n and n in decay_names
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.action_head_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if p.requires_grad and "action_head" in n and n not in decay_names
                ],
                "weight_decay": 0.0,
                "lr": self.args.action_head_lr,
            },
        ]
        grouped = [group for group in grouped if len(group["params"]) > 0]
        self._check_zero_partition_overflow(grouped)
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        return self.optimizer

    def _check_zero_partition_overflow(self, grouped: list[dict]) -> None:
        ds = getattr(self.args, "hf_deepspeed_config", None)
        if ds is None:
            return
        try:
            stage = int(ds.config.get("zero_optimization", {}).get("stage", 0))
        except Exception:
            return
        if stage not in (1, 2):
            return
        world = max(1, int(getattr(self.args, "world_size", 1)))
        for i, g in enumerate(grouped):
            numel = sum(p.numel() for p in g["params"])
            partition = -(-numel // world)
            if partition >= 2**31:
                raise ValueError(
                    f"DeepSpeed ZeRO-{stage}: param group {i} has {numel:,} elements "
                    f"-> per-rank partition {partition:,} >= 2^31 at world_size={world}. "
                    "This overflows DeepSpeed's int32 flat-buffer indexing and SILENTLY "
                    "trains NaN weights. Use more GPUs (the 8-GPU default is safe) or "
                    "split the optimizer param groups."
                )

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            opt = optimizer if optimizer is not None else self.optimizer
            self._det_base_lrs = [group["lr"] for group in opt.param_groups]
            self._det_warmup_steps = max(1, int(self.args.get_warmup_steps(num_training_steps)))
            self._det_total_steps = max(self._det_warmup_steps + 1, int(num_training_steps))
            kwargs = self.args.lr_scheduler_kwargs or {}
            self._det_min_lr_ratio = (
                float(kwargs.get("min_lr_rate", 0.0)) if isinstance(kwargs, dict) else 0.0
            )
            self.lr_scheduler = _FrozenScheduler(
                opt, [lambda _step: 1.0 for _ in opt.param_groups]
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _deterministic_lr_scale(self, step: int) -> float:
        warmup = self._det_warmup_steps
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, self._det_total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._det_min_lr_ratio + (1.0 - self._det_min_lr_ratio) * cosine

    def _apply_deterministic_lr(self) -> None:
        base_lrs = getattr(self, "_det_base_lrs", None)
        if base_lrs is None or self.optimizer is None:
            return
        scale = self._deterministic_lr_scale(self.state.global_step)
        for group, base in zip(self.optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

    def training_step(self, *args, **kwargs):
        self._apply_deterministic_lr()
        return super().training_step(*args, **kwargs)

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None
        if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "auto":
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                print("[SimpleMemVLATrainer] resume=auto: no checkpoint in output_dir, starting fresh.")
            else:
                print(f"[SimpleMemVLATrainer] resume=auto: resuming from {resume_from_checkpoint}")
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise FileNotFoundError(
                    f"resume=true but no checkpoint was found in {self.args.output_dir}."
                )
        elif isinstance(resume_from_checkpoint, str):
            if not Path(resume_from_checkpoint).is_dir():
                raise FileNotFoundError(
                    f"resume checkpoint directory does not exist: {resume_from_checkpoint}"
                )
        if resume_from_checkpoint is not None:
            _permit_trusted_torch_load()
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Model returned no loss. Did the batch include actions?")

        if model.training and not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite training loss ({loss.detach().float().item()}) at step "
                f"{self.state.global_step}: action_loss={outputs.action_loss}, "
                f"vl_loss={outputs.vl_loss}."
            )

        if self.args.process_index == 0 and model.training:
            action_loss = outputs.action_loss
            vl_loss = outputs.vl_loss
            if action_loss is not None:
                self._metric_sums["action_loss"] += float(action_loss.detach().cpu())
            self._metric_sums["vl_loss"] += (
                float(vl_loss.detach().cpu()) if vl_loss is not None else 0.0
            )
            self._metric_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs):
        if "loss" in logs and self._metric_count > 0:
            window = {
                key: value / self._metric_count
                for key, value in self._metric_sums.items()
            }
            logs = {**logs, **window}
            self._metric_sums = {"action_loss": 0.0, "vl_loss": 0.0}
            self._metric_count = 0
        if "learning_rate" in logs and self.optimizer is not None:
            group_lrs = [group["lr"] for group in self.optimizer.param_groups]
            if group_lrs:
                logs["lr/backbone"] = group_lrs[0]
                logs["lr/action_head"] = group_lrs[-1]
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if self.args.process_index != 0 or self.data_root is None:
            return
        output = Path(output_dir or self.args.output_dir)
        src = Path(self.data_root) / "meta" / "stats.json"
        if src.is_file():
            shutil.copyfile(src, output / "stats.json")
