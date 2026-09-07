from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    PreTrainedModel,
)
from transformers.utils import ModelOutput

from simplememvla.model.configuration_simplememvla import SimpleMemVLAConfig
from simplememvla.model.dit_action_head import DiTActionHead


@dataclass
class SimpleMemVLAOutput(ModelOutput):
    loss: torch.Tensor | None = None
    action_loss: torch.Tensor | None = None
    vl_loss: torch.Tensor | None = None
    pred_velocity: torch.Tensor | None = None


def sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    beta_alpha: float = 1.5,
    beta_beta: float = 1.0,
) -> torch.Tensor:
    dist = torch.distributions.Beta(
        torch.tensor(beta_alpha, device=device, dtype=torch.float32),
        torch.tensor(beta_beta, device=device, dtype=torch.float32),
    )
    t = dist.sample((batch_size,))
    return t.to(dtype=dtype)


def masked_flow_matching_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    pred = pred_velocity.float()
    target = target_velocity.float()
    mask = action_mask.to(device=pred.device, dtype=torch.float32)
    squared_error = (pred - target).pow(2) * mask
    per_channel_den = mask.sum(dim=1).clamp_min(1.0)
    per_channel = squared_error.sum(dim=1) / per_channel_den
    active_channels = (mask.sum(dim=1) > 0).to(dtype=torch.float32)
    per_sample_den = active_channels.sum(dim=1).clamp_min(1.0)
    per_sample = (per_channel * active_channels).sum(dim=1) / per_sample_den
    return per_sample.mean()


class SimpleMemVLAForActionPrediction(PreTrainedModel):

    config_class = SimpleMemVLAConfig
    base_model_prefix = "simplememvla"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: SimpleMemVLAConfig,
        backbone: nn.Module | None = None,
    ):
        super().__init__(config)
        self.backbone = backbone
        if self.backbone is None:
            if config.backbone_config:
                backbone_config = AutoConfig.for_model(**dict(config.backbone_config))
            else:
                backbone_config = AutoConfig.from_pretrained(
                    config.backbone_model_name_or_path
                )
            self.backbone = AutoModelForImageTextToText.from_config(backbone_config)

        if config.action_dim is None or config.action_horizon is None:
            raise ValueError(
                "SimpleMemVLAConfig.action_dim and action_horizon must be set; pick "
                "them from the benchmark spec (simplememvla.benchmarks.get_benchmark)."
            )
        if config.use_proprio and config.state_dim is None:
            raise ValueError(
                "SimpleMemVLAConfig.state_dim must be set when use_proprio=True; pick "
                "it from the benchmark spec (simplememvla.benchmarks.get_benchmark)."
            )
        config.hidden_size = self.backbone.config.text_config.hidden_size
        config.backbone_config = self.backbone.config.to_dict()
        rope_parameters = self.backbone.config.text_config.rope_parameters
        if config.dit_rope_theta is None:
            config.dit_rope_theta = rope_parameters.get("rope_theta")
        if config.dit_mrope_section is None:
            config.dit_mrope_section = rope_parameters.get("mrope_section")
        if config.dit_partial_rotary_factor is None:
            config.dit_partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        self.action_head = DiTActionHead(
            vlm_hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            hidden_size=config.dit_hidden_size,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            dropout=config.dit_dropout,
            rope_theta=config.dit_rope_theta,
            mrope_section=config.dit_mrope_section,
            partial_rotary_factor=config.dit_partial_rotary_factor,
            use_proprio=config.use_proprio,
            state_dim=config.state_dim,
            state_dropout_prob=config.state_dropout_prob,
        )
        self._apply_backbone_freeze(config.freeze_backbone)
        for module in self.modules():
            module._is_hf_initialized = True
        self.post_init()

    @classmethod
    def from_backbone(
        cls,
        backbone_model_name_or_path: str,
        config: SimpleMemVLAConfig | None = None,
        dtype: torch.dtype | str | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "SimpleMemVLAForActionPrediction":
        config = config or SimpleMemVLAConfig(
            backbone_model_name_or_path=backbone_model_name_or_path
        )
        config.backbone_model_name_or_path = backbone_model_name_or_path
        model_kwargs = dict(kwargs)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs["trust_remote_code"] = trust_remote_code
        backbone = AutoModelForImageTextToText.from_pretrained(
            backbone_model_name_or_path,
            **model_kwargs,
        )
        model = cls(config=config, backbone=backbone)
        target_dtype = backbone.dtype if dtype is not None else None
        if target_dtype is not None:
            model.action_head.to(dtype=target_dtype)
        return model

    def _apply_backbone_freeze(self, freeze: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_kwargs: dict,
    ) -> torch.Tensor:
        position_ids, _ = self.backbone.model.get_rope_index(
            input_ids,
            image_grid_thw=mm_kwargs.get("image_grid_thw"),
            video_grid_thw=mm_kwargs.get("video_grid_thw"),
            attention_mask=attention_mask,
            mm_token_type_ids=mm_kwargs.get("mm_token_type_ids"),
        )
        return position_ids

    def _backbone_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mm_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        mm_kwargs["use_cache"] = False
        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.backbone.parameters())
        ):
            outputs = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **mm_kwargs,
            )
            hidden_states = outputs.last_hidden_state
            vl_loss = None
            if labels is not None:
                vl_loss = self._sparse_ce_loss(hidden_states, labels)
        return hidden_states, vl_loss

    def _sparse_ce_loss(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        lm_head = self.backbone.get_output_embeddings()
        if lm_head is None:
            lm_head = self.backbone.lm_head
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:].to(shift_hidden.device)
        mask = shift_labels != -100
        if not bool(mask.any()):
            return (hidden_states.sum() + lm_head.weight.sum()) * 0.0
        selected = shift_hidden[mask]
        targets = shift_labels[mask]
        logits = lm_head(selected).float()
        return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

    def _gather_condition(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        span: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        span = span.to(device=hidden_states.device, dtype=torch.bool)
        lengths = span.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        if max_len == 0:
            raise ValueError("sub-task condition span is empty")

        batch_size, _, hdim = hidden_states.shape
        device = hidden_states.device
        embed = self.get_input_embeddings()
        cond_hidden = hidden_states.new_zeros(batch_size, max_len, hdim)
        cond_token = hidden_states.new_zeros(batch_size, max_len, hdim)
        cond_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=device)
        cond_pos = position_ids.new_zeros(3, batch_size, max_len)
        for b in range(batch_size):
            idx = span[b].nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            if n == 0:
                continue
            cond_hidden[b, :n] = hidden_states[b, idx]
            cond_token[b, :n] = embed(input_ids[b, idx])
            cond_mask[b, :n] = 1
            cond_pos[:, b, :n] = position_ids[:, b, idx]
        return cond_hidden, cond_token, cond_mask, cond_pos

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        **kwargs,
    ) -> SimpleMemVLAOutput:
        if position_ids is None:
            position_ids = self._position_ids(input_ids, attention_mask, kwargs)
        hidden_states, vl_loss = self._backbone_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        action_loss = None
        pred_velocity = None
        total_loss = None
        if actions is not None:
            actions = actions.to(device=hidden_states.device, dtype=hidden_states.dtype)
            batch_size = actions.shape[0]
            noise = torch.randn_like(actions)
            timesteps = sample_flow_timesteps(
                batch_size,
                device=actions.device,
                dtype=actions.dtype,
                beta_alpha=self.config.timestep_beta_alpha,
                beta_beta=self.config.timestep_beta_beta,
            )
            t_view = timesteps.view(batch_size, 1, 1)
            noisy_actions = (1 - t_view) * actions + t_view * noise
            target_velocity = noise - actions
            if action_mask is None:
                raise ValueError("action_mask is required when training on actions")
            action_mask = action_mask.to(device=actions.device)
            if labels is None:
                raise ValueError(
                    "labels (the supervised sub-task span) are required to train "
                    "the action head"
                )
            span = labels.ne(-100)
            cond_hidden, cond_token, cond_mask, cond_pos = self._gather_condition(
                hidden_states=hidden_states,
                input_ids=input_ids,
                span=span,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            pred_velocity = self.action_head(
                condition_hidden_states=cond_hidden,
                condition_token_embeds=cond_token,
                noisy_actions=noisy_actions,
                timesteps=timesteps,
                condition_position_ids=cond_pos,
                condition_attention_mask=cond_mask,
                state=state,
            )
            action_loss = masked_flow_matching_loss(
                pred_velocity=pred_velocity,
                target_velocity=target_velocity,
                action_mask=action_mask,
            )
            total_loss = self.config.action_loss_weight * action_loss

        if vl_loss is not None and self.config.vl_loss_weight:
            weighted_vl = self.config.vl_loss_weight * vl_loss
            total_loss = weighted_vl if total_loss is None else total_loss + weighted_vl

        return SimpleMemVLAOutput(
            loss=total_loss,
            action_loss=action_loss,
            vl_loss=vl_loss,
            pred_velocity=pred_velocity,
        )

    @torch.no_grad()
    def predict_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        condition_span: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        num_steps: int = 10,
        temperature: float = 1.0,
        state: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = self._position_ids(input_ids, attention_mask, kwargs)
        hidden_states, _ = self._backbone_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        cond_hidden, cond_token, cond_mask, cond_pos = self._gather_condition(
            hidden_states=hidden_states,
            input_ids=input_ids,
            span=condition_span,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return self.action_head.sample(
            cond_hidden,
            condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask,
            condition_token_embeds=cond_token,
            num_steps=num_steps,
            temperature=temperature,
            state=state,
        )

    @torch.no_grad()
    def generate_subtask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        **gen_kwargs,
    ) -> torch.LongTensor:
        gen_inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )
        if eos_token_id is not None:
            gen_inputs["eos_token_id"] = eos_token_id
        gen_inputs.update(gen_kwargs)
        return self.backbone.generate(**gen_inputs)
