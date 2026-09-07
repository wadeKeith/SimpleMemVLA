import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _normalize_mrope_section(
    half_head_dim: int,
    mrope_section: Sequence[int],
) -> tuple[int, int, int]:
    section = [int(v) for v in mrope_section]
    if len(section) != 3:
        raise ValueError("mrope_section must contain exactly three values")
    if any(v <= 0 for v in section):
        raise ValueError("mrope_section values must be positive")
    if sum(section) == half_head_dim:
        return tuple(section)
    if half_head_dim < 3:
        raise ValueError("Qwen-style M-RoPE requires head_dim // 2 >= 3")

    total = float(sum(section))
    raw = [half_head_dim * value / total for value in section]
    scaled = [max(1, math.floor(value)) for value in raw]
    remainder = half_head_dim - sum(scaled)
    order = sorted(range(3), key=lambda idx: raw[idx] - scaled[idx], reverse=True)
    while remainder > 0:
        for idx in order:
            if remainder == 0:
                break
            scaled[idx] += 1
            remainder -= 1
    while remainder < 0:
        for idx in reversed(order):
            if remainder == 0:
                break
            if scaled[idx] > 1:
                scaled[idx] -= 1
                remainder += 1
    return tuple(scaled)


class QwenAlignedMRotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        mrope_section: Sequence[int],
        partial_rotary_factor: float,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if not 0 < partial_rotary_factor <= 1:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        self.head_dim = head_dim
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(head_dim * self.partial_rotary_factor)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1
        if self.rotary_dim <= 0:
            raise ValueError("partial_rotary_factor produced an empty rotary dimension")
        self.mrope_section = _normalize_mrope_section(
            self.rotary_dim // 2,
            mrope_section,
        )
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape (3, batch, seq)")

        inv_freq = self.inv_freq[None, None, :, None].float().to(x.device)
        inv_freq = inv_freq.expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float().to(x.device)
        freqs = (inv_freq @ pos).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class RotarySelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_theta: float,
        mrope_section: Sequence[int],
        partial_rotary_factor: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout
        self.rotary_emb = QwenAlignedMRotaryEmbedding(
            self.head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            partial_rotary_factor=partial_rotary_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        cos, sin = self.rotary_emb(x, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        attn = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        return self.out_proj(out)


class DiTBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_theta: float,
        mrope_section: Sequence[int],
        partial_rotary_factor: float,
        mlp_ratio: float = 3.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = RotarySelfAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        key_padding_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
            time_emb
        ).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h, key_padding_mask=key_padding_mask, position_ids=position_ids)
        x = x + gate_msa.unsqueeze(1) * h
        h = self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, action_dim)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(time_emb).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class DiTActionHead(nn.Module):

    def __init__(
        self,
        vlm_hidden_size: int,
        action_dim: int,
        action_horizon: int,
        rope_theta: float,
        mrope_section: Sequence[int],
        partial_rotary_factor: float,
        hidden_size: int = 2048,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 3.5,
        dropout: float = 0.0,
        use_proprio: bool = True,
        state_dim: int = 8,
        state_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.use_proprio = use_proprio
        self.state_dim = state_dim
        self.state_dropout_prob = float(state_dropout_prob)
        self.condition_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.condition_token_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        if self.use_proprio:
            self.state_proj = nn.Linear(state_dim, hidden_size)
            self.state_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.time_embed = TimestepEmbedding(hidden_size)
        self.condition_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.action_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.action_pos = nn.Parameter(torch.zeros(1, action_horizon, hidden_size))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    mrope_section=mrope_section,
                    partial_rotary_factor=partial_rotary_factor,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = DiTFinalLayer(hidden_size, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.condition_type, std=0.02)
        nn.init.normal_(self.action_type, std=0.02)
        nn.init.normal_(self.action_pos, std=0.02)
        if self.use_proprio:
            nn.init.normal_(self.state_type, std=0.02)

    @property
    def action_horizon(self) -> int:
        return self.action_pos.shape[1]

    @property
    def action_dim(self) -> int:
        return self.final_layer.linear.out_features

    def _build_position_ids(
        self,
        condition_hidden_states: torch.Tensor,
        horizon: int,
        condition_position_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor,
        has_state: bool = False,
    ) -> torch.Tensor:
        batch_size = condition_hidden_states.shape[0]
        device = condition_hidden_states.device

        condition_position_ids = condition_position_ids.to(device=device)

        valid = condition_attention_mask.to(device=device, dtype=torch.bool)
        masked_positions = condition_position_ids.masked_fill(
            ~valid.unsqueeze(0),
            -1,
        )
        action_start = masked_positions.amax(dim=(0, 2)).clamp_min(0) + 1

        pieces = [condition_position_ids]
        if has_state:
            state_position_ids = action_start.view(1, batch_size, 1).expand(3, -1, -1)
            pieces.append(state_position_ids)
            action_start = action_start + 1

        action_offsets = torch.arange(
            horizon,
            device=device,
            dtype=condition_position_ids.dtype,
        )
        action_position_ids = action_start.view(1, batch_size, 1) + action_offsets.view(
            1,
            1,
            horizon,
        )
        action_position_ids = action_position_ids.expand(3, -1, -1)
        pieces.append(action_position_ids)
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        condition_hidden_states: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        condition_position_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor,
        condition_token_embeds: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, horizon, _ = noisy_actions.shape

        condition_tokens = F.normalize(
            self.condition_proj(condition_hidden_states), p=2, dim=-1
        )
        if condition_token_embeds is not None:
            condition_tokens = condition_tokens + self.condition_token_proj(
                condition_token_embeds
            )
        condition_tokens = condition_tokens + self.condition_type

        valid_condition = condition_attention_mask.to(dtype=torch.bool)
        tokens = [condition_tokens]
        masks = [valid_condition]

        use_state = self.use_proprio and state is not None
        if use_state:
            state_token = self.state_proj(state.to(dtype=condition_tokens.dtype))
            state_token = state_token.unsqueeze(1) + self.state_type
            tokens.append(state_token)
            state_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=valid_condition.device
            )
            if self.training and self.state_dropout_prob > 0.0:
                keep = (
                    torch.rand(batch_size, 1, device=state_mask.device)
                    >= self.state_dropout_prob
                )
                state_mask = state_mask & keep
            masks.append(state_mask)

        action_tokens = self.action_proj(noisy_actions)
        action_tokens = action_tokens + self.action_type + self.action_pos[:, :horizon]
        tokens.append(action_tokens)
        masks.append(
            torch.ones(batch_size, horizon, dtype=torch.bool, device=valid_condition.device)
        )

        x = torch.cat(tokens, dim=1)
        key_padding_mask = ~torch.cat(masks, dim=1)

        position_ids = self._build_position_ids(
            condition_hidden_states=condition_hidden_states,
            horizon=horizon,
            condition_attention_mask=condition_attention_mask,
            condition_position_ids=condition_position_ids,
            has_state=use_state,
        )
        time_emb = self.time_embed(timesteps)
        for block in self.blocks:
            x = block(
                x,
                time_emb,
                key_padding_mask=key_padding_mask,
                position_ids=position_ids,
            )
        action_hidden = x[:, -horizon:]
        return self.final_layer(action_hidden, time_emb)

    @torch.no_grad()
    def sample(
        self,
        condition_hidden_states: torch.Tensor,
        condition_position_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor,
        num_steps: int = 10,
        temperature: float = 1.0,
        condition_token_embeds: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        batch_size = condition_hidden_states.shape[0]
        device = condition_hidden_states.device
        dtype = condition_hidden_states.dtype
        actions = (
            torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            * temperature
        )
        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        for i in range(num_steps):
            t = steps[i].expand(batch_size)
            dt = steps[i] - steps[i + 1]
            velocity = self.forward(
                condition_hidden_states,
                actions,
                t,
                condition_attention_mask=condition_attention_mask,
                condition_position_ids=condition_position_ids,
                condition_token_embeds=condition_token_embeds,
                state=state,
            )
            actions = actions - dt * velocity
        return actions
