
import json
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from simplememvla.data.collator import answer_span_token_ids, subtask_span_mask
from simplememvla.data.messages import (
    build_simplememvla_messages,
    derive_video_sampling,
    variable_history_frames,
)
from simplememvla.data.normalize import MeanStdNormalizer, load_stats_file
from simplememvla.benchmarks.robomemarena import CAMERA_LABELS
from simplememvla.model.configuration_simplememvla import SimpleMemVLAConfig
from simplememvla.model.modeling_simplememvla import SimpleMemVLAForActionPrediction


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
if DEVICE.type == "cuda" and torch.cuda.is_available():
    torch.cuda.set_device(DEVICE)


def _load_norm_stats(checkpoint_path: str, key: str = "action"):
    stats = load_stats_file(os.path.join(checkpoint_path, "stats.json"), key)
    return {k: np.asarray(value, dtype=np.float32) for k, value in stats.items()}


def _load_checkpoint_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(os.path.join(checkpoint, shard)))
        return state_dict
    return load_file(os.path.join(checkpoint, "model.safetensors"))


def _build_vla(
    checkpoint: str,
    dtype: torch.dtype,
    attn_implementation: str | None = None,
) -> SimpleMemVLAForActionPrediction:
    config = SimpleMemVLAConfig.from_pretrained(checkpoint)
    state_dict = _load_checkpoint_state_dict(checkpoint)
    backbone_config = AutoConfig.for_model(**dict(config.backbone_config))
    from_config_kwargs = {}
    if attn_implementation:
        from_config_kwargs["attn_implementation"] = attn_implementation
    backbone = AutoModelForImageTextToText.from_config(
        backbone_config, **from_config_kwargs
    )
    for param in backbone.parameters():
        param.data = param.data.to(dtype)
    vla = SimpleMemVLAForActionPrediction(config=config, backbone=backbone)
    missing, unexpected = vla.load_state_dict(state_dict, strict=False)
    if any(key.endswith("lm_head.weight") for key in missing):
        vla.backbone.tie_weights()
        missing = [key for key in missing if not key.endswith("lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint weight mismatch loading {checkpoint}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    vla.action_head.to(dtype=dtype)
    wrong_dtype = [n for n, p in vla.backbone.named_parameters() if p.dtype != dtype]
    if wrong_dtype:
        raise RuntimeError(
            f"Backbone parameters were not cast to {dtype}: {wrong_dtype[:4]}"
        )
    rotary = [(name, buf.dtype) for name, buf in vla.backbone.named_buffers() if "inv_freq" in name]
    if not rotary:
        raise RuntimeError(
            "No backbone buffer matching '*inv_freq*' was found, so the fp32 rotary "
            "invariant cannot be checked. The backbone is not the expected Qwen3.5 stack."
        )
    demoted = [(name, str(buf_dtype)) for name, buf_dtype in rotary if buf_dtype != torch.float32]
    if demoted:
        raise RuntimeError(
            "Backbone rotary inv_freq buffers must stay fp32 to reproduce the trained "
            f"M-RoPE phases; got {demoted[:4]}"
        )
    return vla


def get_vla(
    checkpoint: str,
    compute_dtype: str = "bfloat16",
    attn_implementation: str | None = "flash_attention_2",
):
    vla = _build_vla(
        checkpoint, getattr(torch, compute_dtype), attn_implementation=attn_implementation
    )
    import transformers as _tf

    saved_tf = vla.config.transformers_version
    if saved_tf is not None and saved_tf != _tf.__version__:
        print(
            f"[policy] WARNING: checkpoint was saved with transformers=={saved_tf} "
            f"but this environment runs {_tf.__version__}; the rendered prompt may "
            "differ byte-wise from training. Pin transformers to the training "
            "version.",
            flush=True,
        )
    vla.eval().to(DEVICE)
    unnormalize_action = MeanStdNormalizer(_load_norm_stats(str(checkpoint), "action"))
    normalize_state = None
    if vla.config.use_proprio:
        normalize_state = MeanStdNormalizer(
            _load_norm_stats(str(checkpoint), "observation.state")
        )
    return vla, unnormalize_action, normalize_state


def _token_id(processor: AutoProcessor, token: str) -> int:
    token_id = processor.tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    raise ValueError(f"Tokenizer does not define required token {token!r}")


class RoboMemArenaPolicy:

    def __init__(
        self,
        vla: SimpleMemVLAForActionPrediction,
        processor: AutoProcessor,
        unnormalize_action: MeanStdNormalizer,
        normalize_state: MeanStdNormalizer | None = None,
        num_denoising_steps: int = 10,
        temperature: float = 1.0,
        max_reasoning_tokens: int = 256,
    ):
        if vla.config.control_frequency_hz is None:
            raise ValueError(
                "Checkpoint config is missing control_frequency_hz; cannot rebuild "
                "the embodiment prompt."
            )
        self.vla = vla
        self.processor = processor
        self.unnormalize = unnormalize_action
        self.normalize_state = normalize_state
        self.use_proprio = vla.config.use_proprio
        if self.use_proprio and self.normalize_state is None:
            raise ValueError(
                "Checkpoint uses proprioception but no state normalizer was provided."
            )
        self.num_denoising_steps = num_denoising_steps
        self.temperature = temperature
        self.max_reasoning_tokens = max_reasoning_tokens

        tok = processor.tokenizer
        self.think_close_id, self.whitespace_ids = answer_span_token_ids(tok)
        self.suppress_token_ids = [
            i for i in (tok.convert_tokens_to_ids("<think>"), self.think_close_id)
            if isinstance(i, int) and i >= 0
        ]
        newline_ids = tok.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected '\\n' to be one token, got {newline_ids}")
        self.newline_id = newline_ids[0]
        self.im_end_id = _token_id(processor, "<|im_end|>")
        self.pad_token_id = tok.pad_token_id

        cfg = vla.config
        self.image_keys = list(cfg.image_keys)
        self.camera_tags = [key.split(".")[-1] for key in self.image_keys]
        self.robot_tag = cfg.robot_tag
        self.control_frequency_hz = cfg.control_frequency_hz
        self.native_fps = float(cfg.native_video_fps)
        self.n_frames, self.stride = derive_video_sampling(
            cfg.history_video_sec, cfg.history_video_fps, self.native_fps
        )
        history_set = set(cfg.history_image_keys)
        if not cfg.variable_history:
            raise ValueError(
                "This checkpoint was trained with variable_history=False (a fixed-length "
                "history window with first-frame replication), but the rollout policy only "
                "implements the variable-length rule. Refusing to evaluate it under a "
                "different history construction than it was trained with."
            )
        unknown_history = sorted(history_set.difference(self.image_keys))
        if unknown_history:
            raise ValueError(
                f"history_image_keys {unknown_history} are not among image_keys "
                f"{self.image_keys}; the prompt would be built with fewer history cameras "
                "than the checkpoint was trained with."
            )
        self.history_image_keys = [k for k in self.image_keys if k in history_set]
        self.history_camera_tags = [key.split(".")[-1] for key in self.history_image_keys]
        self._buffer_caps = {
            k: ((self.n_frames - 1) * self.stride + 1 if k in history_set else 1)
            for k in self.image_keys
        }
        self.action_horizon = int(cfg.action_horizon)
        self.action_dim = int(cfg.action_dim)
        self._buffers: dict[str, list[np.ndarray]] = {k: [] for k in self.image_keys}
        self._frames_seen: dict[str, int] = {k: 0 for k in self.image_keys}
        self.runaway_decodes = 0
        self.last_decode_truncated = False
        self._runaway_warned_this_episode = False

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        self._frames_seen = {k: 0 for k in self.image_keys}
        self.last_decode_truncated = False
        self._runaway_warned_this_episode = False

    def observe(self, frames: dict[str, np.ndarray]) -> None:
        missing = [k for k in self.image_keys if k not in frames]
        if missing:
            raise KeyError(f"observe() missing frames for cameras: {missing}")
        for key in self.image_keys:
            buf = self._buffers[key]
            buf.append(np.asarray(frames[key]))
            self._frames_seen[key] += 1
            cap = self._buffer_caps[key]
            if len(buf) > cap:
                del buf[: len(buf) - cap]

    def _history_positions(self, key: str) -> list[int]:
        buf = self._buffers[key]
        cur = len(buf) - 1
        m = variable_history_frames(cur // self.stride + 1)
        return [max(0, cur - (m - 1 - i) * self.stride) for i in range(m)]

    def _history_video(self, key: str) -> np.ndarray:
        buf = self._buffers[key]
        idxs = self._history_positions(key)
        return np.stack([np.asarray(buf[i]) for i in idxs], axis=0)

    def _visual_inputs(self) -> tuple[list[np.ndarray], list[VideoMetadata], list[np.ndarray]]:
        sampled_fps = self.native_fps / self.stride
        history = set(self.history_image_keys)
        videos, metas, images = [], [], []
        for key in self.image_keys:
            if key in history:
                clip = self._history_video(key)
                n = clip.shape[0]
                videos.append(clip)
                metas.append(
                    VideoMetadata(
                        total_num_frames=n,
                        fps=sampled_fps,
                        frames_indices=list(range(n)),
                        duration=n / sampled_fps,
                        video_backend="rollout",
                    )
                )
            else:
                images.append(np.asarray(self._buffers[key][-1]))
        return videos, metas, images

    def _process(self, prompt: str, add_generation_prompt: bool):
        messages = build_simplememvla_messages(
            camera_tags=self.camera_tags,
            instruction=prompt,
            robot_tag=self.robot_tag,
            control_frequency_hz=self.control_frequency_hz,
            action_horizon=self.action_horizon,
            camera_labels=CAMERA_LABELS,
            subtask=None,
            history_camera_tags=self.history_camera_tags,
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        if not isinstance(text, str):
            text = text[0]
        videos, metas, images = self._visual_inputs()
        enc = self.processor(
            text=[text],
            images=images or None,
            videos=videos,
            video_metadata=metas,
            do_sample_frames=False,
            return_tensors="pt",
        )
        return {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in enc.items()
        }

    def _clip_len(self) -> int:
        if not self.history_image_keys:
            return 0
        return len(self._history_positions(self.history_image_keys[0]))

    def predict(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        observe: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], str]:
        if observe is not None:
            self.observe(observe)

        state_tensor = None
        if self.use_proprio:
            if state is None:
                raise ValueError(
                    "This checkpoint was trained with proprioception; predict() "
                    "requires the current `state` (eef pose + gripper_qpos)."
                )
            state_arr = np.asarray(state, dtype=np.float32)
            expected = int(self.vla.config.state_dim)
            if state_arr.ndim != 1 or state_arr.shape[0] != expected:
                raise ValueError(
                    f"Expected the current state as a 1-D ({expected},) vector "
                    f"[eef_pos(3) m, eef_axis_angle(3) rad, gripper_qpos(2) m], got "
                    f"shape {state_arr.shape}. The closed-loop state must match the "
                    "dataset's observation.state (same order, units AND axis-angle "
                    "convention -- see robomemarena_sim/policy_adapter.encode_state)."
                )
            normalized_state = self.normalize_state.normalize(torch.from_numpy(state_arr))
            state_tensor = normalized_state.unsqueeze(0).to(DEVICE)

        gen = self._process(prompt, add_generation_prompt=True)
        prompt_len = gen["input_ids"].shape[1]
        image_kwargs = {k: gen[k] for k in ("pixel_values", "image_grid_thw")}
        full_ids = self.vla.generate_subtask(
            input_ids=gen["input_ids"],
            attention_mask=gen["attention_mask"],
            pixel_values_videos=gen["pixel_values_videos"],
            video_grid_thw=gen["video_grid_thw"],
            mm_token_type_ids=gen["mm_token_type_ids"],
            max_new_tokens=self.max_reasoning_tokens,
            eos_token_id=self.im_end_id,
            pad_token_id=self.pad_token_id,
            suppress_tokens=self.suppress_token_ids or None,
            **image_kwargs,
        )
        reasoning_text = self.processor.tokenizer.decode(
            full_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()

        terminated = int(full_ids[0, -1]) == self.im_end_id
        self.last_decode_truncated = not terminated
        if terminated:
            newline = torch.full(
                (full_ids.shape[0], 1),
                self.newline_id,
                dtype=full_ids.dtype,
                device=full_ids.device,
            )
            full_ids = torch.cat([full_ids, newline], dim=1)
        else:
            self.runaway_decodes += 1
            if not self._runaway_warned_this_episode:
                self._runaway_warned_this_episode = True
                print(
                    f"[policy] WARNING: sub-task decode used all {self.max_reasoning_tokens} "
                    "tokens without emitting <|im_end|>; the DiT condition span is "
                    "UN-TERMINATED (over-long and missing the trailing newline the SFT "
                    "sequence had). Not patched up by design — see the comment above.",
                    flush=True,
                )

        pad = torch.zeros(
            (1, full_ids.shape[1] - prompt_len),
            dtype=gen["mm_token_type_ids"].dtype,
            device=full_ids.device,
        )
        mm_token_type_ids = torch.cat([gen["mm_token_type_ids"], pad], dim=1)
        span = subtask_span_mask(
            full_ids[0], self.think_close_id, self.whitespace_ids
        ).unsqueeze(0)
        normalized_actions = self.vla.predict_action(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            condition_span=span.to(DEVICE),
            pixel_values_videos=gen["pixel_values_videos"],
            video_grid_thw=gen["video_grid_thw"],
            mm_token_type_ids=mm_token_type_ids,
            num_steps=self.num_denoising_steps,
            temperature=self.temperature,
            state=state_tensor,
            **image_kwargs,
        )
        actions = self.unnormalize.unnormalize(
            normalized_actions[0, :, : self.action_dim].float().cpu()
        ).numpy()
        return [actions[i] for i in range(len(actions))], reasoning_text
