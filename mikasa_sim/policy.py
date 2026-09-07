
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
from simplememvla.benchmarks.mikasa import CAMERA_LABELS
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
    has_state_proj = any("action_head.state_proj" in key for key in state_dict)
    if config.use_proprio and not has_state_proj:
        config.use_proprio = False
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
    bad_params = [n for n, p in vla.backbone.named_parameters() if p.dtype != dtype]
    if bad_params:
        raise RuntimeError(
            f"backbone params not cast to {dtype}: {bad_params[:4]}"
        )
    demoted = [
        f"backbone.{name}"
        for name, buf in vla.backbone.named_buffers()
        if "inv_freq" in name and buf.dtype != torch.float32
    ]
    if demoted:
        raise RuntimeError(
            "backbone rotary inv_freq buffers must stay fp32 to reproduce the "
            f"trained M-RoPE phases; got bf16 for {demoted[:4]}"
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


class MikasaPolicy:

    def __init__(
        self,
        vla: SimpleMemVLAForActionPrediction,
        processor: AutoProcessor,
        unnormalize_action: MeanStdNormalizer,
        normalize_state: MeanStdNormalizer | None = None,
        num_denoising_steps: int = 10,
        temperature: float = 1.0,
        max_subtask_tokens: int = 64,
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
        self.max_subtask_tokens = max_subtask_tokens

        tok = processor.tokenizer
        self.think_close_id, self.whitespace_ids = answer_span_token_ids(tok)
        newline_ids = tok.encode("\n", add_special_tokens=False)
        self.newline_id = newline_ids[0] if len(newline_ids) == 1 else None
        self.im_end_id = _token_id(processor, "<|im_end|>")
        self.video_token_id = int(vla.backbone.model.config.video_token_id)
        self.image_token_id = int(vla.backbone.model.config.image_token_id)
        self.pad_token_id = (
            tok.pad_token_id if tok.pad_token_id is not None else self.im_end_id
        )

        cfg = vla.config
        self.image_keys = list(cfg.image_keys)
        self.camera_tags = [key.split(".")[-1] for key in self.image_keys]
        self.robot_tag = cfg.robot_tag
        self.control_frequency_hz = cfg.control_frequency_hz
        native = getattr(cfg, "native_video_fps", None)
        if native is None:
            native = cfg.control_frequency_hz
        self.native_fps = float(native)
        self.n_frames, self.stride = derive_video_sampling(
            cfg.history_video_sec, cfg.history_video_fps, self.native_fps
        )
        self.variable_history = bool(getattr(cfg, "variable_history", False))
        history = getattr(cfg, "history_image_keys", None)
        self.history_image_keys = list(history) if history else list(self.image_keys)
        unknown = [k for k in self.history_image_keys if k not in self.image_keys]
        if unknown:
            raise ValueError(
                f"Checkpoint history_image_keys {unknown} not among image_keys "
                f"{self.image_keys}"
            )
        history_set = set(self.history_image_keys)
        self.history_image_keys = [k for k in self.image_keys if k in history_set]
        self.history_camera_tags = [
            key.split(".")[-1] for key in self.history_image_keys
        ]
        history_set = set(self.history_image_keys)
        self._buffer_caps = {
            k: ((self.n_frames - 1) * self.stride + 1 if k in history_set else 1)
            for k in self.image_keys
        }
        self.action_horizon = int(cfg.action_horizon)
        self.action_dim = int(cfg.action_dim)
        self._buffers: dict[str, list[np.ndarray]] = {k: [] for k in self.image_keys}
        self._frames_seen: dict[str, int] = {k: 0 for k in self.image_keys}
        self._patch_feat_cache: dict[tuple, torch.Tensor] = {}
        self._layout_cache: dict[tuple[str, int], dict] = {}
        self.grid_hw: tuple[int, int] | None = None

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        self._frames_seen = {k: 0 for k in self.image_keys}
        self._patch_feat_cache.clear()

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
        if cur < 0:
            raise RuntimeError("call observe() before predict()")
        n, stride = self.n_frames, self.stride
        if self.variable_history:
            avail = cur // stride + 1
            m = variable_history_frames(avail)
            return [max(0, cur - (m - 1 - i) * stride) for i in range(m)]
        offsets = [-(n - 1 - i) * stride for i in range(n)] if n > 1 else [0]
        return [max(0, cur + off) for off in offsets]

    def _history_video(self, key: str) -> np.ndarray:
        buf = self._buffers[key]
        idxs = self._history_positions(key)
        return np.stack([np.asarray(buf[i]) for i in idxs], axis=0)

    def _visual_inputs(self) -> tuple[list[np.ndarray], list[VideoMetadata], list[np.ndarray]]:
        history = set(self.history_image_keys)
        videos, metas, images = [], [], []
        for key in self.image_keys:
            if key in history:
                clip = self._history_video(key)
                videos.append(clip)
                metas.append(self._clip_metadata(clip.shape[0]))
            else:
                buf = self._buffers[key]
                if not buf:
                    raise RuntimeError("call observe() before predict()")
                images.append(np.asarray(buf[-1]))
        return videos, metas, images

    def _encode(self, prompt: str, videos, metas, images, add_generation_prompt: bool):
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

    def _process(self, prompt: str, add_generation_prompt: bool):
        return self._encode(prompt, *self._visual_inputs(), add_generation_prompt)

    def _clip_metadata(self, n_frames: int) -> VideoMetadata:
        sampled_fps = self.native_fps / self.stride
        return VideoMetadata(
            total_num_frames=n_frames,
            fps=sampled_fps,
            frames_indices=list(range(n_frames)),
            duration=n_frames / sampled_fps,
            video_backend="rollout",
        )


    def _grid_global_indices(self, cur_global: int) -> list[int]:
        n, stride = self.n_frames, self.stride
        if self.variable_history:
            avail = min(cur_global // stride + 1, n)
            m = variable_history_frames(avail)
            return [max(0, cur_global - (m - 1 - i) * stride) for i in range(m)]
        offsets = [-(n - 1 - i) * stride for i in range(n)] if n > 1 else [0]
        return [max(0, cur_global + off) for off in offsets]

    def _history_global_indices(self, key: str) -> list[int]:
        base = self._frames_seen[key] - len(self._buffers[key])
        return [base + pos for pos in self._history_positions(key)]

    @staticmethod
    def _patch_keys_from_indices(tag: str, gidx: list[int]) -> list[tuple]:
        if len(gidx) % 2:
            raise RuntimeError("history clip length must be even (temporal_patch_size=2)")
        return [(tag, gidx[j], gidx[j + 1]) for j in range(0, len(gidx), 2)]

    def _video_patch_keys(self) -> list[tuple]:
        keys: list[tuple] = []
        for key in self.history_image_keys:
            keys.extend(self._patch_keys_from_indices(
                key.split(".")[-1], self._history_global_indices(key)))
        return keys

    def future_patch_keys(self, steps_ahead: int) -> list[tuple]:
        keys: list[tuple] = []
        for key in self.history_image_keys:
            tag = key.split(".")[-1]
            cur = self._frames_seen[key] - 1
            if cur < 0:
                raise RuntimeError("call observe() before future_patch_keys()")
            if self._patch_keys_from_indices(tag, self._grid_global_indices(cur)) != \
                    self._patch_keys_from_indices(tag, self._history_global_indices(key)):
                raise RuntimeError(
                    "history-window rule mismatch: _grid_global_indices disagrees "
                    "with the live clip at steps_ahead=0"
                )
            keys.extend(self._patch_keys_from_indices(
                tag, self._grid_global_indices(cur + int(steps_ahead))))
        return keys

    def _preprocess_patch_pixels(self, patch_key: tuple) -> torch.Tensor:
        tag, ga, gb = patch_key
        cam = next(k for k in self.history_image_keys if k.split(".")[-1] == tag)
        buf = self._buffers[cam]
        base = self._frames_seen[cam] - len(buf)
        clip = np.stack([np.asarray(buf[ga - base]), np.asarray(buf[gb - base])], axis=0)
        out = self.processor.video_processor(
            videos=[clip], do_sample_frames=False, return_tensors="pt"
        )
        grid = out["video_grid_thw"][0].tolist()
        if self.grid_hw is not None and (grid[0] != 1 or grid[1:] != list(self.grid_hw)):
            raise RuntimeError(
                f"single-patch preprocess grid {grid} != cached full-clip grid "
                f"[1, {self.grid_hw[0]}, {self.grid_hw[1]}]"
            )
        return out["pixel_values_videos"].to(DEVICE)

    def _wrist_pixels(self) -> dict[str, torch.Tensor]:
        history = set(self.history_image_keys)
        images = []
        for key in self.image_keys:
            if key in history:
                continue
            buf = self._buffers[key]
            if not buf:
                raise RuntimeError("call observe() before predict()")
            images.append(np.asarray(buf[-1]))
        if not images:
            return {}
        out = self.processor.image_processor(images=images, return_tensors="pt")
        return {
            "pixel_values": out["pixel_values"].to(DEVICE),
            "image_grid_thw": out["image_grid_thw"].to(DEVICE),
        }

    def _layout(self, prompt: str, n_slots: int) -> dict:
        cached = self._layout_cache.get((prompt, n_slots))
        if cached is not None:
            return cached

        history = set(self.history_image_keys)
        videos, metas, images = [], [], []
        for key in self.image_keys:
            buf = self._buffers[key]
            if not buf:
                raise RuntimeError("call observe() before predict()")
            if key in history:
                videos.append(np.stack(
                    [np.asarray(buf[i % len(buf)]) for i in range(2 * n_slots)]))
                metas.append(self._clip_metadata(2 * n_slots))
            else:
                images.append(np.asarray(buf[-1]))
        enc = self._encode(prompt, videos, metas, images, True)

        grids = enc["video_grid_thw"]
        if int(grids.shape[0]) != len(self.history_image_keys):
            raise RuntimeError("one history video per history camera expected")
        gh = gw = None
        for v in range(grids.shape[0]):
            t, h, w = (int(x) for x in grids[v])
            if t != n_slots:
                raise RuntimeError(f"video {v} has {t} temporal patches, expected {n_slots}")
            if gh is None:
                gh, gw = h, w
            elif (h, w) != (gh, gw):
                raise RuntimeError("history cameras must share one frame size")
        if self.grid_hw is None:
            self.grid_hw = (gh, gw)
            rows = gh * gw
            probe = self.processor.video_processor(
                videos=[videos[0][:2]], do_sample_frames=False, return_tensors="pt"
            )["pixel_values_videos"].to(DEVICE)
            if not torch.equal(probe, enc["pixel_values_videos"][:rows]):
                self.grid_hw = None
                raise RuntimeError(
                    "single-patch video preprocessing does not reproduce the "
                    "full-clip pixel rows (smart_resize resolution-budget-bound "
                    "at this camera size)"
                )
        elif (gh, gw) != self.grid_hw:
            raise RuntimeError(f"frame grid changed mid-run: {self.grid_hw} -> {(gh, gw)}")

        ids = enc["input_ids"]
        position_ids, deltas = self.vla.backbone.model.get_rope_index(
            ids,
            image_grid_thw=enc.get("image_grid_thw"),
            video_grid_thw=grids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=enc["mm_token_type_ids"],
        )
        vpos = (ids[0] == self.video_token_id).nonzero().squeeze(-1)
        total_slots = n_slots * len(self.history_image_keys)
        if vpos.numel() % total_slots:
            raise RuntimeError(
                f"{vpos.numel()} video tokens do not tile into {total_slots} patches"
            )
        per = vpos.numel() // total_slots
        ipos = (ids[0] == self.image_token_id).nonzero().squeeze(-1)
        if ipos.numel() and int(ipos.min()) < int(vpos[-1]):
            raise RuntimeError(
                "current-frame image tokens appear before the end of the history "
                "video block; the pipelined decision path needs every history "
                "camera to precede every current-frame camera in image_keys"
            )
        layout = {
            "prompt": prompt,
            "n_slots": n_slots,
            "input_ids": ids,
            "mm_token_type_ids": enc["mm_token_type_ids"],
            "video_grid_thw": grids,
            "image_grid_thw": enc.get("image_grid_thw"),
            "position_ids": position_ids,
            "rope_deltas": deltas,
            "slot_rows": [vpos[per * s: per * (s + 1)] for s in range(total_slots)],
            "tokens_per_slot": per,
        }
        self._layout_cache[(prompt, n_slots)] = layout
        return layout

    def _prepare_inputs(self, prompt: str) -> dict:
        keys = self._video_patch_keys()
        n_cams = len(self.history_image_keys)
        if len(keys) % n_cams:
            raise RuntimeError("history cameras disagree on clip length")
        layout = self._layout(prompt, len(keys) // n_cams)

        new_pixels = {
            k: self._preprocess_patch_pixels(k)
            for k in keys if k not in self._patch_feat_cache
        }
        live = set(keys)
        for k in [k for k in self._patch_feat_cache if k not in live]:
            del self._patch_feat_cache[k]

        out = {
            "policy": self,
            "layout": layout,
            "video_patch_keys": keys,
            "video_patch_pixels": new_pixels,
        }
        out.update(self._wrist_pixels())
        return out

    def _state_tensor(self, state: np.ndarray | None) -> torch.Tensor | None:
        if not self.use_proprio:
            return None
        if state is None:
            raise ValueError(
                "This checkpoint was trained with proprioception; predict() "
                "requires the current `state` in the dataset's "
                "observation.state layout ([qpos(7), gripper width])."
            )
        state_arr = np.asarray(state, dtype=np.float32)
        expected = int(self.vla.config.state_dim)
        if state_arr.ndim != 1 or state_arr.shape[0] != expected:
            raise ValueError(
                f"Expected the current state as a 1-D ({expected},) vector in the "
                f"dataset observation.state layout, got shape {state_arr.shape}. "
                "The closed-loop state must match the dataset's observation.state "
                "(same order/units: qpos radians, gripper width metres)."
            )
        normalized_state = self.normalize_state.normalize(torch.from_numpy(state_arr))
        return normalized_state.unsqueeze(0).to(DEVICE)

    def unnormalize_chunk(self, normalized_actions: torch.Tensor) -> list[np.ndarray]:
        actions = self.unnormalize.unnormalize(
            normalized_actions[0, :, : self.action_dim].float().cpu()
        ).numpy()
        return [actions[i] for i in range(len(actions))]

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        observe: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], str]:
        if observe is not None:
            self.observe(observe)

        state_tensor = self._state_tensor(state)

        gen = self._process(prompt, add_generation_prompt=True)
        prompt_len = gen["input_ids"].shape[1]
        image_kwargs = {
            k: gen[k] for k in ("pixel_values", "image_grid_thw") if k in gen
        }
        full_ids = self.vla.generate_subtask(
            input_ids=gen["input_ids"],
            attention_mask=gen["attention_mask"],
            pixel_values_videos=gen["pixel_values_videos"],
            video_grid_thw=gen["video_grid_thw"],
            mm_token_type_ids=gen["mm_token_type_ids"],
            max_new_tokens=self.max_subtask_tokens,
            eos_token_id=self.im_end_id,
            pad_token_id=self.pad_token_id,
            **image_kwargs,
        )
        subtask_text = self.processor.tokenizer.decode(
            full_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()

        if int(full_ids[0, -1]) != self.im_end_id:
            print(
                f"[policy] WARNING: sub-task decode hit the "
                f"{self.max_subtask_tokens}-token budget without <|im_end|>; "
                "forcing a terminator (degenerate sub-task).",
                flush=True,
            )
            full_ids = torch.cat(
                [
                    full_ids,
                    torch.full((full_ids.shape[0], 1), self.im_end_id,
                               dtype=full_ids.dtype, device=full_ids.device),
                ],
                dim=1,
            )

        if self.newline_id is not None and int(full_ids[0, -1]) == self.im_end_id:
            newline = torch.full(
                (full_ids.shape[0], 1),
                self.newline_id,
                dtype=full_ids.dtype,
                device=full_ids.device,
            )
            full_ids = torch.cat([full_ids, newline], dim=1)

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
        return self.unnormalize_chunk(normalized_actions), subtask_text
