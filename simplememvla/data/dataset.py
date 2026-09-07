from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from transformers.video_utils import VideoMetadata

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from simplememvla.benchmarks import BenchmarkSpec
from simplememvla.data.augment import ClipAugmenter
from simplememvla.data.messages import (
    build_simplememvla_messages,
    derive_video_sampling,
    variable_history_frames,
)
from simplememvla.data.normalize import MeanStdNormalizer, load_action_stats, load_state_stats


def _frames_to_video(value: Any) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Cannot convert {type(value)} to a video clip")
    tensor = value.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W) image stack, got {tuple(tensor.shape)}")
    tensor = tensor.permute(0, 2, 3, 1)
    if tensor.dtype.is_floating_point:
        tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.uint8)
    return tensor.numpy()


def _history_offsets(n_frames: int, stride: int) -> list[int]:
    if n_frames <= 1:
        return [0]
    return [-(n_frames - 1 - i) * stride for i in range(n_frames)]


def _resolve_delta_timestamps(data_args, ds_meta, offsets, history_keys):
    delta_timestamps = {}
    if data_args.action_delta_indices is not None:
        delta_timestamps[data_args.action_key] = [
            i / ds_meta.fps for i in data_args.action_delta_indices
        ]
    if len(offsets) > 1:
        for key in history_keys:
            delta_timestamps[key] = [o / ds_meta.fps for o in offsets]
    return delta_timestamps or None


def _flat_bool_column(hf_dataset, name: str) -> np.ndarray:
    col = hf_dataset.data.column(name).combine_chunks()
    if hasattr(col, "flatten") and not isinstance(col, pa.BooleanArray):
        try:
            col = col.flatten()
        except (NotImplementedError, pa.ArrowInvalid):
            pass
    arr = np.asarray(col.to_pylist())
    return arr.reshape(len(hf_dataset), -1)[:, 0].astype(bool)


class SimpleMemVLADataset(Dataset):

    def __init__(self, data_args, spec: BenchmarkSpec):
        self.data_args = data_args
        self.spec = spec
        self.root = Path(data_args.root)
        self.ds_meta = LeRobotDatasetMetadata(
            data_args.repo_id,
            root=self.root,
            revision=data_args.revision,
        )
        self.native_fps = float(
            data_args.native_video_fps
            if data_args.native_video_fps is not None
            else spec.native_video_fps
        )
        self.n_frames, self.stride = derive_video_sampling(
            data_args.history_video_sec,
            data_args.history_video_fps,
            self.native_fps,
        )
        self.variable_history = bool(data_args.variable_history)
        requested_history = data_args.history_image_keys
        if requested_history is None:
            self.history_image_keys = list(data_args.image_keys)
        else:
            self.history_image_keys = list(requested_history)
        if not self.history_image_keys:
            raise ValueError("history_image_keys must name at least one camera")
        unknown = [
            k for k in self.history_image_keys if k not in data_args.image_keys
        ]
        if unknown:
            raise ValueError(
                f"history_image_keys {unknown} not among image_keys "
                f"{data_args.image_keys}"
            )
        offsets = _history_offsets(self.n_frames, self.stride)
        delta_timestamps = _resolve_delta_timestamps(
            data_args, self.ds_meta, offsets, self.history_image_keys
        )
        self.dataset = LeRobotDataset(
            data_args.repo_id,
            root=self.root,
            episodes=data_args.episodes,
            delta_timestamps=delta_timestamps,
            revision=data_args.revision,
            download_videos=data_args.download_videos,
            video_backend=data_args.video_backend,
        )
        self.action_stats = load_action_stats(self.root)
        self.normalizer = MeanStdNormalizer(self.action_stats)
        self.state_stats = load_state_stats(self.root)
        self.state_normalizer = MeanStdNormalizer(self.state_stats)
        self._anchor_indices: np.ndarray | None = None
        skip_demo = bool(getattr(data_args, "skip_video_demo_frames", False))
        if skip_demo:
            if "is_video_demo" not in self.ds_meta.features:
                raise ValueError(
                    "skip_video_demo_frames=True but the dataset has no "
                    "`is_video_demo` feature; this knob is only meaningful for "
                    "datasets with a replayed conditioning-demo segment (RoboMME)."
                )
            hf_dataset = getattr(self.dataset.reader, "hf_dataset", None)
            if hf_dataset is None:
                raise RuntimeError(
                    "skip_video_demo_frames=True but the LeRobot reader exposes no "
                    "loaded hf_dataset; cannot build the execution-frame anchor "
                    "filter."
                )
            is_demo = _flat_bool_column(hf_dataset, "is_video_demo")
            if len(is_demo) != len(self.dataset):
                raise ValueError(
                    f"is_video_demo column length {len(is_demo)} != dataset length "
                    f"{len(self.dataset)}"
                )
            self._anchor_indices = np.nonzero(~is_demo)[0]
            if len(self._anchor_indices) == 0:
                raise ValueError(
                    "skip_video_demo_frames left no execution frames to train on"
                )
        self.subtask_key = str(data_args.subtask_key)
        self.subtask_index_key = str(
            getattr(data_args, "subtask_index_key", None) or "subtask_index"
        )
        if self.subtask_index_key != "subtask_index":
            if self.subtask_index_key not in self.ds_meta.features:
                raise ValueError(
                    f"subtask_index_key {self.subtask_index_key!r} is not a dataset "
                    f"feature; expected one of the subtask index columns."
                )
            if self.ds_meta.subtasks is None:
                raise ValueError(
                    "Dataset has no meta/subtasks.parquet; cannot resolve "
                    f"{self.subtask_index_key!r}."
                )
        if (
            self.ds_meta.subtasks is not None
            and "subtask_index" in getattr(self.ds_meta.subtasks, "columns", ())
        ):
            col = self.ds_meta.subtasks["subtask_index"].to_numpy()
            if not (col == np.arange(len(col))).all():
                raise ValueError(
                    "meta/subtasks.parquet rows are not ordered by subtask_index "
                    "(row position != index value); the positional string "
                    "resolution would silently mislabel sub-tasks. Re-sort the "
                    "parquet."
                )
        self.control_frequency_hz = (
            data_args.control_frequency_hz
            if data_args.control_frequency_hz is not None
            else round(self.native_fps)
        )
        self.camera_tags = [key.split(".")[-1] for key in data_args.image_keys]
        self.history_camera_tags = [
            key.split(".")[-1] for key in self.history_image_keys
        ]
        self.augmenter = ClipAugmenter() if data_args.image_aug else None

    def __len__(self):
        if self._anchor_indices is not None:
            return len(self._anchor_indices)
        return len(self.dataset)

    def _normalize_text(self, text: Any) -> str:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        if self.spec.verbatim_text:
            return str(text)
        return str(text).strip().replace("_", " ")

    def _get_task(self, item: dict[str, Any]) -> str:
        if self.data_args.task_key not in item:
            raise KeyError(f"Missing task key {self.data_args.task_key!r} in sample")
        task = self._normalize_text(item[self.data_args.task_key])
        if not task:
            raise ValueError(f"{self.spec.name} task instruction is empty")
        return task

    def _get_subtask(self, item: dict[str, Any]) -> str:
        if self.subtask_index_key != "subtask_index":
            idx = int(torch.as_tensor(item[self.subtask_index_key]).reshape(-1)[0])
            subtask = self.ds_meta.subtasks.iloc[idx].name
        else:
            key = self.subtask_key
            if key not in item:
                raise KeyError(
                    f"Missing sub-task key {key!r} in sample. SimpleMemVLA resolves the "
                    "per-frame sub-task as the LeRobot 'subtask' string (from "
                    "subtask_index + meta/subtasks.parquet); use a dataset that "
                    "carries it."
                )
            subtask = item[key]
        subtask = self._normalize_text(subtask)
        if not subtask:
            raise ValueError(f"{self.spec.name} sub-task string is empty")
        return subtask

    def _video_metadata(self, num_frames: int) -> VideoMetadata:
        sampled_fps = self.native_fps / self.stride
        return VideoMetadata(
            total_num_frames=num_frames,
            fps=sampled_fps,
            frames_indices=list(range(num_frames)),
            duration=num_frames / sampled_fps,
            video_backend="lerobot",
        )

    def _get_visuals(
        self, item: dict[str, Any]
    ) -> tuple[list[np.ndarray], list[VideoMetadata], list[np.ndarray]]:
        videos, metas, images, missing = [], [], [], []
        history = set(self.history_image_keys)
        for key in self.data_args.image_keys:
            if key not in item:
                missing.append(key)
                continue
            clip = _frames_to_video(item[key])
            if key in history:
                if clip.shape[0] != self.n_frames:
                    raise ValueError(
                        f"History camera {key!r} returned {clip.shape[0]} frame(s); "
                        f"expected {self.n_frames}."
                    )
                if self.variable_history:
                    pad_key = f"{key}_is_pad"
                    if pad_key not in item:
                        raise KeyError(
                            f"Missing LeRobot padding key {pad_key!r}; cannot crop "
                            "the history clip for variable_history."
                        )
                    is_pad = torch.as_tensor(item[pad_key], dtype=torch.bool)
                    if is_pad.shape[0] != self.n_frames:
                        raise ValueError(
                            f"{pad_key} has {is_pad.shape[0]} entries; expected "
                            f"{self.n_frames}."
                        )
                    num_real = int((~is_pad).sum())
                    if num_real == 0 or bool(is_pad[-num_real:].any()):
                        raise ValueError(
                            f"{pad_key} is not a True-prefix mask; history "
                            "deltas must be ascending and <= 0 for tail-crop."
                        )
                    clip = clip[-variable_history_frames(num_real):]
                videos.append(clip)
                metas.append(self._video_metadata(clip.shape[0]))
            else:
                if clip.shape[0] != 1:
                    raise ValueError(
                        f"Current-frame camera {key!r} returned {clip.shape[0]} "
                        "frame(s); expected 1. Do not configure delta_timestamps "
                        "for it."
                    )
                images.append(clip[0])
        if missing:
            raise KeyError(
                f"Missing configured image keys in sample: {missing}. "
                f"Expected all of {self.data_args.image_keys}."
            )
        return videos, metas, images

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._anchor_indices is not None:
            idx = int(self._anchor_indices[idx])
        item = self.dataset[idx]
        if self.data_args.action_key not in item:
            raise KeyError(f"Missing action key {self.data_args.action_key!r} in sample")
        actions = torch.as_tensor(item[self.data_args.action_key], dtype=torch.float32)
        actions = self.normalizer.normalize(actions)
        expected_shape = (self.spec.num_actions_chunk, self.spec.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"Expected actions chunk of shape {expected_shape}, "
                f"got {tuple(actions.shape)}"
            )
        pad_key = f"{self.data_args.action_key}_is_pad"
        if pad_key not in item:
            raise KeyError(
                f"Missing LeRobot padding key {pad_key!r}; cannot mask episode-tail actions"
            )
        valid_steps = ~torch.as_tensor(item[pad_key], dtype=torch.bool)
        action_mask = valid_steps.unsqueeze(-1).repeat(1, self.spec.action_dim)
        if self.data_args.state_key not in item:
            raise KeyError(f"Missing state key {self.data_args.state_key!r} in sample")
        state = torch.as_tensor(item[self.data_args.state_key], dtype=torch.float32)
        if tuple(state.shape) != (self.spec.state_dim,):
            raise ValueError(
                f"Expected a single current state of shape ({self.spec.state_dim},), "
                f"got {tuple(state.shape)}. Do not configure delta_timestamps for "
                f"{self.data_args.state_key!r}."
            )
        state = self.state_normalizer.normalize(state)
        videos, video_metadata, images = self._get_visuals(item)
        if self.augmenter is not None:
            params = self.augmenter.draw()
            videos = [self.augmenter.apply(clip, params) for clip in videos]
            images = [self.augmenter.apply(img, params) for img in images]
        messages = build_simplememvla_messages(
            camera_tags=self.camera_tags,
            instruction=self._get_task(item),
            robot_tag=self.data_args.robot_tag,
            control_frequency_hz=self.control_frequency_hz,
            action_horizon=self.spec.num_actions_chunk,
            camera_labels=self.spec.camera_labels,
            subtask=self._get_subtask(item),
            history_camera_tags=self.history_camera_tags,
        )
        return {
            "messages": messages,
            "videos": videos,
            "video_metadata": video_metadata,
            "images": images,
            "actions": actions,
            "action_mask": action_mask,
            "state": state,
        }
