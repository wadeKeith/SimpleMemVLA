# -*- coding: utf-8 -*-
# Common utility for saving videos with captions during reset phase (demonstration phase).

import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import cv2
import imageio
import torch

from ...logging_utils import logger

TEXT_AREA_HEIGHT = 60

# Tasks without subgoal demonstration: do not add red border when saving video (no demonstration phase to highlight)
NO_HIGHLIGHT_BORDER_ENV_IDS = frozenset({
    "SwingXtimes",
    "PickXtimes",
    "ButtonUnmaskSwap",
    "StopCube",
    "PickHighlight",
    "ButtonUnmask",
    "BinFill",
})


def _frame_to_numpy(frame: Any) -> np.ndarray:
    """Convert frame-like input to CPU numpy array for OpenCV/imageio writing."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach()
        if frame.is_cuda:
            frame = frame.cpu()
        frame = frame.numpy()
    else:
        frame = np.asarray(frame)
    return frame


def _flatten_column(batch_dict: Dict[str, List[Any]], key: str) -> List[Any]:
    out = []
    for item in (batch_dict or {}).get(key, []) or []:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend([x for x in item if x is not None])
        else:
            out.append(item)
    return out


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return frame


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    if frame.shape[0] == target_h:
        return frame
    scale = target_h / max(1, frame.shape[0])
    target_w = max(1, int(round(frame.shape[1] * scale)))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _concat_horizontal(frames: List[Any]) -> np.ndarray:
    rgb_frames = [_ensure_rgb(_frame_to_numpy(frame)) for frame in frames if frame is not None]
    if not rgb_frames:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    target_h = rgb_frames[0].shape[0]
    aligned = [_resize_to_height(frame, target_h) for frame in rgb_frames]
    return np.hstack(aligned)


def add_text_to_frame(
    frame: np.ndarray,
    text: Any,
    text_area_height: int = TEXT_AREA_HEIGHT,
) -> np.ndarray:
    """Overlay caption at the top of the frame (white text on black background), style consistent with DemonstrationWrapper.save_video."""
    frame = _frame_to_numpy(frame).copy()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if text is None:
        text = ""
    if isinstance(text, (list, tuple)):
        text = " | ".join(str(t).strip() for t in text if t)
    text = str(text).strip()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    max_width = max(1, frame.shape[1] - 20)
    lines = []
    if text:
        words = text.replace(",", " ").split()
        if words:
            current_line = words[0]
            for word in words[1:]:
                test_line = f"{current_line} {word}"
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
    if not lines:
        text_area = np.zeros((text_area_height, frame.shape[1], 3), dtype=np.uint8)
        return np.vstack((text_area, frame))
    line_height = 20
    text_area = np.zeros((text_area_height, frame.shape[1], 3), dtype=np.uint8)
    text_area[:] = (0, 0, 0)
    max_visible_lines = (text_area_height - 15) // line_height
    for i, line in enumerate(lines[:max_visible_lines]):
        y_position = 15 + i * line_height
        cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)
    return np.vstack((text_area, frame))


def add_border_to_frame(
    frame: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 4,
) -> np.ndarray:
    """Overlay border around the frame (RGB color)."""
    frame = _frame_to_numpy(frame).copy()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return frame

    h, w = frame.shape[:2]
    t = max(1, int(thickness))
    t = min(t, h // 2 if h > 1 else 1, w // 2 if w > 1 else 1)

    frame[:t, :, :] = color
    frame[-t:, :, :] = color
    frame[:, :t, :] = color
    frame[:, -t:, :] = color
    return frame


def save_robomme_video(
    reset_base_frames: List[Any],
    reset_wrist_frames: List[Any],
    rollout_base_frames: List[Any],
    rollout_wrist_frames: List[Any],
    reset_subgoal_grounded: List[Any],
    rollout_subgoal_grounded: List[Any],
    out_video_dir: str,
    action_space: str,
    env_id: str,
    episode: int,
    episode_success: bool,
    reset_right_frames: Optional[List[Any]] = None,
    rollout_right_frames: Optional[List[Any]] = None,
    reset_far_right_frames: Optional[List[Any]] = None,
    rollout_far_right_frames: Optional[List[Any]] = None,
    fps: int = 20,
    highlight_color: Tuple[int, int, int] = (255, 0, 0),
    highlight_thickness: int = 4,
) -> bool:
    """
    Unified method to save replay videos (including reset prefix highlighting, naming, and output path concatenation).
    Whether to draw the red border on reset-phase frames is determined by env_id: tasks in NO_HIGHLIGHT_BORDER_ENV_IDS (no subgoal demonstration) do not get the border.

    Args:
        reset_base_frames/reset_wrist_frames/reset_right_frames/reset_far_right_frames: List of camera frames from reset phase.
        rollout_base_frames/rollout_wrist_frames/rollout_right_frames/rollout_far_right_frames: List of camera frames from rollout phase.
        reset_subgoal_grounded/rollout_subgoal_grounded: List of captions for corresponding phases.
        out_video_dir: Output directory.
        action_space: Current action space, used for filename prefix.
        env_id: Environment ID (used to decide if highlight border is drawn; see NO_HIGHLIGHT_BORDER_ENV_IDS).
        episode: Current episode index.
        episode_success: Whether the current episode was successful.
        fps: Output frame rate.
        highlight_color: Border color (RGB).
        highlight_thickness: Border thickness (pixels).

    Returns:
        True if at least one frame was written, False otherwise.
    """
    success_prefix = "success" if episode_success else "fail"
    mode_prefix = action_space
    out_video_path = os.path.join(
        out_video_dir,
        f"{success_prefix}_replay_{mode_prefix}_{env_id}_ep{episode}.mp4",
    )

    reset_base_frames = list(reset_base_frames or [])
    reset_wrist_frames = list(reset_wrist_frames or [])
    reset_right_frames = list(reset_right_frames or [])
    reset_far_right_frames = list(reset_far_right_frames or [])
    rollout_base_frames = list(rollout_base_frames or [])
    rollout_wrist_frames = list(rollout_wrist_frames or [])
    rollout_right_frames = list(rollout_right_frames or [])
    rollout_far_right_frames = list(rollout_far_right_frames or [])
    reset_subgoal_grounded = list(reset_subgoal_grounded or [])
    rollout_subgoal_grounded = list(rollout_subgoal_grounded or [])

    merged_base_frames = reset_base_frames + rollout_base_frames
    merged_wrist_frames = reset_wrist_frames + rollout_wrist_frames
    merged_right_frames = reset_right_frames + rollout_right_frames
    merged_far_right_frames = reset_far_right_frames + rollout_far_right_frames
    merged_subgoal_grounded = reset_subgoal_grounded + rollout_subgoal_grounded

    if not (
        merged_base_frames
        or merged_wrist_frames
        or merged_right_frames
        or merged_far_right_frames
    ):
        logger.debug(f"Skipped video (no frames): {out_video_path}")
        return False

    base_camera = [_frame_to_numpy(f) for f in merged_base_frames if f is not None]
    wrist_camera = [_frame_to_numpy(f) for f in merged_wrist_frames if f is not None]
    right_camera = [_frame_to_numpy(f) for f in merged_right_frames if f is not None]
    far_right_camera = [_frame_to_numpy(f) for f in merged_far_right_frames if f is not None]
    reset_base_camera = [_frame_to_numpy(f) for f in reset_base_frames if f is not None]
    reset_wrist_camera = [_frame_to_numpy(f) for f in reset_wrist_frames if f is not None]
    reset_right_camera = [_frame_to_numpy(f) for f in reset_right_frames if f is not None]
    reset_far_right_camera = [
        _frame_to_numpy(f) for f in reset_far_right_frames if f is not None
    ]

    merged_feeds: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
    if base_camera:
        merged_feeds.append((base_camera, reset_base_camera))
    if wrist_camera:
        merged_feeds.append((wrist_camera, reset_wrist_camera))
    if right_camera:
        merged_feeds.append((right_camera, reset_right_camera))
    if far_right_camera:
        merged_feeds.append((far_right_camera, reset_far_right_camera))
    if not merged_feeds:
        logger.debug(f"Skipped video (no valid frames): {out_video_path}")
        return False

    n_frames = min(len(feed) for feed, _ in merged_feeds)
    if n_frames <= 0:
        logger.debug(f"Skipped video (no aligned frames): {out_video_path}")
        return False
    image = [
        _concat_horizontal([feed[i] for feed, _ in merged_feeds])
        for i in range(n_frames)
    ]
    reset_frame_count = min(len(reset_feed) for _, reset_feed in merged_feeds)
    reset_frame_count = max(0, min(reset_frame_count, n_frames))

    draw_highlight_border = env_id not in NO_HIGHLIGHT_BORDER_ENV_IDS
    highlight_prefix_count = reset_frame_count if draw_highlight_border else 0

    subgoal_grounded = [text for text in merged_subgoal_grounded if text is not None]

    out_dir = os.path.dirname(os.path.abspath(out_video_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with imageio.get_writer(out_video_path, fps=fps, codec="libx264", quality=8) as writer:
        for i in range(n_frames):
            frame = _frame_to_numpy(image[i])
            caption = subgoal_grounded[i] if i < len(subgoal_grounded) else ""
            combined = add_text_to_frame(frame, caption)
            if i < max(0, int(highlight_prefix_count)):
                combined = add_border_to_frame(
                    combined,
                    color=highlight_color,
                    thickness=highlight_thickness,
                )
            writer.append_data(combined)

    logger.debug(f"Saved video: {out_video_path}")
    return True
