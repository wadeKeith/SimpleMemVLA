from typing import Any


def resolve_camera_label(raw_tag: str, camera_labels: dict[str, str]) -> str:
    try:
        return camera_labels[raw_tag]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported camera tag {raw_tag!r}. Expected one of "
            f"{sorted(camera_labels)}."
        ) from exc


def derive_video_sampling(
    history_video_sec: float, history_video_fps: float, native_fps: float
) -> tuple[int, int]:
    n_frames = max(2, round(history_video_sec * history_video_fps))
    if n_frames % 2:
        n_frames += 1
    stride = max(1, round(native_fps / history_video_fps))
    return n_frames, stride


def variable_history_frames(num_real_frames: int) -> int:
    num = int(num_real_frames)
    return max(2, num + (num % 2))


def build_embodiment_prompt(
    instruction: str,
    robot_tag: str,
    control_frequency_hz: int,
    action_horizon: int,
    num_cameras: int = 1,
    num_current_cameras: int = 0,
) -> str:
    instruction = instruction.strip().replace("_", " ").rstrip(". ")
    if not instruction:
        raise ValueError("instruction must be a non-empty string")
    if num_current_cameras:
        history_part = (
            f"the recent history video from {num_cameras} cameras"
            if num_cameras and num_cameras > 1
            else "the recent history video from one camera"
        )
        current_part = (
            f"{num_current_cameras} more cameras"
            if num_current_cameras > 1
            else "one more camera"
        )
        cameras_clause = (
            f"You are given {history_part} "
            "(each frame is tagged with its timestamp in seconds), "
            f"plus the current image from {current_part}. "
        )
    else:
        cameras_clause = (
            f"You are given the recent history video from {num_cameras} cameras "
            "(each frame is tagged with its timestamp in seconds). "
            if num_cameras and num_cameras > 1
            else "You are given the recent history video (frames tagged with timestamps). "
        )
    return (
        f"The robot is a {robot_tag}. "
        f"The control frequency is {control_frequency_hz} Hz. "
        f"The next {action_horizon} control actions are predicted from the current sub-task. "
        f"{cameras_clause}"
        f"The overall task is: {instruction}. "
        f"Identify the current sub-task."
    )


def build_simplememvla_messages(
    camera_tags: list[str],
    instruction: str,
    robot_tag: str,
    control_frequency_hz: int,
    action_horizon: int,
    camera_labels: dict[str, str],
    subtask: str | None = None,
    history_camera_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    if history_camera_tags is None:
        history_camera_tags = list(camera_tags)
    unknown = [t for t in history_camera_tags if t not in camera_tags]
    if unknown:
        raise ValueError(
            f"history_camera_tags {unknown} not among camera_tags {camera_tags}"
        )
    history = set(history_camera_tags)
    current_tags = [t for t in camera_tags if t not in history]
    content: list[dict[str, Any]] = []
    for tag in camera_tags:
        content.append(
            {"type": "text", "text": f"{resolve_camera_label(tag, camera_labels)}:"}
        )
        content.append({"type": "video" if tag in history else "image"})
    content.append(
        {
            "type": "text",
            "text": build_embodiment_prompt(
                instruction=instruction,
                robot_tag=robot_tag,
                control_frequency_hz=control_frequency_hz,
                action_horizon=action_horizon,
                num_cameras=len(camera_tags) - len(current_tags),
                num_current_cameras=len(current_tags),
            ),
        }
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if subtask:
        messages.append({"role": "assistant", "content": subtask})
    return messages
