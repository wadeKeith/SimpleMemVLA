
from simplememvla.benchmarks import BenchmarkSpec

NUM_ACTIONS_CHUNK = 16
ACTION_DIM = 8
STATE_DIM = 8

CONTROL_FREQ_HZ = 20
NATIVE_VIDEO_FPS = float(CONTROL_FREQ_HZ)

ROBOT_TAG = "Panda single-arm robot"

CAMERA_LABELS = {
    "top": "Top camera",
    "base_camera": "Top camera",
    "wrist": "Wrist camera",
    "wrist_camera": "Wrist camera",
    "hand_camera": "Wrist camera",
}

SPEC = BenchmarkSpec(
    name="mikasa",
    num_actions_chunk=NUM_ACTIONS_CHUNK,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    robot_tag=ROBOT_TAG,
    native_video_fps=NATIVE_VIDEO_FPS,
    camera_labels=CAMERA_LABELS,
    repo_id="yinchenghust/mikasa_lerobot",
    root="./data/datasets/yinchenghust/mikasa_lerobot",
    image_keys=(
        "observation.images.top",
        "observation.images.wrist",
    ),
    history_image_keys=("observation.images.top",),
    history_video_sec=3.0,
    history_video_fps=20.0,
    variable_history=True,
    image_aug=True,
)
