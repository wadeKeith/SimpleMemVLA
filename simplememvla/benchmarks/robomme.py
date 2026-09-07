
from simplememvla.benchmarks import BenchmarkSpec

NUM_ACTIONS_CHUNK = 30
ACTION_DIM = 8
STATE_DIM = 8

CONTROL_FREQ_HZ = 20
NATIVE_VIDEO_FPS = float(CONTROL_FREQ_HZ)

ROBOT_TAG = "Panda single-arm robot"

CAMERA_LABELS = {
    "front": "Front camera",
    "front_camera": "Front camera",
    "base_camera": "Front camera",
    "wrist": "Wrist camera",
    "wrist_camera": "Wrist camera",
    "hand_camera": "Wrist camera",
}

SPEC = BenchmarkSpec(
    name="robomme",
    num_actions_chunk=NUM_ACTIONS_CHUNK,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    robot_tag=ROBOT_TAG,
    native_video_fps=NATIVE_VIDEO_FPS,
    camera_labels=CAMERA_LABELS,
    repo_id="yinchenghust/robomme_lerobot",
    root="./data/datasets/yinchenghust/robomme_lerobot",
    image_keys=(
        "observation.images.front",
        "observation.images.wrist",
    ),
    history_image_keys=("observation.images.front",),
    history_video_sec=60.0,
    history_video_fps=2.0,
    variable_history=True,
    image_aug=True,
    subtask_index_key="subtask_online_index",
    skip_video_demo_frames=True,
)
