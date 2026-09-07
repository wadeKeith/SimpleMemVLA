
from simplememvla.benchmarks import BenchmarkSpec

NUM_ACTIONS_CHUNK = 16
ACTION_DIM = 7
JOINT_ACTION_DIM = 8
STATE_DIM = 9

CONTROL_FREQ_HZ = 20
NATIVE_VIDEO_FPS = float(CONTROL_FREQ_HZ)

ROBOT_TAG = "Franka Panda single-arm robot"

CAMERA_LABELS = {
    "image": "Front camera",
    "agentview": "Front camera",
    "agentview_image": "Front camera",
    "wrist_image": "Wrist camera",
    "eye_in_hand": "Wrist camera",
    "robot0_eye_in_hand_image": "Wrist camera",
}

SPEC = BenchmarkSpec(
    name="libero",
    num_actions_chunk=NUM_ACTIONS_CHUNK,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    robot_tag=ROBOT_TAG,
    native_video_fps=NATIVE_VIDEO_FPS,
    camera_labels=CAMERA_LABELS,
    repo_id="yinchenghust/libero_lerobot",
    root="./data/datasets/yinchenghust/libero_lerobot",
    image_keys=(
        "observation.images.image",
        "observation.images.wrist_image",
    ),
    history_image_keys=("observation.images.image",),
    history_video_sec=30.0,
    history_video_fps=2.0,
    variable_history=True,
    image_aug=True,
)
