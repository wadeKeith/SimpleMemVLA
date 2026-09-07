
from simplememvla.benchmarks import BenchmarkSpec

NUM_ACTIONS_CHUNK = 30
ACTION_DIM = 14
STATE_DIM = 14

SIM_SCENE_HZ = 250
SAVE_FREQ = 15
NATIVE_VIDEO_FPS = SIM_SCENE_HZ / SAVE_FREQ

ROBOT_TAG = "Aloha-AgileX dual-arm robot"

CAMERA_LABELS = {
    "cam_high": "Head camera",
    "head_camera": "Head camera",
    "image": "Head camera",
    "cam_left_wrist": "Left wrist camera",
    "left_wrist_image": "Left wrist camera",
    "cam_right_wrist": "Right wrist camera",
    "right_wrist_image": "Right wrist camera",
}

SPEC = BenchmarkSpec(
    name="rmbench",
    num_actions_chunk=NUM_ACTIONS_CHUNK,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    robot_tag=ROBOT_TAG,
    native_video_fps=NATIVE_VIDEO_FPS,
    camera_labels=CAMERA_LABELS,
    repo_id="yinchenghust/rmbench_lerobot",
    root="./data/datasets/yinchenghust/rmbench_lerobot",
    image_keys=(
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ),
    history_image_keys=("observation.images.cam_high",),
    history_video_sec=60.0,
    history_video_fps=2.0,
    variable_history=True,
    image_aug=True,
)
