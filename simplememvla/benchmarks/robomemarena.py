
from simplememvla.benchmarks import BenchmarkSpec

NUM_ACTIONS_CHUNK = 16
ACTION_DIM = 7

STATE_DIM = 8

CONTROL_FREQ_HZ = 20
NATIVE_VIDEO_FPS = float(CONTROL_FREQ_HZ)

ROBOT_TAG = "Franka Panda single-arm robot"

RENDER_HEIGHT = 480
RENDER_WIDTH = 640
IMAGE_SIZE = 256

CAMERA_LABELS = {
    "front": "Front camera",
    "agentview": "Front camera",
    "agentview_image": "Front camera",
    "wrist": "Wrist camera",
    "wrist_image": "Wrist camera",
    "eye_in_hand": "Wrist camera",
    "robot0_eye_in_hand_image": "Wrist camera",
}

RAW_FRONT = "agentview_image"
RAW_WRIST = "robot0_eye_in_hand_image"

CAM_FRONT_KEY = "observation.images.front"
CAM_WRIST_KEY = "observation.images.wrist"

MAX_EPISODE_FRAMES = 1918

EVAL_MAX_STEPS = 2500

SPEC = BenchmarkSpec(
    name="robomemarena",
    num_actions_chunk=NUM_ACTIONS_CHUNK,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    robot_tag=ROBOT_TAG,
    native_video_fps=NATIVE_VIDEO_FPS,
    camera_labels=CAMERA_LABELS,
    repo_id="yinchenghust/robomemarena_lerobot",
    root="./data/datasets/yinchenghust/robomemarena_lerobot",
    image_keys=(CAM_FRONT_KEY, CAM_WRIST_KEY),
    history_image_keys=(CAM_FRONT_KEY,),
    history_video_sec=126.0,
    history_video_fps=1.0,
    variable_history=True,
    image_aug=True,
    verbatim_text=True,
)
