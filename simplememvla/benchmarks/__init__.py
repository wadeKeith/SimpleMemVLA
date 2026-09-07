
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:

    name: str
    num_actions_chunk: int
    action_dim: int
    state_dim: int
    robot_tag: str
    native_video_fps: float
    camera_labels: dict[str, str]

    repo_id: str
    root: str
    image_keys: tuple[str, ...]
    history_image_keys: tuple[str, ...]
    history_video_sec: float
    history_video_fps: float
    variable_history: bool
    image_aug: bool

    verbatim_text: bool = False

    subtask_index_key: str = "subtask_index"
    skip_video_demo_frames: bool = False


_REGISTRY: dict[str, str] = {
    "rmbench": "simplememvla.benchmarks.rmbench",
    "robomme": "simplememvla.benchmarks.robomme",
    "mikasa": "simplememvla.benchmarks.mikasa",
    "robomemarena": "simplememvla.benchmarks.robomemarena",
    "libero": "simplememvla.benchmarks.libero",
}

BENCHMARK_NAMES = tuple(_REGISTRY)


def get_benchmark(name: str) -> BenchmarkSpec:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown benchmark {name!r}. Supported: {', '.join(BENCHMARK_NAMES)}."
        )
    import importlib

    module = importlib.import_module(_REGISTRY[key])
    return module.SPEC
