from transformers import PretrainedConfig


class SimpleMemVLAConfig(PretrainedConfig):

    model_type = "simplememvla"

    def __init__(
        self,
        backbone_model_name_or_path: str = "Qwen/Qwen3.5-4B",
        hidden_size: int | None = None,
        action_dim: int | None = None,
        action_horizon: int | None = None,
        use_proprio: bool = True,
        state_dim: int | None = None,
        state_dropout_prob: float = 0.0,
        dit_hidden_size: int = 2048,
        dit_depth: int = 16,
        dit_num_heads: int = 16,
        dit_mlp_ratio: float = 3.5,
        dit_dropout: float = 0.0,
        dit_rope_theta: float | None = None,
        dit_mrope_section: list[int] | tuple[int, int, int] | None = None,
        dit_partial_rotary_factor: float | None = None,
        timestep_beta_alpha: float = 1.5,
        timestep_beta_beta: float = 1.0,
        action_loss_weight: float = 1.0,
        vl_loss_weight: float = 1.0,
        freeze_backbone: bool = False,
        robot_tag: str | None = None,
        control_frequency_hz: int | None = None,
        history_video_sec: float = 60.0,
        history_video_fps: float = 2.0,
        native_video_fps: float | None = None,
        variable_history: bool = False,
        image_keys: list[str] | None = None,
        history_image_keys: list[str] | None = None,
        subtask_key: str = "subtask",
        subtask_index_key: str = "subtask_index",
        backbone_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_proprio = use_proprio
        self.state_dim = state_dim
        self.state_dropout_prob = state_dropout_prob
        self.dit_hidden_size = dit_hidden_size
        self.dit_depth = dit_depth
        self.dit_num_heads = dit_num_heads
        self.dit_mlp_ratio = dit_mlp_ratio
        self.dit_dropout = dit_dropout
        self.dit_rope_theta = dit_rope_theta
        self.dit_mrope_section = (
            list(dit_mrope_section) if dit_mrope_section is not None else None
        )
        self.dit_partial_rotary_factor = dit_partial_rotary_factor
        self.timestep_beta_alpha = timestep_beta_alpha
        self.timestep_beta_beta = timestep_beta_beta
        self.action_loss_weight = action_loss_weight
        self.vl_loss_weight = vl_loss_weight
        self.freeze_backbone = freeze_backbone
        self.robot_tag = robot_tag
        self.control_frequency_hz = control_frequency_hz
        self.history_video_sec = history_video_sec
        self.history_video_fps = history_video_fps
        self.native_video_fps = native_video_fps
        self.variable_history = variable_history
        self.image_keys = list(image_keys) if image_keys is not None else None
        self.history_image_keys = (
            list(history_image_keys) if history_image_keys is not None else None
        )
        self.subtask_key = subtask_key
        self.subtask_index_key = subtask_index_key
        self.backbone_config = backbone_config
