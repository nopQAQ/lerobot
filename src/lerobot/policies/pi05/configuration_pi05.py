#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    # Frozen V-JEPA2 alignment on PaliGemma scene queries (off by default).
    # One learnable embedding per temporal tubelet, repeated over the spatial
    # grid (VLA-JEPA <|action_i|> style). Initialized from the mean of the
    # PaliGemma language embedding table. Tokens sit after the action suffix.
    vjepa_enabled: bool = False
    vjepa_model_path: str = "/mnt/nas/code/starVLA/playground/Pretrained_models/vjepa2-vitl-fpc64-256"
    vjepa_align_loss_type: str = "cosine"
    vjepa_align_loss_scale: float = 0.4
    # Dedicated LR for the trainable AlignProjector (starVLA QwenOFT uses 1e-4).
    # The rest of PI0.5 still uses optimizer_lr. Set equal to optimizer_lr to disable.
    vjepa_align_projector_lr: float = 1e-4
    vjepa_use_layerwise_alignment: bool = True
    vjepa_layer_mapping: str = "uniform"
    vjepa_include_at_inference: bool = False
    vjepa_num_frames: int = 8
    vjepa_spatial_pool_size: int = 8
    vjepa_video_resolution: int = 256
    # False: repeat the current primary-camera frame (bring-up). True: stack
    # observation_delta_indices frames and use the primary camera clip.
    vjepa_use_video_clip: bool = False

    # Single-teacher alternatives, matching starVLA-single's main-view experiments.
    vggtomega_enabled: bool = False
    vggtomega_source_path: str = "/mnt/nas/code/starVLA/third_party/vggtomega"
    vggtomega_checkpoint_path: str = "/mnt/nas/code/starVLA/third_party/vggtomega/vggt_omega_1b_512.pt"
    vggtomega_num_frames: int = 8
    vggtomega_image_size: int = 224
    vggtomega_align_loss_type: str = "cosine"
    vggtomega_align_loss_scale: float = 0.05
    vggtomega_align_projector_lr: float = 1e-4
    vggtomega_use_layerwise_alignment: bool = True
    vggtomega_layer_mapping: str = "uniform"
    vggtomega_use_target_norm: bool = False
    vggtomega_scene_attention_mask_mode: str = "image_only"

    locateanything_enabled: bool = False
    locateanything_model_path: str = "/mnt/nas/code/starVLA/playground/Pretrained_models/LocateAnything-3B"
    locateanything_attn_implementation: str = "sdpa"
    locateanything_prompt_template: str = (
        "Locate all the instances that matches the following description: {text}."
    )
    locateanything_align_loss_type: str = "cosine"
    locateanything_align_loss_scale: float = 0.1
    locateanything_align_projector_lr: float = 1e-4
    locateanything_use_layerwise_alignment: bool = True
    locateanything_layer_mapping: str = "uniform"
    locateanything_use_target_norm: bool = False
    locateanything_scene_attention_mask_mode: str = "image_and_instruction"
    # Persist architecture metadata so action inference never needs teacher files.
    locateanything_teacher_dim: int = 2048
    locateanything_teacher_layers: int = 36

    # Low-frequency action/alignment gradient diagnostics (disabled by default).
    grad_conflict_enabled: bool = False
    grad_conflict_freq: int = 100
    grad_conflict_scope: str = "paligemma_language_model"

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        allowed_align = {"l1", "mse", "cosine", "smooth_l1", "mse_cosine", "cosine_mse"}
        enabled = [
            name for name in ("vjepa", "vggtomega", "locateanything") if getattr(self, f"{name}_enabled")
        ]
        if len(enabled) > 1:
            raise ValueError(f"PI05 supports one alignment teacher at a time, got {enabled}")
        for name in ("vggtomega", "locateanything"):
            if getattr(self, f"{name}_align_loss_type") not in allowed_align:
                raise ValueError(f"Invalid {name}_align_loss_type")
            if getattr(self, f"{name}_align_loss_scale") < 0:
                raise ValueError(f"{name}_align_loss_scale must be >= 0")
            if getattr(self, f"{name}_align_projector_lr") <= 0:
                raise ValueError(f"{name}_align_projector_lr must be > 0")
            if getattr(self, f"{name}_scene_attention_mask_mode") not in {
                "image_only",
                "image_and_instruction",
            }:
                raise ValueError(f"Unsupported {name}_scene_attention_mask_mode")
        if self.vggtomega_num_frames < 1 or self.vggtomega_image_size < 16 or self.vggtomega_image_size % 16:
            raise ValueError("VGGT requires >= 1 frame and an image size divisible by 16")
        if self.locateanything_teacher_dim < 1 or self.locateanything_teacher_layers < 1:
            raise ValueError("LocateAnything architecture dimensions must be positive")
        if self.locateanything_enabled and not self.locateanything_use_layerwise_alignment:
            raise ValueError("LocateAnything requires use_layerwise_alignment=true, as in starVLA-single")
        if self.vjepa_align_loss_type not in allowed_align:
            raise ValueError(
                f"Invalid vjepa_align_loss_type: {self.vjepa_align_loss_type}. "
                f"Expected one of {sorted(allowed_align)}"
            )
        if self.vjepa_num_frames < 2:
            raise ValueError(f"vjepa_num_frames must be >= 2, got {self.vjepa_num_frames}")
        if self.vjepa_spatial_pool_size < 1:
            raise ValueError(f"vjepa_spatial_pool_size must be >= 1, got {self.vjepa_spatial_pool_size}")
        if self.vjepa_align_loss_scale < 0:
            raise ValueError(f"vjepa_align_loss_scale must be >= 0, got {self.vjepa_align_loss_scale}")
        if self.grad_conflict_freq < 1:
            raise ValueError(f"grad_conflict_freq must be >= 1, got {self.grad_conflict_freq}")
        if self.grad_conflict_scope != "paligemma_language_model":
            raise ValueError(f"Unsupported grad_conflict_scope: {self.grad_conflict_scope}")
        if self.vjepa_align_projector_lr <= 0:
            raise ValueError(f"vjepa_align_projector_lr must be > 0, got {self.vjepa_align_projector_lr}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def alignment_backend(self) -> str | None:
        return next(
            (name for name in ("vjepa", "vggtomega", "locateanything") if getattr(self, f"{name}_enabled")),
            None,
        )

    @property
    def alignment_loss_scale(self) -> float:
        backend = self.alignment_backend
        return float(getattr(self, f"{backend}_align_loss_scale")) if backend else 0.0

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if self.vggtomega_enabled:
            return list(range(self.vggtomega_num_frames))
        if self.vjepa_enabled and self.vjepa_use_video_clip:
            return list(range(self.vjepa_num_frames))
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
