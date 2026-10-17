"""Training-only frozen teachers for PI05's single-teacher alignment.

VGGT uses the local upstream inference package. LocateAnything uses its HF
checkpoint code and our small feature wrapper; neither imports starVLA.
"""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_pi05 import PI05Config
from .locateanything import LocateAnythingBackbone

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_utils import PreTrainedModel


@contextmanager
def _locateanything_hf_compat(remote_class):
    """Bridge the checkpoint's Transformers 4.x hooks during 5.x construction.

    Changes are limited to this checkpoint's dynamic classes and restored after
    loading. The actual attention/feature computation is unchanged.
    """
    qwen = importlib.import_module(remote_class.__module__.rsplit(".", 1)[0] + ".modeling_qwen2")
    changes = [
        (
            qwen.Qwen2PreTrainedModel,
            "_check_and_adjust_attn_implementation",
            PreTrainedModel._check_and_adjust_attn_implementation,
        ),
        (qwen.Qwen2ForCausalLM, "_tied_weights_keys", {"lm_head.weight": "model.embed_tokens.weight"}),
    ]
    originals = [(cls, name, getattr(cls, name)) for cls, name, _ in changes]
    try:
        for cls, name, value in changes:
            setattr(cls, name, value)
        yield
    finally:
        for cls, name, value in originals:
            setattr(cls, name, value)


def teacher_image_batch(images: Tensor) -> Tensor:
    """Convert BCHW/BHWC or BTCHW/BTHWC to float BTCHW in [0, 1]."""
    if images.ndim == 4:
        images = images.unsqueeze(1)
    if images.ndim != 5:
        raise ValueError(f"Expected a batched image or clip, got {tuple(images.shape)}")
    if images.shape[2] != 3:
        if images.shape[-1] != 3:
            raise ValueError(f"Expected three image channels, got {tuple(images.shape)}")
        images = images.permute(0, 1, 4, 2, 3)
    is_byte = images.dtype == torch.uint8
    images = images.float()
    if is_byte or images.max() > 1.5:
        images = images / 255.0
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Teacher images must contain finite values in [0,1] or [0,255]")
    return images.contiguous()


class VGGTOmegaTeacher(nn.Module):
    def __init__(self, model: nn.Module, num_frames: int, image_size: int):
        super().__init__()
        self.model = model
        self.num_frames = num_frames
        self.image_size = image_size

    @torch.no_grad()
    def forward(self, clips: Tensor, texts=None) -> list[Tensor]:
        if clips.shape[1] < self.num_frames:
            raise ValueError(f"VGGT requires {self.num_frames} real video frames, got {clips.shape[1]}")
        clips = clips[:, : self.num_frames]
        batch, frames, channels, height, width = clips.shape
        if (height, width) != (self.image_size, self.image_size):
            clips = F.interpolate(
                clips.reshape(batch * frames, channels, height, width),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, frames, channels, self.image_size, self.image_size)
        with torch.autocast(clips.device.type, dtype=torch.bfloat16, enabled=clips.device.type == "cuda"):
            layers, patch_start = self.model.aggregator(clips)
        if not layers or any(layer is None for layer in layers):
            raise RuntimeError("VGGT must cache every aggregator layer for alignment")
        return [layer[:, 0, patch_start:].contiguous() for layer in layers]


class LocateAnythingTeacher(nn.Module):
    def __init__(self, backbone: LocateAnythingBackbone):
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, clips: Tensor, texts: list[str] | None = None) -> list[Tensor]:
        if (
            texts is None
            or len(texts) != clips.shape[0]
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise ValueError("LocateAnything requires one original raw_task string per image")
        pixels = (clips[:, 0].permute(0, 2, 3, 1).cpu() * 255).to(torch.uint8).numpy()
        return self.backbone([Image.fromarray(image) for image in pixels], texts)


def _restore_locateanything_rope(model: nn.Module) -> None:
    """Rebuild legacy non-persistent RoPE buffers after HF's meta-device load."""
    for module in model.language_model.modules():
        if module.__class__.__name__ == "Qwen2RotaryEmbedding":
            module.inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float() / module.dim))
            module._set_cos_sin_cache(module.max_position_embeddings, device="cpu", dtype=torch.float32)


def load_alignment_teacher(config: PI05Config, device: torch.device) -> nn.Module:
    backend = config.alignment_backend
    if backend == "vggtomega":
        source = Path(config.vggtomega_source_path).expanduser().resolve()
        if not (source / "vggt_omega" / "__init__.py").is_file():
            raise FileNotFoundError(f"Set vggtomega_source_path to the VGGT-Omega source directory: {source}")
        checkpoint = Path(config.vggtomega_checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT checkpoint does not exist: {checkpoint}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        vggt_class = importlib.import_module("vggt_omega").VGGTOmega
        model = vggt_class()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.aggregator.cached_layer_indices = set(range(model.aggregator.depth))
        teacher = VGGTOmegaTeacher(model, config.vggtomega_num_frames, config.vggtomega_image_size)
    elif backend == "locateanything":
        require_package("transformers", extra="pi")
        require_package("peft", extra="pi05-teachers")
        require_package("lmdb", extra="pi05-teachers")
        require_package("decord", extra="pi05-teachers")
        path = str(Path(config.locateanything_model_path).expanduser())
        teacher_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        text_config = teacher_config.text_config
        if (text_config.hidden_size, text_config.num_hidden_layers) != (
            config.locateanything_teacher_dim,
            config.locateanything_teacher_layers,
        ):
            raise ValueError(
                "LocateAnything checkpoint dimensions differ from the saved PI05 teacher metadata"
            )
        attention = config.locateanything_attn_implementation
        teacher_config._attn_implementation = attention
        text_config._attn_implementation = attention
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        # The checkpoint overrides an HF private method with a pre-5.5 signature.
        # Adapt only this remote model class, without patching Transformers globally.
        remote_class = get_class_from_dynamic_module(teacher_config.auto_map["AutoModel"], path)
        qwen = importlib.import_module(remote_class.__module__.rsplit(".", 1)[0] + ".configuration_qwen2")
        # HF 5 folds rope_theta into rope_parameters. The checkpoint's custom
        # Qwen implementation still reads the original flat configuration.
        with (Path(path) / "config.json").open() as handle:
            original_text_config = json.load(handle)["text_config"]
        original_text_config.setdefault("pad_token_id", None)
        teacher_config.text_config = qwen.Qwen2Config(**original_text_config)
        teacher_config.text_config._attn_implementation = attention

        class CompatibleLocateAnything(remote_class):
            @classmethod
            def is_remote_code(cls):
                return True

            def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
                return PreTrainedModel._check_and_adjust_attn_implementation(
                    self, attn_implementation, **kwargs
                )

            def __init__(self, config, *args, **kwargs):
                super().__init__(config, *args, **kwargs)
                self.post_init()

        with _locateanything_hf_compat(remote_class):
            model, loading_info = CompatibleLocateAnything.from_pretrained(
                path,
                config=teacher_config,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                attn_implementation=attention,
                output_loading_info=True,
            )
        if (
            loading_info["missing_keys"]
            or loading_info["unexpected_keys"]
            or loading_info.get("mismatched_keys")
        ):
            raise RuntimeError(f"LocateAnything teacher checkpoint did not load exactly: {loading_info}")
        # HF's meta loader materializes non-persistent buffers as empty tensors.
        # This old Qwen implementation creates its RoPE caches in __init__ rather
        # than _init_weights, so rebuild them explicitly from the checkpoint config.
        _restore_locateanything_rope(model)
        teacher = LocateAnythingTeacher(
            LocateAnythingBackbone(
                model,
                processor,
                prompt_template=config.locateanything_prompt_template,
            )
        )
    else:
        raise ValueError(f"Unsupported teacher loader: {backend}")
    teacher.requires_grad_(False)
    return teacher.to(device).eval()
