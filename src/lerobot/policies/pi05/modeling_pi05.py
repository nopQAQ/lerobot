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

import builtins
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import (
    clone_past_key_values,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    prepare_attention_masks_4d,
    resize_with_pad_torch,
)
from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .align import (
    AlignProjector,
    build_align_layer_pairs,
    compute_projected_align_loss,
    pool_vjepa_tubelet_tokens,
    reduce_align_loss_results,
    scene_prealign_consistency_stats,
    scene_query_embedding_consistency_stats,
)
from .configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from .teachers import load_alignment_teacher, teacher_image_batch

_ALIGNMENT_TEACHER_KEY_MARKERS = ("vj_encoder.", "vj_processor")


def _is_alignment_teacher_key(key: str) -> bool:
    return key.startswith(("vj_encoder", "_alignment_teacher.", "model._alignment_teacher.")) or any(
        marker in key for marker in _ALIGNMENT_TEACHER_KEY_MARKERS
    )


def _loss_metrics(action_loss: Tensor, align_loss: Tensor | None, align_scale: float) -> dict[str, float]:
    metrics = {
        "action_loss/raw": float(action_loss.detach().float().item()),
        "action_loss/weighted": float(action_loss.detach().float().item()),
    }
    if align_loss is not None:
        metrics["align_loss/raw"] = float(align_loss.detach().float().item())
        metrics["align_loss/weighted"] = float((align_scale * align_loss).detach().float().item())
    return metrics


def _take_current_frame(img: Tensor) -> tuple[Tensor, Tensor | None]:
    """Drop an optional time axis for the VLM; keep a [B, T, C, H, W] clip when present."""
    if img.ndim == 4:
        return img, None
    if img.ndim != 5:
        raise ValueError(f"Expected image tensor with 4 or 5 dims, got shape {tuple(img.shape)}")
    if img.shape[-1] == 3:
        clip = img.permute(0, 1, 4, 2, 3).contiguous()
        return img[:, 0], clip
    if img.shape[2] == 3:
        return img[:, 0], img
    raise ValueError(f"Could not interpret 5D image layout with shape {tuple(img.shape)}")


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def _reorder_prefix_scene_behind_action(states: list[Tensor], scene_len: int) -> Tensor:
    """Reorder ``[prefix+scene, action]`` Q/K/V to ``[prefix | action | scene]``.

    Scene tokens still live on the PaliGemma stream (same width / layer), but the
    joint attention sequence places them after the action expert tokens.
    """
    prefix_states, suffix_states = states
    prefix_len = prefix_states.shape[2]
    if scene_len <= 0 or scene_len > prefix_len:
        raise RuntimeError(f"Invalid scene_len={scene_len} for prefix sequence length {prefix_len}")
    base_len = prefix_len - scene_len
    return torch.cat(
        [prefix_states[:, :, :base_len], suffix_states, prefix_states[:, :, base_len:]],
        dim=2,
    )


def _restitch_attention_output_to_module_order(
    att_output: Tensor,
    prefix_len: int,
    suffix_len: int,
    scene_len: int,
) -> Tensor:
    """Map ``[prefix | action | scene]`` attention output back to ``[prefix+scene | action]``."""
    base_len = prefix_len - scene_len
    base_att = att_output[:, :base_len]
    action_att = att_output[:, base_len : base_len + suffix_len]
    scene_att = att_output[:, base_len + suffix_len :]
    return torch.cat([base_att, scene_att, action_att], dim=1)


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb, scene_len=0
):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention. When scene tokens are present they sit at
    # the end of the PaliGemma stream, but joint attention uses
    # ``[image | language | action | scene]``.
    if scene_len:
        query_states = _reorder_prefix_scene_behind_action(query_states, scene_len)
        key_states = _reorder_prefix_scene_behind_action(key_states, scene_len)
        value_states = _reorder_prefix_scene_behind_action(value_states, scene_len)
    else:
        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    if scene_len:
        att_output = _restitch_attention_output_to_module_order(
            att_output,
            prefix_len=inputs_embeds[0].shape[1],
            suffix_len=inputs_embeds[1].shape[1],
            scene_len=scene_len,
        )
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        collect_prefix_scene_len: int | None = None,
        scene_len: int = 0,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        self._prefix_scene_hiddens = None
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            prefix_scene_hiddens = [] if collect_prefix_scene_len else None
            for layers in zip(paligemma_layers, gemma_expert_layers, strict=True):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                        scene_len=scene_len,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                        scene_len=scene_len,
                    )
                if prefix_scene_hiddens is not None:
                    prefix_scene_hiddens.append(inputs_embeds[0][:, -collect_prefix_scene_len:])
            self._prefix_scene_hiddens = prefix_scene_hiddens

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.scene_query = None
        self.vjepa_align_projector = None
        self.vjepa_scene_len = 0
        self.vjepa_temporal_tokens = 0
        self.vjepa_teacher_dim = 0
        self.vjepa_layer_pairs: list[tuple[int, int]] = []
        if config.vjepa_enabled:
            # tubelet_size=2 is the V-JEPA2-ViT-L default; refined after encoder load.
            tubelet_size = 2
            self.vjepa_temporal_tokens = config.vjepa_num_frames // tubelet_size
            self.vjepa_scene_len = self.vjepa_temporal_tokens * (config.vjepa_spatial_pool_size**2)
            self._init_scene_query_from_language_embeddings()

        self.alignment_scene_len = 0
        self.alignment_layer_pairs = []
        backend = config.alignment_backend
        if backend in {"vggtomega", "locateanything"}:
            query = nn.Embedding(1, paligemma_config.width)
            language_weight = (
                self.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings().weight
            )
            with torch.no_grad():
                query.weight.copy_(language_weight.detach().float().mean(dim=0, keepdim=True))
            setattr(self, f"{backend}_scene_query", query)
            teacher_dim = 2048 if backend == "vggtomega" else config.locateanything_teacher_dim
            teacher_layers = 24 if backend == "vggtomega" else config.locateanything_teacher_layers
            setattr(
                self,
                f"{backend}_align_projector",
                AlignProjector(
                    llm_dim=paligemma_config.width,
                    vggt_dim=teacher_dim,
                    align_loss_type=getattr(config, f"{backend}_align_loss_type"),
                    use_vlm_norm=True,
                    use_target_norm=getattr(config, f"{backend}_use_target_norm"),
                ),
            )
            self.alignment_layer_pairs = build_align_layer_pairs(
                student_num_layers=paligemma_config.depth,
                teacher_num_layers=teacher_layers,
                layer_mapping=getattr(config, f"{backend}_layer_mapping"),
            )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(
            bsize,
            device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def _init_scene_query_from_language_embeddings(self) -> None:
        """Initialize per-tubelet scene queries like VLA-JEPA latent tokens.

        VLA-JEPA registers one special token per timestep (``<|action_i|>``) and
        repeats it spatially; new rows are filled with the mean of the existing
        language embedding table (``init_strategy=avg``).
        """
        paligemma_width = get_gemma_config(self.config.paligemma_variant).width
        if self.vjepa_temporal_tokens <= 0:
            raise RuntimeError("vjepa_temporal_tokens must be set before initializing scene_query")
        self.scene_query = nn.Embedding(self.vjepa_temporal_tokens, paligemma_width)
        lang_weight = self.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings().weight
        with torch.no_grad():
            ref = lang_weight.detach().mean(dim=0).to(dtype=self.scene_query.weight.dtype)
            self.scene_query.weight.copy_(ref.unsqueeze(0).expand_as(self.scene_query.weight))

    @staticmethod
    def _scene_query_token_ids(temporal_tokens: int, spatial_tokens: int, device: torch.device) -> Tensor:
        """Map a flattened scene sequence onto per-tubelet embedding ids.

        Layout matches pooled V-JEPA tokens: ``[t0 x H x W | t1 x H x W | ...]``.
        """
        return torch.arange(temporal_tokens, device=device).repeat_interleave(spatial_tokens)

    @property
    def active_scene_query(self):
        backend = getattr(self.config, "alignment_backend", None)
        if backend in {"vggtomega", "locateanything"}:
            return getattr(self, f"{backend}_scene_query")
        return self.scene_query

    @property
    def active_scene_len(self) -> int:
        if getattr(self.config, "alignment_backend", None) in {"vggtomega", "locateanything"}:
            return self.alignment_scene_len
        return self.vjepa_scene_len

    def embed_scene(
        self, bsize: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed auxiliary V-JEPA scene queries that sit after the action suffix."""
        if getattr(self.config, "alignment_backend", None) in {"vggtomega", "locateanything"}:
            length = self.active_scene_len
            if length <= 0:
                raise RuntimeError("Teacher tokens must be extracted before embedding scene queries")
            embeddings = self.active_scene_query(torch.zeros(length, dtype=torch.long, device=device))
            embeddings = embeddings.to(dtype).unsqueeze(0).expand(bsize, -1, -1)
            padding = torch.ones(bsize, length, dtype=torch.bool, device=device)
            attention = torch.zeros_like(padding)
            attention[:, 0] = True
            return embeddings, padding, attention
        if self.scene_query is None or self.vjepa_scene_len <= 0:
            raise RuntimeError("embed_scene requires an initialized scene_query")
        spatial_tokens = self.config.vjepa_spatial_pool_size**2
        expected_len = self.vjepa_temporal_tokens * spatial_tokens
        if expected_len != self.vjepa_scene_len:
            raise RuntimeError(
                f"scene_len {self.vjepa_scene_len} != temporal {self.vjepa_temporal_tokens} "
                f"* spatial {spatial_tokens}"
            )
        if self.scene_query.num_embeddings != self.vjepa_temporal_tokens:
            raise RuntimeError(
                f"scene_query has {self.scene_query.num_embeddings} rows, "
                f"expected {self.vjepa_temporal_tokens} temporal tokens"
            )
        token_ids = self._scene_query_token_ids(self.vjepa_temporal_tokens, spatial_tokens, device=device)
        scene_emb = self.scene_query(token_ids).to(dtype=dtype)
        scene_emb = scene_emb.unsqueeze(0).expand(bsize, -1, -1).contiguous()
        pad_masks = torch.ones(bsize, self.vjepa_scene_len, dtype=torch.bool, device=device)
        # New attention block after actions so make_att_2d_masks already prevents
        # action → scene. Scene is further restricted to image+language only
        # (no scene-to-scene) in _restrict_scene_query_attention.
        att_masks = [1] + ([0] * (self.vjepa_scene_len - 1))
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, self.vjepa_scene_len)
        return scene_emb, pad_masks, att_masks

    def _restrict_scene_query_attention(
        self,
        att_2d_masks: Tensor,
        pad_masks: Tensor,
        prefix_len: int,
        scene_len: int,
        action_len: int | None = None,
        scene_key_mask: Tensor | None = None,
    ) -> Tensor:
        """Scene queries attend only to image+language.

        Sequence layout is ``[image | language | suffix | scene]``. Scene is the
        last ``scene_len`` tokens. Image/language/action cannot attend to scene,
        and scene cannot attend to action or to other scene tokens, so the scene
        branch stays a set of independent queries over the VLM prefix.
        """
        if scene_len <= 0:
            return att_2d_masks
        seq_len = att_2d_masks.shape[-1]
        if action_len is None:
            action_len = seq_len - prefix_len - scene_len
        if action_len < 0:
            raise RuntimeError(
                f"Invalid action_len={action_len} for prefix_len={prefix_len}, "
                f"scene_len={scene_len}, seq_len={seq_len}"
            )
        scene_start = prefix_len + action_len
        scene_end = scene_start + scene_len
        if scene_start < prefix_len or scene_end > seq_len:
            raise RuntimeError(
                f"Invalid scene attention span [{scene_start}, {scene_end}) for "
                f"prefix_len={prefix_len}, action_len={action_len}, "
                f"scene_len={scene_len}, seq_len={seq_len}"
            )

        att_2d_masks = att_2d_masks.clone()
        att_2d_masks[:, :scene_start, scene_start:scene_end] = False
        att_2d_masks[:, scene_start:scene_end, :] = False
        att_2d_masks[:, scene_start:scene_end, :prefix_len] = (
            True if scene_key_mask is None else scene_key_mask[:, None, :]
        )
        pad_queries = pad_masks[:, :, None]
        pad_keys = pad_masks[:, None, :]
        return att_2d_masks & pad_queries & pad_keys

    @staticmethod
    def _build_joint_position_ids(
        prefix_pad_masks: Tensor,
        suffix_pad_masks: Tensor,
        scene_pad_masks: Tensor | None = None,
    ) -> Tensor:
        """Build RoPE positions with scene tokens after actions.

        Action positions continue from image+language only, matching inference
        (which never inserts scene tokens). Scene positions start after the
        action chunk so they cannot shift action RoPE.
        """
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        suffix_offsets = torch.sum(prefix_pad_masks, dim=-1, keepdim=True)
        suffix_position_ids = suffix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        if scene_pad_masks is None:
            return torch.cat([prefix_position_ids, suffix_position_ids], dim=1)
        scene_offsets = suffix_offsets + torch.sum(suffix_pad_masks, dim=-1, keepdim=True)
        scene_position_ids = scene_offsets + torch.cumsum(scene_pad_masks, dim=1) - 1
        return torch.cat([prefix_position_ids, suffix_position_ids, scene_position_ids], dim=1)

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        adarms_cond = time_emb

        bsize, action_time_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))
        att_masks = torch.tensor(att_masks, dtype=action_emb.dtype, device=action_emb.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return action_emb, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        include_scene: bool = False,
        return_scene_hiddens: bool = False,
        instruction_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor] | None]:
        """Do a full training forward pass and compute the loss."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        scene_len = 0
        scene_embs = None
        scene_pad_masks = None
        scene_att_masks = None
        if include_scene and self.active_scene_query is not None and self.active_scene_len > 0:
            scene_embs, scene_pad_masks, scene_att_masks = self.embed_scene(
                prefix_embs.shape[0], prefix_embs.device, prefix_embs.dtype
            )
            scene_len = self.active_scene_len

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            if scene_embs is not None:
                scene_embs = scene_embs.to(dtype=torch.bfloat16)

        if scene_len:
            vlm_embs = torch.cat([prefix_embs, scene_embs], dim=1)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks, scene_pad_masks], dim=1)
            att_masks = torch.cat(
                [
                    prefix_att_masks.to(dtype=suffix_att_masks.dtype),
                    suffix_att_masks,
                    scene_att_masks.to(dtype=suffix_att_masks.dtype),
                ],
                dim=1,
            )
        else:
            vlm_embs = prefix_embs
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        scene_key_mask = None
        backend = self.config.alignment_backend
        if scene_len and backend in {"vggtomega", "locateanything"}:
            image_len = prefix_pad_masks.shape[1] - tokens.shape[1]
            scene_key_mask = prefix_pad_masks.clone()
            scene_key_mask[:, image_len:] = False
            if getattr(self.config, f"{backend}_scene_attention_mask_mode") == "image_and_instruction":
                if instruction_mask is None or instruction_mask.shape != masks.shape:
                    raise ValueError(
                        "Teacher alignment requires the PI05 instruction mask from its preprocessor"
                    )
                scene_key_mask[:, image_len:] = instruction_mask.bool() & masks.bool()
        if scene_len:
            att_2d_masks = self._restrict_scene_query_attention(
                att_2d_masks,
                pad_masks,
                prefix_len=prefix_pad_masks.shape[1],
                scene_len=scene_len,
                action_len=suffix_pad_masks.shape[1],
                scene_key_mask=scene_key_mask,
            )
        position_ids = self._build_joint_position_ids(
            prefix_pad_masks,
            suffix_pad_masks,
            scene_pad_masks=scene_pad_masks,
        )

        att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)
        collect_scene_len = scene_len if (include_scene and return_scene_hiddens and scene_len > 0) else None
        layerwise = bool(getattr(self.config, f"{backend}_use_layerwise_alignment", False))
        scene_hiddens = None

        if collect_scene_len:
            # Keep per-layer scene slices on the graph; do not wrap this joint
            # forward in an extra checkpoint (nested tensor lists are not supported).
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[vlm_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                collect_prefix_scene_len=collect_scene_len if layerwise else None,
                scene_len=scene_len,
            )
            if layerwise:
                scene_hiddens = self.paligemma_with_expert._prefix_scene_hiddens
            else:
                scene_hiddens = [prefix_out[:, -collect_scene_len:]]
        else:

            def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
                (_, suffix_out), _ = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                    scene_len=scene_len,
                )
                return suffix_out

            suffix_out = self._apply_checkpoint(
                forward_func, vlm_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        if not return_scene_hiddens:
            return losses
        return losses, scene_hiddens

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        # V-JEPA alignment is training-only. Inference follows the original PI0.5
        # path and never inserts scene tokens.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.vj_encoder = None
        self.vj_processor = None
        self._vjepa_model_path = None
        self._vjepa_clip_btchw = None
        self._alignment_teacher = None

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)
        self._init_vjepa_alignment()

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model.
            # V-JEPA adds per-tubelet scene_query / align_projector (vocab-mean init
            # from PaliGemma embeddings, not present in pi05_base) and a frozen
            # encoder loaded from its own checkpoint, so strict load cannot be used.
            strict_load = False if model.config.alignment_backend else strict
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict_load)
            backend = model.config.alignment_backend
            if (
                backend in {"vggtomega", "locateanything"}
                and f"model.{backend}_scene_query.weight" in missing_keys
            ):
                # Initialize new queries from the loaded base vocabulary, not
                # from the random embeddings created before checkpoint loading.
                language_weight = model.model.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings().weight
                with torch.no_grad():
                    model.model.active_scene_query.weight.copy_(
                        language_weight.detach().float().mean(dim=0, keepdim=True)
                    )
            missing_keys = [k for k in missing_keys if not _is_alignment_teacher_key(k)]
            unexpected_keys = [k for k in unexpected_keys if not _is_alignment_teacher_key(k)]

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in.") or key.startswith("model.action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        backend = getattr(self.config, "alignment_backend", "vjepa" if self.config.vjepa_enabled else None)
        if backend is None:
            return self.parameters()

        projector = getattr(self.model, f"{backend}_align_projector")
        projector_ids = {id(param) for param in projector.parameters()} if projector is not None else set()
        projector_params = []
        other_params = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in projector_ids:
                projector_params.append(param)
            else:
                other_params.append(param)

        if not projector_params:
            return other_params

        return [
            {"params": other_params, "lr": self.config.optimizer_lr},
            {
                "params": projector_params,
                "lr": getattr(self.config, f"{backend}_align_projector_lr"),
                "name": f"{backend}_align_projector",
            },
        ]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vj_encoder is not None:
            self.vj_encoder.eval()
        if self._alignment_teacher is not None:
            self._alignment_teacher.eval()
        return self

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        if state_dict is None:
            state_dict = self.state_dict()
        state_dict = {key: value for key, value in state_dict.items() if not _is_alignment_teacher_key(key)}
        super()._save_pretrained(save_directory, state_dict=state_dict)

    def _init_vjepa_alignment(self) -> None:
        """Initialize trainable alignment modules without loading the frozen teacher."""
        if not self.config.vjepa_enabled:
            return

        from transformers import AutoConfig

        model_path = Path(self.config.vjepa_model_path).expanduser()
        model_path_str = str(model_path)
        self._vjepa_model_path = model_path_str
        vj_config = AutoConfig.from_pretrained(model_path_str, trust_remote_code=True)
        tubelet_size = int(getattr(vj_config, "tubelet_size", 2))
        if self.config.vjepa_num_frames % tubelet_size != 0:
            raise ValueError(
                f"vjepa_num_frames ({self.config.vjepa_num_frames}) must be divisible by "
                f"tubelet_size ({tubelet_size})"
            )
        self.model.vjepa_temporal_tokens = self.config.vjepa_num_frames // tubelet_size
        self.model.vjepa_scene_len = self.model.vjepa_temporal_tokens * (
            self.config.vjepa_spatial_pool_size**2
        )
        if (
            self.model.scene_query is None
            or self.model.scene_query.num_embeddings != self.model.vjepa_temporal_tokens
        ):
            self.model._init_scene_query_from_language_embeddings()
        self.model.vjepa_teacher_dim = int(vj_config.hidden_size)
        paligemma_width = get_gemma_config(self.config.paligemma_variant).width
        self.model.vjepa_align_projector = AlignProjector(
            llm_dim=paligemma_width,
            vggt_dim=self.model.vjepa_teacher_dim,
            align_loss_type=self.config.vjepa_align_loss_type,
            use_vlm_norm=True,
        )

        student_layers = get_gemma_config(self.config.paligemma_variant).depth
        teacher_layers = int(getattr(vj_config, "num_hidden_layers", 24))
        self.model.vjepa_layer_pairs = build_align_layer_pairs(
            student_num_layers=student_layers,
            teacher_num_layers=teacher_layers,
            layer_mapping=self.config.vjepa_layer_mapping,
        )
        logging.info(
            "V-JEPA prefix alignment enabled "
            f"(scene_len={self.model.vjepa_scene_len}, "
            f"temporal_tokens={self.model.vjepa_temporal_tokens}, "
            f"teacher_dim={self.model.vjepa_teacher_dim}, "
            f"layerwise={self.config.vjepa_use_layerwise_alignment}, "
            f"pairs={len(self.model.vjepa_layer_pairs)}, "
            f"align_scale={self.config.vjepa_align_loss_scale}, "
            f"projector_lr={self.config.vjepa_align_projector_lr})"
        )

    def _ensure_vjepa_teacher(self) -> None:
        """Lazily load V-JEPA only when a training/evaluation loss needs targets."""
        if not self.config.vjepa_enabled or self.vj_encoder is not None:
            return
        if self._vjepa_model_path is None:
            raise RuntimeError("V-JEPA model path is not initialized")

        from transformers import AutoModel, AutoVideoProcessor

        logging.info(f"Loading frozen V-JEPA encoder from {self._vjepa_model_path}")
        try:
            processor = AutoVideoProcessor.from_pretrained(
                self._vjepa_model_path,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "V-JEPA alignment requires its AutoVideoProcessor; refusing to use an "
                "unnormalized resize-only fallback"
            ) from exc

        encoder = AutoModel.from_pretrained(self._vjepa_model_path, trust_remote_code=True)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.to(next(self.model.parameters()).device)

        self.vj_processor = processor
        self.vj_encoder = encoder

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []
        self._vjepa_clip_btchw = None

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key_idx, key in enumerate(present_img_keys):
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            img, clip_btchw = _take_current_frame(img)
            if self.config.vjepa_enabled and key_idx == 0:
                self._vjepa_clip_btchw = clip_btchw

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(
        self, batch: dict[str, Tensor], reduction: str = "mean", *, return_loss_tensors: bool = False
    ) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        backend = self.config.alignment_backend
        include_scene = backend is not None
        teacher_layers = None
        if backend in {"vggtomega", "locateanything"}:
            if self._alignment_teacher is None:
                self._alignment_teacher = load_alignment_teacher(self.config, actions.device)
            # Use the first configured real camera, never silently substitute a wrist view.
            main_key = next(iter(self.config.image_features))
            if main_key not in batch:
                raise ValueError(f"Alignment main camera is missing: {main_key}")
            pixels = teacher_image_batch(batch[main_key]).to(actions.device)
            with torch.no_grad():
                teacher_layers = self._alignment_teacher(pixels, batch.get("raw_task"))
            self.model.alignment_scene_len = teacher_layers[0].shape[1]
        model_out = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            include_scene=include_scene,
            return_scene_hiddens=include_scene,
            instruction_mask=batch.get("observation.language.instruction_mask"),
        )
        if include_scene:
            losses, scene_hiddens = model_out
        else:
            losses = model_out
            scene_hiddens = None

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        align_loss = None
        if include_scene and scene_hiddens is not None:
            prealign = scene_prealign_consistency_stats(
                scene_hiddens[-1], self.model.vjepa_temporal_tokens if backend == "vjepa" else 1
            )
            if self.model.active_scene_query is not None:
                prealign.update(scene_query_embedding_consistency_stats(self.model.active_scene_query.weight))
            loss_dict.update({key: value.detach().float().item() for key, value in prealign.items()})
            if backend == "vjepa":
                video = self._prepare_vjepa_video(images)
                teacher_layers = self._vjepa_encoder_layer_tokens(video)
            align_loss, align_components = self._compute_alignment_loss(scene_hiddens, teacher_layers)
            for key, value in align_components.items():
                metric_key = (
                    key
                    if key.startswith("align_loss/layer_")
                    else f"align_loss/{key.removeprefix('align_').removesuffix('_loss')}"
                )
                loss_dict[metric_key] = float(value.detach().float().item())

        align_scale = self.config.alignment_loss_scale if align_loss is not None else 0.0
        action_raw = losses.mean()
        align_raw = align_loss if align_loss is not None else losses.new_zeros(())
        loss_dict.update(_loss_metrics(action_raw, align_loss, align_scale))

        if return_loss_tensors:
            loss_dict["_loss_tensors"] = {
                "action_loss": action_raw,
                "align_loss": align_raw,
            }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            if align_loss is not None:
                per_sample_loss = per_sample_loss + align_scale * align_loss
            loss_dict["loss"] = float(per_sample_loss.mean().item())
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            if align_loss is not None:
                loss = loss + align_scale * align_loss
            loss_dict["loss"] = float(loss.item())
            return loss, loss_dict

    def _prepare_vjepa_video(self, vlm_images: list[Tensor]) -> Tensor:
        """Build a [B, T, C, H, W] clip in [0, 1] for the frozen V-JEPA encoder.

        Prefer a real primary-camera clip from ``delta_timestamps``. Otherwise repeat
        the current VLM frame (bring-up path).
        """
        num_frames = self.config.vjepa_num_frames
        if self._vjepa_clip_btchw is not None:
            video = self._vjepa_clip_btchw
            if video.dtype != torch.float32:
                video = video.to(dtype=torch.float32)
            if video.max() > 1.5:
                video = video / 255.0
        else:
            current = vlm_images[0]
            current = ((current + 1.0) * 0.5).clamp(0.0, 1.0)
            video = current.unsqueeze(1).expand(-1, num_frames, -1, -1, -1).contiguous()

        time_len = video.shape[1]
        if time_len < num_frames:
            pad = video[:, -1:].expand(-1, num_frames - time_len, -1, -1, -1)
            video = torch.cat([video, pad], dim=1)
        elif time_len > num_frames:
            video = video[:, :num_frames]
        return video

    def _process_vjepa_clip(self, clip: Tensor) -> Tensor:
        """Apply V-JEPA transforms to a float [0, 1] clip without rescaling twice."""
        if self.vj_processor is None:
            raise RuntimeError("V-JEPA video processor is not initialized")
        return self.vj_processor(
            videos=clip,
            return_tensors="pt",
            do_rescale=False,
        )["pixel_values_videos"]

    @torch.no_grad()
    def _vjepa_encoder_layer_tokens(self, video_btchw: Tensor) -> list[Tensor]:
        self._ensure_vjepa_teacher()
        if self.vj_encoder is None:
            raise RuntimeError("V-JEPA encoder is not initialized")

        encoder_device = next(self.vj_encoder.parameters()).device
        video_btchw = video_btchw.to(encoder_device)
        clips = []
        for sample_idx in range(video_btchw.shape[0]):
            clip = video_btchw[sample_idx]
            processed = self._process_vjepa_clip(clip)
            clips.append(processed.to(device=encoder_device))
        pixel_values = torch.cat(clips, dim=0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoder_outputs = self.vj_encoder(
                pixel_values_videos=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        layer_tokens = []
        for layer_hidden in encoder_outputs.hidden_states[1:]:
            pooled = pool_vjepa_tubelet_tokens(
                layer_hidden,
                self.model.vjepa_temporal_tokens,
                self.config.vjepa_spatial_pool_size,
            )
            layer_tokens.append(pooled.contiguous())
        if not layer_tokens:
            raise RuntimeError("V-JEPA encoder returned no hidden states for alignment")
        return layer_tokens

    def _compute_alignment_loss(
        self,
        scene_hiddens: list[Tensor],
        teacher_layers: list[Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        backend = self.config.alignment_backend
        projector = getattr(self.model, f"{backend}_align_projector")
        if projector is None:
            raise RuntimeError(f"{backend} align projector is not initialized")
        expected_len = self.model.active_scene_len
        loss_type = getattr(self.config, f"{backend}_align_loss_type")
        pairs = self.model.vjepa_layer_pairs if backend == "vjepa" else self.model.alignment_layer_pairs
        if any(target.ndim != 3 or target.shape[1] != expected_len for target in teacher_layers):
            raise ValueError(f"{backend} teacher token layout does not match scene queries")

        if getattr(self.config, f"{backend}_use_layerwise_alignment"):
            results = []
            for student_idx, teacher_idx in pairs:
                scene = scene_hiddens[student_idx - 1]
                if scene.shape[1] != expected_len:
                    raise RuntimeError(
                        f"scene hidden len {scene.shape[1]} != expected {expected_len} "
                        f"(student layer {student_idx})"
                    )
                target = projector.normalize_target(teacher_layers[teacher_idx])
                projected = projector.align_dimension(scene)
                results.append(compute_projected_align_loss(projected, target, loss_type))
            return reduce_align_loss_results(results)

        scene = scene_hiddens[-1]
        if scene.shape[1] != expected_len:
            raise RuntimeError(f"scene hidden len {scene.shape[1]} != expected {expected_len}")
        target = projector.normalize_target(teacher_layers[-1])
        projected = projector.align_dimension(scene)
        return compute_projected_align_loss(projected, target, loss_type)

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
