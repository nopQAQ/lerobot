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

"""Frozen-teacher alignment helpers for PI05 prefix scene tokens.

Copied from starVLA QwenOFT (`projectors.AlignProjector` and layer-mapping
utilities) so lerobot does not import the starVLA package.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

_ALIGN_LOSS_TYPES = frozenset(
    {
        "cosine",
        "l1",
        "mse",
        "smooth_l1",
        "smoothl1",
        "huber",
        "mse_cosine",
        "cosine_mse",
    }
)
_LAYER_MAPPING_STRIDE4_ALIASES = frozenset(
    {"stride4", "stride_4", "interval4", "interval_4", "every4", "every_4"}
)


def normalize_align_loss_type(loss_type: str) -> str:
    loss_type = str(loss_type).lower()
    if loss_type in {"smoothl1", "huber"}:
        return "smooth_l1"
    return loss_type


class AlignProjector(nn.Module):
    """Project LLM scene tokens into teacher embedding space for alignment."""

    def __init__(
        self,
        llm_dim: int,
        vggt_dim: int,
        align_loss_type: str = "cosine",
        use_vlm_norm: bool = False,
        use_target_norm: bool = False,
    ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type

        self.fc1 = nn.Linear(self.llm_dim, self.vggt_dim, bias=True)
        self.fc2 = nn.Linear(self.vggt_dim, self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()

        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None
        self.target_norm = nn.LayerNorm(vggt_dim) if use_target_norm else None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def align_dimension(self, llm_embedding: torch.Tensor) -> torch.Tensor:
        if self.vlm_norm is not None:
            llm_embedding = self.vlm_norm(llm_embedding)
        projected_features = self.fc1(llm_embedding)
        projected_features = self.act_fn1(projected_features)
        return self.fc2(projected_features)

    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.target_norm is not None:
            return self.target_norm(target)
        return target


def _compute_align_embedding_metric_losses(
    pred: torch.Tensor,
    tgt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.float()
    tgt = tgt.float()
    l1_loss = F.l1_loss(pred, tgt)
    mse_loss = F.mse_loss(pred, tgt)
    pred_norm = F.normalize(pred, dim=-1, eps=1e-6)
    target_norm = F.normalize(tgt, dim=-1, eps=1e-6)
    cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
    return l1_loss, mse_loss, cosine_loss


def align_loss_components_dict(
    l1_loss: torch.Tensor,
    mse_loss: torch.Tensor,
    cosine_loss: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        "align_l1_loss": l1_loss,
        "align_mse_loss": mse_loss,
        "align_cosine_loss": cosine_loss,
    }


def compute_projected_align_loss(
    projected: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    normalized = normalize_align_loss_type(loss_type)
    l1_loss, mse_loss, cosine_loss = _compute_align_embedding_metric_losses(projected, target)
    components = align_loss_components_dict(l1_loss, mse_loss, cosine_loss)

    if normalized == "l1":
        return l1_loss, components
    if normalized == "mse":
        return mse_loss, components
    if normalized == "smooth_l1":
        return F.smooth_l1_loss(projected.float(), target.float()), components
    if normalized in {"mse_cosine", "cosine_mse"}:
        return mse_loss + cosine_loss, components
    if normalized == "cosine":
        return cosine_loss, components
    raise NotImplementedError(f"Align loss type {loss_type} is not implemented.")


def average_align_loss_components(
    component_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not component_dicts:
        return {}
    keys = component_dicts[0].keys()
    return {
        key: torch.stack([components[key] for components in component_dicts]).mean()
        for key in keys
    }


def reduce_align_loss_results(
    layer_results: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses, component_dicts = zip(*layer_results, strict=True)
    stacked = torch.stack(list(losses))
    components = average_align_loss_components(list(component_dicts))
    for i, layer_loss in enumerate(stacked):
        components[f"align_loss/layer_{i}"] = layer_loss.detach()
    return stacked.mean(), components


def _dedupe_align_layer_pairs_by_teacher(
    pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    seen_teachers: set[int] = set()
    deduped: List[Tuple[int, int]] = []
    for student_idx, teacher_idx in pairs:
        if teacher_idx in seen_teachers:
            continue
        seen_teachers.add(teacher_idx)
        deduped.append((student_idx, teacher_idx))
    return deduped


def _resolve_layer_mapping_stride(
    layer_mapping: str,
    layer_mapping_stride: Optional[int] = None,
) -> Optional[int]:
    if layer_mapping in _LAYER_MAPPING_STRIDE4_ALIASES:
        return 4
    if layer_mapping == "stride":
        return int(layer_mapping_stride) if layer_mapping_stride is not None else 4
    return None


def build_align_layer_pairs(
    student_num_layers: int,
    teacher_num_layers: int,
    layer_mapping: str = "uniform",
    student_layer_indices: Optional[List[int]] = None,
    teacher_layer_indices: Optional[List[int]] = None,
    layer_mapping_stride: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Build (student_hf_hidden_idx, teacher_list_idx) pairs.

    ``student_hf_hidden_idx`` is 1-indexed (1 = first transformer block).
    ``teacher_list_idx`` is 0-indexed into teacher hidden_states with embedding skipped.
    """
    layer_mapping = str(layer_mapping).lower()
    if layer_mapping == "custom":
        if not student_layer_indices or not teacher_layer_indices:
            raise ValueError(
                "layer_mapping=custom requires student_layer_indices and teacher_layer_indices"
            )
        if len(student_layer_indices) != len(teacher_layer_indices):
            raise ValueError(
                "student_layer_indices and teacher_layer_indices must have the same length, "
                f"got {len(student_layer_indices)} vs {len(teacher_layer_indices)}"
            )
        pairs: List[Tuple[int, int]] = []
        for student_idx, teacher_idx in zip(student_layer_indices, teacher_layer_indices, strict=True):
            student_idx = int(student_idx)
            teacher_idx = int(teacher_idx)
            if student_idx < 1 or teacher_idx < 1:
                raise ValueError(
                    "Layer indices must be 1-indexed transformer blocks "
                    f"(skip embedding); got student={student_idx}, teacher={teacher_idx}"
                )
            pairs.append((student_idx, teacher_idx - 1))
        return _dedupe_align_layer_pairs_by_teacher(pairs)

    if student_num_layers <= 0 or teacher_num_layers <= 0:
        raise ValueError(
            f"Invalid layer counts for alignment: student={student_num_layers}, teacher={teacher_num_layers}"
        )

    if layer_mapping == "uniform":
        student_layers = np.linspace(1, student_num_layers, student_num_layers, dtype=int)
        teacher_layers = np.linspace(1, teacher_num_layers, student_num_layers, dtype=int)
    elif layer_mapping == "top":
        if student_num_layers > teacher_num_layers:
            raise ValueError(
                "top layer_mapping requires student layers <= teacher layers, "
                f"got student={student_num_layers}, teacher={teacher_num_layers}"
            )
        offset = teacher_num_layers - student_num_layers
        student_layers = np.arange(1, student_num_layers + 1)
        teacher_layers = student_layers + offset
    elif layer_mapping == "match":
        if student_num_layers != teacher_num_layers:
            raise ValueError(
                "match layer_mapping requires equal student/teacher layer counts, "
                f"got student={student_num_layers}, teacher={teacher_num_layers}"
            )
        student_layers = np.arange(1, student_num_layers + 1)
        teacher_layers = student_layers
    elif layer_mapping == "tail":
        num_pairs = min(student_num_layers, teacher_num_layers)
        student_start = max(1, student_num_layers - teacher_num_layers + 1)
        student_layers = np.arange(student_start, student_start + num_pairs)
        teacher_layers = np.arange(1, num_pairs + 1)
    else:
        stride = _resolve_layer_mapping_stride(layer_mapping, layer_mapping_stride)
        if stride is None:
            raise ValueError(
                "layer_mapping must be one of "
                "{'uniform', 'top', 'match', 'tail', 'stride4', 'stride', 'custom'}"
            )
        if stride < 1:
            raise ValueError(f"layer_mapping_stride must be >= 1, got {stride}")
        student_layers = np.arange(1, student_num_layers + 1, stride, dtype=int)
        teacher_layers = np.arange(1, teacher_num_layers + 1, stride, dtype=int)
        num_pairs = min(len(student_layers), len(teacher_layers))
        student_layers = student_layers[:num_pairs]
        teacher_layers = teacher_layers[:num_pairs]

    pairs = [(int(s), int(t) - 1) for s, t in zip(student_layers, teacher_layers)]
    return _dedupe_align_layer_pairs_by_teacher(pairs)


def pool_vjepa_tubelet_tokens(
    feats: torch.Tensor,
    num_tubelets: int,
    pool_size: int,
) -> torch.Tensor:
    """Avg-pool each tubelet's spatial grid to ``pool_size x pool_size``."""
    batch_size, token_len, hidden_dim = feats.shape
    if token_len % num_tubelets != 0:
        raise RuntimeError(
            f"V-JEPA embedding length {token_len} is not divisible by temporal tokens {num_tubelets}"
        )
    tokens_per_tubelet = token_len // num_tubelets
    grid = int(tokens_per_tubelet**0.5)
    if grid * grid != tokens_per_tubelet:
        raise RuntimeError(f"V-JEPA tokens_per_tubelet {tokens_per_tubelet} is not a perfect square")
    x = feats.view(batch_size, num_tubelets, grid, grid, hidden_dim).permute(0, 1, 4, 2, 3)
    x = x.reshape(batch_size * num_tubelets, hidden_dim, grid, grid)
    x = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
    x = x.view(batch_size, num_tubelets, hidden_dim, pool_size, pool_size)
    return x.permute(0, 1, 3, 4, 2).reshape(
        batch_size, num_tubelets * pool_size * pool_size, hidden_dim
    )
