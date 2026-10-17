"""Low-frequency PI05 action/alignment gradient diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def select_language_model_layer_param_groups(
    model: nn.Module, scope: str = "paligemma_language_model"
) -> list[tuple[str, list[nn.Parameter]]]:
    """Return trainable transformer-block parameters for the supported scope."""
    if scope != "paligemma_language_model":
        raise ValueError(f"Unsupported gradient-conflict scope: {scope}")
    layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    return [
        (f"layer_{index}", [param for param in layer.parameters() if param.requires_grad])
        for index, layer in enumerate(layers)
    ]


def _safe_grad(loss: Tensor, params: list[nn.Parameter], retain_graph: bool) -> tuple[Tensor | None, ...]:
    if not params or not loss.requires_grad:
        return (None,) * len(params)
    return torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)


def _stats_for_common_grads(
    action_grads: Iterable[Tensor | None], align_grads: Iterable[Tensor | None], align_scale: float
) -> tuple[dict[str, float], int, int, int]:
    action_sq = align_sq = weighted_sq = dot = weighted_dot = None
    action_count = align_count = common_count = 0
    for action_grad, align_grad in zip(action_grads, align_grads, strict=True):
        action_count += action_grad is not None
        align_count += align_grad is not None
        if action_grad is None or align_grad is None:
            continue
        common_count += 1
        action_fp32, align_fp32 = action_grad.float(), align_grad.float()
        weighted = align_fp32 * align_scale
        values = (
            action_fp32.pow(2).sum(),
            align_fp32.pow(2).sum(),
            weighted.pow(2).sum(),
            (action_fp32 * align_fp32).sum(),
            (action_fp32 * weighted).sum(),
        )
        if action_sq is None:
            action_sq, align_sq, weighted_sq, dot, weighted_dot = values
        else:
            action_sq = action_sq + values[0]
            align_sq = align_sq + values[1]
            weighted_sq = weighted_sq + values[2]
            dot = dot + values[3]
            weighted_dot = weighted_dot + values[4]

    if action_sq is None:
        return {
            "action_norm": 0.0,
            "raw_align_norm": 0.0,
            "weighted_align_norm": 0.0,
            "raw_dot": 0.0,
            "weighted_dot": 0.0,
            "cosine": 0.0,
            "weighted_cosine": 0.0,
            "weighted_norm_ratio": 0.0,
        }, action_count, align_count, common_count

    action_norm, align_norm, weighted_norm = (value.sqrt() for value in (action_sq, align_sq, weighted_sq))
    denominator = torch.finfo(torch.float32).eps
    cosine = dot / (action_norm * align_norm).clamp_min(denominator)
    weighted_cosine = weighted_dot / (action_norm * weighted_norm).clamp_min(denominator)
    return {
        "action_norm": float(action_norm.item()),
        "raw_align_norm": float(align_norm.item()),
        "weighted_align_norm": float(weighted_norm.item()),
        "raw_dot": float(dot.item()),
        "weighted_dot": float(weighted_dot.item()),
        "cosine": float(cosine.item()),
        "weighted_cosine": float(weighted_cosine.item()),
        "weighted_norm_ratio": float((weighted_norm / action_norm.clamp_min(denominator)).item()),
    }, action_count, align_count, common_count


def compute_grad_conflict_metrics(
    action_loss: Tensor,
    align_loss: Tensor,
    layer_param_groups: list[tuple[str, list[nn.Parameter]]],
    align_scale: float,
) -> dict[str, float]:
    """Compute per-layer gradient geometry without concatenating gradient tensors.

    Metrics are rank-local Python floats. The training ``MetricsTracker`` performs
    the eventual rank mean under DDP.
    """
    params = [param for _, group in layer_param_groups for param in group]
    action_grads = _safe_grad(action_loss, params, retain_graph=True)
    align_grads = _safe_grad(align_loss, params, retain_graph=False)
    metrics: dict[str, float] = {}
    valid_layers: list[tuple[int, float, float, float, float]] = []
    offset = 0
    total_action = total_align = total_common = 0
    for layer_index, (layer_name, group) in enumerate(layer_param_groups):
        end = offset + len(group)
        stats, action_count, align_count, common_count = _stats_for_common_grads(
            action_grads[offset:end], align_grads[offset:end], align_scale
        )
        offset = end
        prefix = f"grad_conflict/{layer_name}/"
        metrics.update({f"{prefix}{key}": value for key, value in stats.items()})
        metrics[f"{prefix}action_params"] = float(action_count)
        metrics[f"{prefix}align_params"] = float(align_count)
        metrics[f"{prefix}common_params"] = float(common_count)
        total_action += action_count
        total_align += align_count
        total_common += common_count
        if common_count:
            valid_layers.append(
                (
                    layer_index,
                    stats["cosine"],
                    stats["action_norm"],
                    stats["raw_align_norm"],
                    stats["weighted_align_norm"],
                )
            )
    if valid_layers:
        minimum = min(valid_layers, key=lambda values: values[1])
        metrics.update(
            {
                "grad_conflict/cosine_mean": sum(values[1] for values in valid_layers)
                / len(valid_layers),
                "grad_conflict/cosine_min": minimum[1],
                "grad_conflict/cosine_min_layer": float(minimum[0]),
                "grad_conflict/conflict_frac": sum(values[1] < 0 for values in valid_layers)
                / len(valid_layers),
                "grad_conflict/action_norm_mean": sum(values[2] for values in valid_layers)
                / len(valid_layers),
                "grad_conflict/raw_align_norm_mean": sum(values[3] for values in valid_layers)
                / len(valid_layers),
                "grad_conflict/weighted_align_norm_mean": sum(values[4] for values in valid_layers)
                / len(valid_layers),
                "grad_conflict/weighted_norm_ratio_mean": sum(
                    metrics[f"grad_conflict/layer_{values[0]}/weighted_norm_ratio"]
                    for values in valid_layers
                )
                / len(valid_layers),
            }
        )
    selected_params = len(params)
    metrics["grad_conflict/selected_params"] = float(selected_params)
    metrics["grad_conflict/action_params"] = float(total_action)
    metrics["grad_conflict/align_params"] = float(total_align)
    metrics["grad_conflict/common_params"] = float(total_common)
    metrics["grad_conflict/unused_action_params"] = float(selected_params - total_action)
    metrics["grad_conflict/unused_align_params"] = float(selected_params - total_align)
    return metrics
