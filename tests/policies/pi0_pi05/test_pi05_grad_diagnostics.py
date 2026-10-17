#!/usr/bin/env python

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.grad_diagnostics import compute_grad_conflict_metrics  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import _loss_metrics  # noqa: E402
from lerobot.utils.logging_utils import MetricsTracker  # noqa: E402


def _metrics(action_direction, align_direction, scale=0.4):
    parameter = nn.Parameter(torch.tensor([0.3, -0.2]))
    action = (parameter * torch.tensor(action_direction)).sum()
    align = (parameter * torch.tensor(align_direction)).sum()
    return parameter, compute_grad_conflict_metrics(
        action, align, [("layer_0", [parameter])], align_scale=scale
    )


@pytest.mark.parametrize(
    ("action", "align", "expected"),
    [
        ([1.0, 0.0], [2.0, 0.0], 1.0),
        ([1.0, 0.0], [-2.0, 0.0], -1.0),
        ([1.0, 0.0], [0.0, 2.0], 0.0),
    ],
)
def test_grad_conflict_directions(action, align, expected):
    _, metrics = _metrics(action, align)
    assert metrics["grad_conflict/layer_0/cosine"] == pytest.approx(expected)
    assert metrics["grad_conflict/cosine_mean"] == pytest.approx(expected)


def test_grad_conflict_scale_and_unused_counts():
    common = nn.Parameter(torch.tensor([1.0, 2.0]))
    action_only = nn.Parameter(torch.tensor([3.0]))
    action = common.sum() + action_only.sum()
    align = (2 * common).sum()
    metrics = compute_grad_conflict_metrics(
        action, align, [("layer_0", [common, action_only])], align_scale=0.25
    )

    assert metrics["grad_conflict/layer_0/action_norm"] == pytest.approx(2**0.5)
    assert metrics["grad_conflict/layer_0/raw_align_norm"] == pytest.approx(8**0.5)
    assert metrics["grad_conflict/layer_0/weighted_align_norm"] == pytest.approx(0.5 * 2**0.5)
    assert metrics["grad_conflict/layer_0/raw_dot"] == pytest.approx(4.0)
    assert metrics["grad_conflict/layer_0/weighted_dot"] == pytest.approx(1.0)
    assert metrics["grad_conflict/layer_0/weighted_norm_ratio"] == pytest.approx(0.5)
    assert metrics["grad_conflict/common_params"] == 1.0
    assert metrics["grad_conflict/unused_action_params"] == 0.0
    assert metrics["grad_conflict/unused_align_params"] == 1.0


def test_grad_probe_does_not_write_param_grad_and_preserves_backward():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    probe_action = parameter.square().sum()
    probe_align = (parameter * torch.tensor([3.0, -1.0])).sum()
    compute_grad_conflict_metrics(
        probe_action, probe_align, [("layer_0", [parameter])], align_scale=0.4
    )
    assert parameter.grad is None

    total = parameter.square().sum() + 0.4 * (parameter * torch.tensor([3.0, -1.0])).sum()
    expected = torch.autograd.grad(total, parameter, retain_graph=True)[0]
    total.backward()
    assert torch.equal(parameter.grad, expected)


def test_grad_conflict_config_validation():
    with pytest.raises(ValueError, match="grad_conflict_freq"):
        PI05Config(grad_conflict_freq=0, dtype="float32")
    with pytest.raises(ValueError, match="grad_conflict_scope"):
        PI05Config(grad_conflict_scope="all", dtype="float32")


def test_loss_metric_keys_preserve_raw_and_weighted_relationship():
    metrics = _loss_metrics(torch.tensor(2.0), torch.tensor(3.0), align_scale=0.4)

    assert metrics == {
        "action_loss/raw": 2.0,
        "action_loss/weighted": 2.0,
        "align_loss/raw": 3.0,
        "align_loss/weighted": pytest.approx(1.2),
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_metrics_tracker_accepts_python_grad_scalars():
    _, metrics = _metrics([1.0, 0.0], [1.0, 0.0])
    assert all(isinstance(value, float) for value in metrics.values())
    tracker = MetricsTracker(batch_size=1, num_frames=1, num_episodes=1, metrics={})
    tracker.update_metrics(metrics)
    assert "grad_conflict/cosine_mean" in tracker.metrics
