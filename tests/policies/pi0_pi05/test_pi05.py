#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Test script to verify PI0.5 (pi05) support in PI0 policy"""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.common.vla_utils import make_att_2d_masks  # noqa: E402
from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,  # noqa: E402
)
from lerobot.policies.pi05.align import (  # noqa: E402
    scene_prealign_consistency_stats,
    scene_query_embedding_consistency_stats,
)
from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    PI05Pytorch,
    _reorder_prefix_scene_behind_action,
    _restitch_attention_output_to_module_order,
)
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_hf_token  # noqa: E402


def test_vjepa_scene_attention_is_strictly_auxiliary():
    attention = torch.ones(1, 8, 8, dtype=torch.bool)
    padding = torch.ones(1, 8, dtype=torch.bool)

    restricted = PI05Pytorch._restrict_scene_query_attention(
        None,
        attention,
        padding,
        prefix_len=3,
        scene_len=2,
        action_len=3,
    )

    base_prefix = slice(0, 3)
    action = slice(3, 6)
    scene = slice(6, 8)
    assert restricted[:, scene, base_prefix].all()
    assert not restricted[:, base_prefix, scene].any()
    assert not restricted[:, scene, scene].any()
    assert not restricted[:, scene, action].any()
    assert not restricted[:, action, scene].any()
    assert restricted[:, action, base_prefix].all()


def test_vjepa_attention_states_place_scene_after_action():
    prefix = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)
    suffix = torch.arange(10, 13, dtype=torch.float32).view(1, 1, 3, 1)
    reordered = _reorder_prefix_scene_behind_action([prefix, suffix], scene_len=2)

    assert reordered.shape[2] == 9
    assert torch.equal(reordered[0, 0, :4, 0], prefix[0, 0, :4, 0])
    assert torch.equal(reordered[0, 0, 4:7, 0], suffix[0, 0, :, 0])
    assert torch.equal(reordered[0, 0, 7:, 0], prefix[0, 0, 4:, 0])

    restitched = _restitch_attention_output_to_module_order(
        reordered.squeeze(1),
        prefix_len=6,
        suffix_len=3,
        scene_len=2,
    )
    original = torch.cat([prefix, suffix], dim=2).squeeze(1)
    assert torch.equal(restitched, original)


def test_vjepa_scene_query_ids_follow_vlajepa_tubelet_layout():
    temporal_tokens = 4
    spatial_tokens = 64
    token_ids = PI05Pytorch._scene_query_token_ids(
        temporal_tokens, spatial_tokens, device=torch.device("cpu")
    )
    assert token_ids.shape == (temporal_tokens * spatial_tokens,)
    for tubelet in range(temporal_tokens):
        start = tubelet * spatial_tokens
        end = start + spatial_tokens
        assert torch.equal(token_ids[start:end], torch.full((spatial_tokens,), tubelet))


def test_scene_prealign_stats_identical_tokens_are_consistent():
    temporal = 4
    spatial = 8
    hidden = 6
    token = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scene = token.view(1, 1, hidden).expand(2, temporal * spatial, hidden).contiguous()
    stats = scene_prealign_consistency_stats(scene, temporal)
    assert stats["scene_prealign/pairwise_cosine"].item() == pytest.approx(1.0, abs=1e-5)
    assert stats["scene_prealign/within_tubelet_cosine"].item() == pytest.approx(1.0, abs=1e-5)
    assert stats["scene_prealign/between_tubelet_cosine"].item() == pytest.approx(1.0, abs=1e-5)
    assert stats["scene_prealign/tubelet_mean_cosine"].item() == pytest.approx(1.0, abs=1e-5)
    assert stats["scene_prealign/token_std"].item() == pytest.approx(0.0, abs=1e-5)

    query = token.expand(temporal, hidden).contiguous()
    query_stats = scene_query_embedding_consistency_stats(query)
    assert query_stats["scene_prealign/query_pairwise_cosine"].item() == pytest.approx(1.0, abs=1e-5)
    assert query_stats["scene_prealign/query_row_std"].item() == pytest.approx(0.0, abs=1e-5)


def test_scene_prealign_stats_separate_tubelets_diverge():
    temporal = 2
    spatial = 4
    t0 = torch.tensor([1.0, 0.0])
    t1 = torch.tensor([0.0, 1.0])
    scene = torch.cat(
        [t0.expand(spatial, 2), t1.expand(spatial, 2)],
        dim=0,
    ).unsqueeze(0)
    stats = scene_prealign_consistency_stats(scene, temporal)
    assert stats["scene_prealign/within_tubelet_cosine"].item() == pytest.approx(1.0, abs=1e-5)
    assert stats["scene_prealign/between_tubelet_cosine"].item() == pytest.approx(0.0, abs=1e-5)
    assert stats["scene_prealign/tubelet_mean_cosine"].item() == pytest.approx(0.0, abs=1e-5)


def test_vjepa_embed_scene_repeats_per_tubelet_embedding():
    model = object.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = PI05Config(dtype="float32")
    model.vjepa_temporal_tokens = 2
    spatial = model.config.vjepa_spatial_pool_size**2
    model.vjepa_scene_len = 2 * spatial
    model.scene_query = nn.Embedding.from_pretrained(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), freeze=False
    )
    scene_emb, pad_masks, _ = PI05Pytorch.embed_scene(
        model, bsize=3, device=torch.device("cpu"), dtype=torch.float32
    )
    assert scene_emb.shape == (3, 2 * spatial, 2)
    assert pad_masks.shape == (3, 2 * spatial)
    assert torch.equal(scene_emb[0, :spatial], torch.tensor([1.0, 0.0]).expand(spatial, -1))
    assert torch.equal(scene_emb[0, spatial:], torch.tensor([0.0, 1.0]).expand(spatial, -1))
    assert torch.equal(scene_emb[0], scene_emb[1])


def test_vjepa_block_masks_keep_scene_auxiliary_after_action():
    pad = torch.ones(1, 8, dtype=torch.bool)
    att_1d = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 0]], dtype=torch.bool)
    masks = make_att_2d_masks(pad, att_1d)
    restricted = PI05Pytorch._restrict_scene_query_attention(
        None,
        masks,
        pad,
        prefix_len=3,
        scene_len=2,
        action_len=3,
    )

    base_prefix = slice(0, 3)
    action = slice(3, 6)
    scene = slice(6, 8)
    assert restricted[:, scene, base_prefix].all()
    assert not restricted[:, base_prefix, scene].any()
    assert not restricted[:, scene, scene].any()
    assert not restricted[:, scene, action].any()
    assert not restricted[:, action, scene].any()
    assert restricted[:, action, base_prefix].all()


def test_vjepa_align_projector_uses_separate_lr():
    class DummyProjector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2, bias=False)
            self.vjepa_align_projector = DummyProjector()

    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PI05Config(
                vjepa_enabled=True,
                optimizer_lr=5e-5,
                vjepa_align_projector_lr=1e-4,
                vjepa_align_loss_scale=0.4,
                dtype="float32",
            )
            self.model = DummyModel()

    policy = DummyPolicy()
    groups = PI05Policy.get_optim_params(policy)

    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(5e-5)
    assert groups[1]["lr"] == pytest.approx(1e-4)
    assert groups[1]["name"] == "vjepa_align_projector"
    assert len(groups[0]["params"]) == 1
    assert len(groups[1]["params"]) == 1
    assert groups[1]["params"][0] is policy.model.vjepa_align_projector.fc.weight


def test_vjepa_scene_tokens_do_not_shift_action_positions():
    base_prefix_padding = torch.tensor([[True, True, False]])
    suffix_padding = torch.tensor([[True, True]])
    scene_padding = torch.tensor([[True, True]])

    training_positions = PI05Pytorch._build_joint_position_ids(
        base_prefix_padding,
        suffix_padding,
        scene_pad_masks=scene_padding,
    )
    inference_positions = PI05Pytorch._build_joint_position_ids(
        base_prefix_padding,
        suffix_padding,
    )

    prefix_len = base_prefix_padding.shape[1]
    action_len = suffix_padding.shape[1]
    assert torch.equal(
        training_positions[:, prefix_len : prefix_len + action_len],
        inference_positions[:, prefix_len:],
    )
    assert training_positions.shape[1] == prefix_len + action_len + scene_padding.shape[1]


def test_vjepa_float_clip_is_not_rescaled_twice():
    class CaptureProcessor:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"pixel_values_videos": kwargs["videos"].unsqueeze(0)}

    processor = CaptureProcessor()
    policy = object.__new__(PI05Policy)
    object.__setattr__(policy, "vj_processor", processor)
    clip = torch.rand(8, 3, 16, 16)

    processed = PI05Policy._process_vjepa_clip(policy, clip)

    assert processor.kwargs["do_rescale"] is False
    assert torch.equal(processed, clip.unsqueeze(0))


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05Policy(config)
    # Test forward pass with dummy data
    batch_size = 1
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise

    # Verify pi05 model components exist
    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="pi0",
            max_action_dim=7,
            max_state_dim=14,
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
        print(f"  PaliGemma variant: {config.paligemma_variant}")
        print(f"  Action expert variant: {config.action_expert_variant}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise
