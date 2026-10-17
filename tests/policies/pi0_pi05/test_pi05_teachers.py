"""Single-teacher contracts independent of large checkpoints and GPU hardware."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs import PolicyFeature
from lerobot.configs.types import FeatureType
from lerobot.policies.pi05 import modeling_pi05 as modeling
from lerobot.policies.pi05.align import (
    AlignProjector,
    build_align_layer_pairs,
    scene_prealign_consistency_stats,
    scene_query_embedding_consistency_stats,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.locateanything import LocateAnythingBackbone, format_locateanything_grounding_text
from lerobot.policies.pi05.teachers import (
    LocateAnythingTeacher,
    VGGTOmegaTeacher,
    _restore_locateanything_rope,
    teacher_image_batch,
)


@pytest.mark.parametrize("backend,scale", [("vggtomega", 0.05), ("locateanything", 0.1)])
def test_single_teacher_config(backend, scale):
    config = PI05Config(device="cpu", **{f"{backend}_enabled": True})
    assert config.alignment_backend == backend
    assert config.alignment_loss_scale == scale
    assert config.observation_delta_indices == (list(range(8)) if backend == "vggtomega" else None)
    with pytest.raises(ValueError, match="one alignment teacher"):
        PI05Config(device="cpu", vjepa_enabled=True, **{f"{backend}_enabled": True})


@pytest.mark.parametrize("layout", ["bchw", "bhwc", "btchw", "bthwc"])
def test_teacher_image_range_and_layout(layout):
    pixels = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).to(torch.uint8)
    images = pixels.permute(0, 2, 3, 1) if layout.endswith("hwc") else pixels
    if layout.startswith("bt"):
        images = images.unsqueeze(1)
    expected = pixels.float().unsqueeze(1) / 255
    torch.testing.assert_close(teacher_image_batch(images), expected)
    torch.testing.assert_close(teacher_image_batch(images.float() / 255), expected)


def test_vggt_uses_video_context_but_only_frame_zero_patches():
    class Aggregator(nn.Module):
        def forward(self, video):
            assert video.shape == (2, 3, 3, 16, 16)
            # Distinct frame / special-token values detect incorrect slicing.
            features = torch.arange(2 * 3 * 7 * 4).reshape(2, 3, 7, 4).float()
            self.expected = features[:, 0, 3:]
            return [features, features + 1], 3

    model = nn.Module()
    model.aggregator = Aggregator()
    teacher = VGGTOmegaTeacher(model, num_frames=3, image_size=16)
    results = teacher(torch.rand(2, 5, 3, 32, 32))
    torch.testing.assert_close(results[0], model.aggregator.expected)
    torch.testing.assert_close(results[1], model.aggregator.expected + 1)
    with pytest.raises(ValueError, match="real video frames"):
        teacher(torch.rand(2, 1, 3, 32, 32))


@pytest.mark.parametrize(
    "task,expected",
    [
        (
            "put the black bowl in the bottom drawer of the cabinet",
            "black bowl</c>bottom drawer of the cabinet",
        ),
        ("turn on the stove and put the moka pot on it", "moka pot</c>stove"),
        (
            "put both the alphabet soup and the cream cheese box in the basket",
            "alphabet soup</c>cream cheese box</c>basket",
        ),
    ],
)
def test_locateanything_reference_prompts(task, expected):
    assert format_locateanything_grounding_text(task) == expected


def test_locateanything_image_token_gather_preserves_order():
    hidden = torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    ids = torch.tensor([[1, 9, 9, 2, 9, 0], [9, 2, 9, 9, 0, 0]])
    output = LocateAnythingBackbone._gather_image_token_embeddings(hidden, ids, 9, 3)
    torch.testing.assert_close(output[0], hidden[0, [1, 2, 4]])
    torch.testing.assert_close(output[1], hidden[1, [0, 2, 3]])
    with pytest.raises(RuntimeError, match="insufficient"):
        LocateAnythingBackbone._gather_image_token_embeddings(hidden, ids, 9, 4)


def test_locateanything_requires_original_task():
    teacher = LocateAnythingTeacher(nn.Identity())
    with pytest.raises(ValueError, match="raw_task"):
        teacher(torch.rand(1, 1, 3, 16, 16))


def test_locateanything_scene_length_follows_processor_grid():
    grids = [(20, 20), (20, 20)]
    wrapper = SimpleNamespace(
        processor=SimpleNamespace(image_processor=lambda **kwargs: {"image_grid_hws": grids}),
        merge_kernel_size=(2, 2),
    )
    assert LocateAnythingBackbone.scene_token_len(wrapper, [None, None]) == 100
    grids[:] = [(28, 28), (28, 28)]
    assert LocateAnythingBackbone.scene_token_len(wrapper, [None, None]) == 196
    grids[1] = (20, 20)
    with pytest.raises(RuntimeError, match="equal main-view token counts"):
        LocateAnythingBackbone.scene_token_len(wrapper, [None, None])


def test_single_query_diagnostics_are_finite():
    metrics = scene_query_embedding_consistency_stats(torch.randn(1, 8))
    metrics.update(scene_prealign_consistency_stats(torch.randn(2, 1, 8), 1))
    assert all(value.isfinite() for value in metrics.values())


def test_legacy_rope_buffers_are_rebuilt_after_meta_loading():
    class Qwen2RotaryEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = 1000000.0
            self.dim = 8
            self.max_position_embeddings = 10
            self.register_buffer("inv_freq", torch.full((4,), float("nan")), persistent=False)

        def _set_cos_sin_cache(self, seq_len, device, dtype):
            frequencies = torch.outer(torch.arange(seq_len, device=device), self.inv_freq)
            angles = torch.cat([frequencies, frequencies], dim=-1)
            self.cos_cached = angles.cos().to(dtype)
            self.sin_cached = angles.sin().to(dtype)

    rope = Qwen2RotaryEmbedding()
    model = SimpleNamespace(language_model=nn.Sequential(rope))
    _restore_locateanything_rope(model)
    torch.testing.assert_close(rope.inv_freq, torch.tensor([1, 10**-1.5, 1e-3, 10**-4.5]))
    assert rope.cos_cached.isfinite().all() and rope.sin_cached.isfinite().all()
    torch.testing.assert_close(rope.cos_cached[0], torch.ones(8))
    torch.testing.assert_close(rope.sin_cached[0], torch.zeros(8))


@pytest.mark.parametrize("instruction", [False, True])
def test_scene_visibility_excludes_state_actions_and_other_scene(instruction):
    # 2 image, 2 instruction, 1 state, 2 action, 2 scene positions.
    padding = torch.ones(2, 9, dtype=torch.bool)
    padding[1, 1] = False
    keys = torch.tensor([[True, True, instruction, instruction, False]]).expand(2, -1)
    result = modeling.PI05Pytorch._restrict_scene_query_attention(
        None,
        torch.ones(2, 9, 9, dtype=torch.bool),
        padding,
        prefix_len=5,
        scene_len=2,
        action_len=2,
        scene_key_mask=keys,
    )
    assert not result[:, :7, 7:].any()
    assert not result[:, 7:, 4:].any()
    assert result[0, 7:, :2].all()
    assert result[0, 7:, 2:4].all().item() == instruction
    assert not result[1, 7:, 1].any()


@pytest.mark.parametrize("backend", ["vggtomega", "locateanything"])
def test_alignment_backward_and_projector_optimizer_group(backend):
    config = PI05Config(device="cpu", **{f"{backend}_enabled": True})
    policy = modeling.PI05Policy.__new__(modeling.PI05Policy)
    nn.Module.__init__(policy)
    policy.config = config
    core = modeling.PI05Pytorch.__new__(modeling.PI05Pytorch)
    nn.Module.__init__(core)
    core.config = config
    core.alignment_scene_len = 5
    core.alignment_layer_pairs = build_align_layer_pairs(2, 3, "uniform")
    projector = AlignProjector(8, 4, use_vlm_norm=True, use_target_norm=True)
    setattr(core, f"{backend}_align_projector", projector)
    setattr(core, f"{backend}_scene_query", nn.Embedding(1, 8))
    core.student = nn.Linear(8, 8)
    policy.model = core
    scene, _, _ = core.embed_scene(2, torch.device("cpu"), torch.float32)
    student = [core.student(scene), core.student(scene) + 0.1]
    targets = [torch.randn(2, 5, 4) for _ in range(3)]
    loss, _ = policy._compute_alignment_loss(student, targets)
    loss.backward()
    assert core.active_scene_query.weight.grad.abs().sum() > 0
    assert core.student.weight.grad.abs().sum() > 0
    assert projector.fc1.weight.grad.abs().sum() > 0
    assert projector.target_norm.weight.grad.abs().sum() > 0
    assert all(target.grad is None for target in targets)
    groups = policy.get_optim_params()
    assert groups[1]["name"] == f"{backend}_align_projector"
    assert groups[1]["lr"] == 1e-4
    assert {id(p) for p in groups[1]["params"]} == {id(p) for p in projector.parameters()}


@pytest.mark.parametrize("backend", ["vggtomega", "locateanything"])
def test_checkpoint_excludes_teacher_and_reload_needs_no_teacher_files(tmp_path, monkeypatch, backend):
    class TinyCore(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            setattr(self, f"{backend}_scene_query", nn.Embedding(1, 4))
            setattr(self, f"{backend}_align_projector", nn.Linear(4, 4))

        def sample_actions(self, images, *args, **kwargs):
            return torch.zeros(images[0].shape[0], 2, 2)

    monkeypatch.setattr(modeling, "PI05Pytorch", TinyCore)
    config = PI05Config(
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        image_resolution=(16, 16),
        input_features={"observation.images.main": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        vggtomega_source_path="/missing/source",
        locateanything_model_path="/missing/model",
        **{f"{backend}_enabled": True},
    )
    policy = modeling.PI05Policy(config)
    policy._alignment_teacher = nn.Linear(4, 4).requires_grad_(False)
    policy.train()
    assert not policy._alignment_teacher.training
    policy.save_pretrained(tmp_path)
    from safetensors.torch import load_file

    saved = load_file(str(tmp_path / "model.safetensors"))
    assert not any("_alignment_teacher" in key for key in saved)
    assert any(f"{backend}_scene_query" in key for key in saved)
    restored = modeling.PI05Policy.from_pretrained(tmp_path)
    assert restored._alignment_teacher is None
    for key, value in saved.items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    restored.predict_action_chunk(
        {
            "observation.images.main": torch.rand(1, 3, 16, 16),
            "observation.language.tokens": torch.ones(1, 3, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    )
    assert restored._alignment_teacher is None
