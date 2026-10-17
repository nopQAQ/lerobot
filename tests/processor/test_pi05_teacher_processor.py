"""Original task and exact instruction visibility survive saved PI05 processors."""

import pytest
import torch

pytest.importorskip("transformers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    INSTRUCTION_MASK,
    Pi05AlignmentTokenizerProcessorStep,
    Pi05PrepareStateTokenizerProcessorStep,
    reconcile_pi05_alignment_processor,
)
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.processor.converters import batch_to_transition, transition_to_batch


@pytest.fixture
def tokenizer_path(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    path = tmp_path / "tokenizer"
    fast.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("max_length", [6, 40])
def test_original_task_and_instruction_mask_roundtrip(tokenizer_path, tmp_path, max_length):
    pipeline = PolicyProcessorPipeline(
        steps=[
            Pi05PrepareStateTokenizerProcessorStep(),
            TokenizerProcessorStep(
                tokenizer_name=tokenizer_path,
                max_length=max_length,
                padding="max_length",
            ),
        ],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    # Mimic loading pi05_base's ordinary saved tokenizer before finetuning.
    config = PI05Config(device="cpu", locateanything_enabled=True)
    reconcile_pi05_alignment_processor(config, pipeline)
    assert isinstance(pipeline.steps[1], Pi05AlignmentTokenizerProcessorStep)
    task = "put_the red bowl on the plate"
    batch = {"task": [task], "observation.state": torch.zeros(1, 8)}
    processed = pipeline(batch)
    assert batch["task"] == [task]  # no mutation of dataset input
    assert processed["raw_task"] == [task]
    assert processed["task"][0].startswith("Task: put the red bowl on the plate, State:")
    mask = processed[INSTRUCTION_MASK]
    assert mask.shape == processed["observation.language.tokens"].shape
    offsets = pipeline.steps[1].input_tokenizer(
        processed["task"],
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )["offset_mapping"][0]
    for selected, (start, end) in zip(mask[0], offsets, strict=True):
        if selected:
            assert int(start) >= 6
            assert int(end) <= 6 + len(task.replace("_", " "))
    assert mask.any()
    # Exact boundary checks: neither Task punctuation nor the State label is visible.
    assert not mask[0, :2].any()
    assert transition_to_batch(batch_to_transition(processed))["raw_task"] == [task]
    path = tmp_path / "processor"
    pipeline.save_pretrained(path)
    restored = PolicyProcessorPipeline.from_pretrained(
        path,
        "dataprocessorpipeline.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    torch.testing.assert_close(restored(batch)[INSTRUCTION_MASK], mask)
