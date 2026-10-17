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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_pi05 import PI05Config

INSTRUCTION_MASK = "observation.language.instruction_mask"


@ProcessorStepRegistry.register(name="pi05_alignment_tokenizer_processor")
@dataclass
class Pi05AlignmentTokenizerProcessorStep(TokenizerProcessorStep):
    """Keep exact instruction spans separate from PI05's serialized state."""

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        output = super().observation(observation)
        tasks = self.get_task(self.transition)
        raw_tasks = self.transition[TransitionKey.COMPLEMENTARY_DATA]["raw_task"]
        encoded = self.input_tokenizer(
            tasks,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        ids = output[OBS_LANGUAGE_TOKENS]
        if not torch.equal(encoded["input_ids"].to(ids.device), ids):
            raise RuntimeError("Instruction offsets do not match PI05 language tokens")
        offsets = encoded["offset_mapping"]
        ends = torch.tensor(
            [6 + len(task.strip().replace("_", " ").replace("\n", " ")) for task in raw_tasks]
        )
        # Include the leading space of a SentencePiece word, but never state or prompt markers.
        mask = (offsets[:, :, 1] > 6) & (offsets[:, :, 0] >= 5) & (offsets[:, :, 1] <= ends[:, None])
        mask &= encoded["attention_mask"].bool()
        output[INSTRUCTION_MASK] = mask.to(ids.device)
        return output


def reconcile_pi05_alignment_processor(config: PI05Config, preprocessor) -> None:
    """Upgrade pi05_base's saved tokenizer when starting teacher finetuning."""
    backend = config.alignment_backend
    if backend not in {"vggtomega", "locateanything"}:
        return
    if getattr(config, f"{backend}_scene_attention_mask_mode") != "image_and_instruction":
        return
    for index, step in enumerate(preprocessor.steps):
        if type(step) is TokenizerProcessorStep:
            kwargs = step.get_config()
            preprocessor.steps[index] = Pi05AlignmentTokenizerProcessorStep(
                **kwargs,
                tokenizer=step.input_tokenizer,
            )
            # Save a reloadable tokenizer name, rather than the transient tokenizer object.
            preprocessor.steps[index].tokenizer = None
            preprocessor.steps[index].tokenizer_name = step.tokenizer_name


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")
        transition[TransitionKey.COMPLEMENTARY_DATA] = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
        transition[TransitionKey.COMPLEMENTARY_DATA]["raw_task"] = list(tasks)

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        if state.ndim == 3:
            # [B, T, D] from observation_delta_indices when V-JEPA video clips are enabled
            state = state[:, 0]
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    preprocessor, postprocessor = make_policy_processor_pipelines(
        input_steps=input_steps, output_steps=output_steps
    )
    reconcile_pi05_alignment_processor(config, preprocessor)
    return preprocessor, postprocessor
