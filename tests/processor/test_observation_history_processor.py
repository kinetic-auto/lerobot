# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch

from lerobot.processor import ObservationHistoryProcessorStep, ProcessorStepRegistry, create_transition
from lerobot.processor.pipeline import TransitionKey
from lerobot.utils.constants import OBS_STATE

IMAGE_KEY = "observation.images.wrist"


def process_single_frame_observation(
    step: ObservationHistoryProcessorStep, state_value: float, image_value: float
) -> dict:
    observation = {
        OBS_STATE: torch.full((1, 2), state_value),
        IMAGE_KEY: torch.full((1, 3, 4, 4), image_value),
    }
    return step(create_transition(observation=observation))[TransitionKey.OBSERVATION]


def test_first_observation_is_repeated_to_fill_the_history():
    step = ObservationHistoryProcessorStep(state_history_length=3, image_history_length=2)
    observation = process_single_frame_observation(step, state_value=1.0, image_value=5.0)
    assert observation[OBS_STATE].shape == (1, 3, 2)
    assert observation[IMAGE_KEY].shape == (1, 2, 3, 4, 4)
    torch.testing.assert_close(observation[OBS_STATE][0, :, 0], torch.tensor([1.0, 1.0, 1.0]))


def test_history_is_ordered_oldest_to_current_and_spaced_by_the_stride():
    step = ObservationHistoryProcessorStep(state_history_length=3, observation_history_stride=2)
    for state_value in range(6):
        observation = process_single_frame_observation(step, state_value=float(state_value), image_value=0.0)
    torch.testing.assert_close(observation[OBS_STATE][0, :, 0], torch.tensor([1.0, 3.0, 5.0]))
    assert observation[IMAGE_KEY].shape == (1, 3, 4, 4)


def test_reset_clears_the_history():
    step = ObservationHistoryProcessorStep(state_history_length=2)
    process_single_frame_observation(step, state_value=1.0, image_value=0.0)
    step.reset()
    observation = process_single_frame_observation(step, state_value=7.0, image_value=0.0)
    torch.testing.assert_close(observation[OBS_STATE][0, :, 0], torch.tensor([7.0, 7.0]))


def test_observations_with_a_time_axis_are_returned_unchanged():
    step = ObservationHistoryProcessorStep(state_history_length=2, image_history_length=2)
    state_history = torch.randn(4, 2, 2)
    image_history = torch.randn(4, 2, 3, 4, 4)
    observation = step(create_transition(observation={OBS_STATE: state_history, IMAGE_KEY: image_history}))[
        TransitionKey.OBSERVATION
    ]
    assert observation[OBS_STATE] is state_history
    assert observation[IMAGE_KEY] is image_history


def test_step_is_rebuilt_from_its_registry_name_and_config():
    step = ObservationHistoryProcessorStep(
        state_history_length=2, image_history_length=3, observation_history_stride=4
    )
    restored = ProcessorStepRegistry.get("observation_history_processor")(**step.get_config())
    assert restored.get_config() == step.get_config()


def test_step_raises_on_a_history_length_below_one():
    with pytest.raises(ValueError, match="state_history_length"):
        ObservationHistoryProcessorStep(state_history_length=0)
