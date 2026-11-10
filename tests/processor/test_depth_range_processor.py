#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import DepthRangeProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

DEPTH_KEY = f"{OBS_IMAGES}.depth"
RGB_KEY = f"{OBS_IMAGES}.rgb"
VALID_DEPTH_RANGE_METERS = (0.06, 2.0)


def make_depth_step() -> DepthRangeProcessorStep:
    return DepthRangeProcessorStep(*VALID_DEPTH_RANGE_METERS)


def make_depth_millimeters(batch_shape: tuple[int, ...] = (2,)) -> torch.Tensor:
    depth_millimeters = torch.full((*batch_shape, 1, 4, 4), 300.0)
    depth_millimeters[..., :, 0] = 10000.0
    return depth_millimeters


def test_zeroes_pixels_outside_the_range_and_keeps_valid_depth_in_millimeters():
    depth_millimeters = make_depth_millimeters()
    rgb = torch.rand(2, 3, 4, 4)
    state = torch.rand(2, 5)
    transition = {
        TransitionKey.OBSERVATION: {DEPTH_KEY: depth_millimeters.clone(), RGB_KEY: rgb, OBS_STATE: state},
    }

    range_limited_observation = make_depth_step()(transition)[TransitionKey.OBSERVATION]

    assert torch.equal(range_limited_observation[DEPTH_KEY][..., 0], torch.zeros(2, 1, 4))
    assert torch.allclose(range_limited_observation[DEPTH_KEY][..., 1:], torch.full((2, 1, 4, 3), 300.0))
    assert torch.equal(range_limited_observation[RGB_KEY], rgb)
    assert torch.equal(range_limited_observation[OBS_STATE], state)
    assert torch.equal(transition[TransitionKey.OBSERVATION][DEPTH_KEY], depth_millimeters)


def test_zeroes_readings_below_the_minimum_and_beyond_the_maximum():
    depth_millimeters = torch.tensor([[[0.0, 59.0, float("nan"), 2001.0, 10000.0, float("inf"), 300.0]]])

    range_limited_depth = make_depth_step().observation({DEPTH_KEY: depth_millimeters})[DEPTH_KEY]

    assert torch.allclose(range_limited_depth, torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0]]]))


def test_keeps_depth_exactly_on_the_range_limits():
    depth_millimeters = torch.tensor([[[60.0, 2000.0]]])

    range_limited_depth = make_depth_step().observation({DEPTH_KEY: depth_millimeters})[DEPTH_KEY]

    assert torch.allclose(range_limited_depth, torch.tensor([[[60.0, 2000.0]]]))


def test_zeroes_an_unbatched_depth_image():
    range_limited_observation = make_depth_step().observation(
        {DEPTH_KEY: make_depth_millimeters(batch_shape=())}
    )

    assert range_limited_observation[DEPTH_KEY].shape == (1, 4, 4)
    assert range_limited_observation[DEPTH_KEY][0, 0, 0].item() == pytest.approx(0.0)
    assert range_limited_observation[DEPTH_KEY][0, 0, 1].item() == pytest.approx(300.0)


def test_ignores_single_channel_entries_that_are_not_images():
    wide_state = torch.full((2, 1, 8), 5000.0)

    result = make_depth_step().observation({OBS_STATE: wide_state})

    assert torch.equal(result[OBS_STATE], wide_state)


def test_is_registered_and_round_trips_its_config():
    step_class = ProcessorStepRegistry.get("depth_range_processor")
    step = DepthRangeProcessorStep(0.1, 1.5)

    restored = step_class(**step.get_config())

    assert step_class is DepthRangeProcessorStep
    assert (restored.minimum_depth_meters, restored.maximum_depth_meters) == (0.1, 1.5)


def test_leaves_features_unchanged():
    features = {PipelineFeatureType.OBSERVATION: {DEPTH_KEY: PolicyFeature(FeatureType.VISUAL, (1, 4, 4))}}

    assert make_depth_step().transform_features(features) == features


@pytest.mark.parametrize(
    ("minimum_depth_meters", "maximum_depth_meters", "expected_error_pattern"),
    [
        (-0.1, 2.0, "minimum_depth_meters"),
        (2.0, 2.0, "maximum_depth_meters"),
        (2.0, 0.06, "maximum_depth_meters"),
    ],
)
def test_rejects_an_invalid_range(minimum_depth_meters, maximum_depth_meters, expected_error_pattern):
    with pytest.raises(ValueError, match=expected_error_pattern):
        DepthRangeProcessorStep(minimum_depth_meters, maximum_depth_meters)
