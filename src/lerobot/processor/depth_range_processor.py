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
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.depth_range import zero_depth_outside_valid_range

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry

SINGLE_CHANNEL = 1


@ProcessorStepRegistry.register("depth_range_processor")
@dataclass
class DepthRangeProcessorStep(ObservationProcessorStep):
    """Set depth outside the configured range to zero millimeters.

    Attributes:
        minimum_depth_meters (float): Minimum valid depth in meters.
        maximum_depth_meters (float): Maximum valid depth in meters.
    """

    minimum_depth_meters: float
    maximum_depth_meters: float

    def __post_init__(self) -> None:
        if self.minimum_depth_meters < 0:
            raise ValueError(f"`minimum_depth_meters` must be non-negative. Got {self.minimum_depth_meters}.")
        if self.maximum_depth_meters <= self.minimum_depth_meters:
            raise ValueError(
                f"`maximum_depth_meters` must be greater than `minimum_depth_meters`. Got {self.maximum_depth_meters} <= {self.minimum_depth_meters}."
            )

    def observation(self, observation: dict) -> dict:
        range_limited_observation = dict(observation)
        for key, value in observation.items():
            if key.startswith(OBS_IMAGES) and _is_single_channel_image(value):
                range_limited_observation[key] = zero_depth_outside_valid_range(
                    value, self.minimum_depth_meters, self.maximum_depth_meters
                )
        return range_limited_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "minimum_depth_meters": self.minimum_depth_meters,
            "maximum_depth_meters": self.maximum_depth_meters,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _is_single_channel_image(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[-3] == SINGLE_CHANNEL
