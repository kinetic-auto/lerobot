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

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry

# Dimensions of one batched observation (without history).
STATE_DIMENSIONS_WITHOUT_HISTORY = 2
IMAGE_DIMENSIONS_WITHOUT_HISTORY = 4


@ProcessorStepRegistry.register("observation_history_processor")
@dataclass
class ObservationHistoryProcessorStep(ObservationProcessorStep):
    state_history_length: int = 1
    image_history_length: int = 1
    observation_history_stride: int = 1
    _buffers: dict[str, deque[Tensor]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("state_history_length", "image_history_length", "observation_history_stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"`{name}` must be at least 1. Got {getattr(self, name)}.")

    def observation(self, observation: dict) -> dict:
        stacked_observation = dict(observation)
        for observation_key, observation_value in observation.items():
            history_length = self._get_history_length(observation_key)
            if history_length > 1 and isinstance(observation_value, Tensor) and not self._has_history(observation_key, observation_value):
                stacked_observation[observation_key] = self._stack_history(observation_key, observation_value, history_length)
        return stacked_observation

    def _get_history_length(self, key: str) -> int:
        if key == OBS_STATE:
            return self.state_history_length
        if key.startswith(OBS_IMAGE):
            return self.image_history_length
        return 1

    def _has_history(self, key: str, value: Tensor) -> bool:
        return value.ndim > (STATE_DIMENSIONS_WITHOUT_HISTORY if key == OBS_STATE else IMAGE_DIMENSIONS_WITHOUT_HISTORY)

    def _stack_history(self, key: str, value: Tensor, history_length: int) -> Tensor:
        buffer_length = (history_length - 1) * self.observation_history_stride + 1
        buffer = self._buffers.get(key)
        if buffer is None:
            # The first observation of an episode fills the history, as LeRobotDataset pads at an episode start.
            buffer = deque([value] * buffer_length, maxlen=buffer_length)
            self._buffers[key] = buffer
        else:
            buffer.append(value)
        oldest_to_current = list(buffer)[:: -self.observation_history_stride][::-1]
        return torch.stack(oldest_to_current, dim=1)

    def reset(self) -> None:
        self._buffers.clear()

    def get_config(self) -> dict[str, Any]:
        return {
            "state_history_length": self.state_history_length,
            "image_history_length": self.image_history_length,
            "observation_history_stride": self.observation_history_stride,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
