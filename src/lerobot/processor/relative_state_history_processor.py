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

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_STATE

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry
from .relative_features import build_relative_mask, to_relative_features, to_se3_relative_features
from .relative_pose import find_pose_dimension_indices

STATE_HISTORY_DIMENSIONS = 3


@ProcessorStepRegistry.register("relative_state_history_processor")
@dataclass
class RelativeStateHistoryProcessorStep(ObservationProcessorStep):
    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    state_names: list[str] | None = None
    use_se3_relative_pose: bool = False

    def __post_init__(self) -> None:
        if self.use_se3_relative_pose and find_pose_dimension_indices(self.state_names or []) is None:
            raise ValueError(
                f"SE(3) relative poses need pose channels in the state names {self.state_names}."
            )

    def observation(self, observation: dict) -> dict:
        state = observation.get(OBS_STATE)
        if not self.enabled or not isinstance(state, Tensor) or state.ndim != STATE_HISTORY_DIMENSIONS:
            return observation
        relative_observation = dict(observation)
        relative_observation[OBS_STATE] = self.to_relative_state_history(state)
        return relative_observation

    def to_relative_state_history(self, state_history: Tensor) -> Tensor:
        # Every history step is expressed relative to the current step.
        current_state = state_history[:, -1]
        mask = build_relative_mask(self.state_names, self.exclude_joints, state_history.shape[-1])
        if self.use_se3_relative_pose:
            pose_indices = find_pose_dimension_indices(self.state_names or [])
            return to_se3_relative_features(state_history, current_state, mask, pose_indices)
        return to_relative_features(state_history, current_state, mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "state_names": self.state_names,
            "use_se3_relative_pose": self.use_se3_relative_pose,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
