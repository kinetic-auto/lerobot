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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE


def test_act_rejects_invalid_relative_action_mode():
    with pytest.raises(ValueError, match="relative_action_mode"):
        ACTConfig(relative_action_mode="invalid")


def test_act_rejects_relative_actions_with_temporal_ensembling():
    with pytest.raises(ValueError, match="incompatible"):
        ACTConfig(
            use_relative_actions=True,
            temporal_ensemble_coeff=0.01,
            n_action_steps=1,
        )


def test_act_relative_actions_require_robot_state():
    config = ACTConfig(use_relative_actions=True)
    config.input_features = {OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    with pytest.raises(ValueError, match="observation.state"):
        config.validate_features()
