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

from lerobot.datasets.joint_layout import (
    feature_channel_names,
    io_block_layout,
    io_layout_from_dataset_features,
    packed_feature_names,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_packed_layout_roundtrip():
    names = packed_feature_names(["shoulder", "gripper"], ["position", "effort"])
    assert io_block_layout(names) == (
        ["shoulder", "gripper"],
        ["position", "effort"],
    )


def test_feature_channel_names_flattens_groups():
    assert feature_channel_names({"names": {"arm": ["a", "b"], "hand": ["g"]}}) == [
        "a",
        "b",
        "g",
    ]
    assert feature_channel_names({"names": {"a": 0, "b": 1}}) == ["a", "b"]


def test_io_layout_from_dataset_features():
    features = {
        OBS_STATE: {"names": ["shoulder.position", "gripper.position"]},
        ACTION: {
            "names": [
                "shoulder.position",
                "gripper.position",
                "shoulder.effort",
                "gripper.effort",
            ]
        },
    }
    assert io_layout_from_dataset_features(features) == {
        "joint_names": ["shoulder", "gripper"],
        "state_blocks": ["position"],
        "action_blocks": ["position", "effort"],
    }


def test_io_layout_rejects_mismatched_joints():
    with pytest.raises(ValueError, match="differ"):
        io_layout_from_dataset_features(
            {
                OBS_STATE: {"names": ["shoulder"]},
                ACTION: {"names": ["wrist"]},
            }
        )
