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

import json
import os

import numpy as np
import pytest

from lerobot.ros2.observation import (
    policy_state_action_layout_from_checkpoint,
    resize_image_with_pad,
    map_joint_names,
    build_observation_state,
    resolve_checkpoint_dir,
    build_action,
)

MODEL_JOINTS = [
    "right_finger_joint1",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
]

def test_build_observation_state_orders_position_then_effort():
    positions = [float(index) for index in range(7)]
    efforts = [float(index + 10) for index in range(7)]
    packed = build_observation_state(
        joint_state_values={"position": positions, "effort": efforts},
        joint_state_types=["position", "effort"],
    )
    assert packed.dtype == np.float32
    assert packed.shape == (14,)
    assert packed.tolist() == pytest.approx(positions + efforts)

def test_resize_image_with_pad_pads_1080p_into_480_by_640():
    pytest.importorskip("cv2")
    image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    resized = resize_image_with_pad(image, 480, 640)
    assert resized.shape == (480, 640, 3)
    assert resized.dtype == np.uint8
    assert int(resized[:60].sum()) == 0
    assert int(resized[420:].sum()) == 0
    assert int(resized[60:420].min()) == 255

def test_map_joint_names_rewrites_the_arm_prefix():
    mapped = map_joint_names(
        ["follower_r_finger_joint1", "follower_r_joint1"],
        "follower_r_",
        "right_",
    )
    assert mapped == ["right_finger_joint1", "right_joint1"]
    assert map_joint_names(MODEL_JOINTS, "right_", "follower_r_") == [
        "follower_r_finger_joint1",
        "follower_r_joint1",
        "follower_r_joint2",
        "follower_r_joint3",
        "follower_r_joint4",
        "follower_r_joint5",
        "follower_r_joint6",
    ]

def test_map_joint_names_rejects_a_foreign_prefix():
    with pytest.raises(ValueError, match="follower_l_joint1"):
        map_joint_names(["follower_l_joint1"], "follower_r_", "right_")

def test_policy_state_action_layout_from_checkpoint_reads_metadata(tmp_path):
    checkpoint_dir = os.fspath(tmp_path)
    _write_checkpoint(
        checkpoint_dir,
        {
            "io": {
                "joint_names": MODEL_JOINTS,
                "state_blocks": ["position", "effort"],
                "action_blocks": ["position", "effort"],
            }
        },
    )
    layout = policy_state_action_layout_from_checkpoint(resolve_checkpoint_dir(checkpoint_dir))
    assert layout.joint_names == tuple(MODEL_JOINTS)
    assert layout.state_types == ("position", "effort")
    assert layout.action_types == ("position", "effort")

def test_policy_state_action_layout_from_checkpoint_requires_the_io_block(tmp_path):
    checkpoint_dir = os.fspath(tmp_path)
    _write_checkpoint(checkpoint_dir, {})
    with pytest.raises(RuntimeError, match="has no io block"):
        policy_state_action_layout_from_checkpoint(resolve_checkpoint_dir(checkpoint_dir))

def test_resolve_checkpoint_dir_accepts_the_pretrained_model_child(tmp_path):
    checkpoint_dir = os.fspath(tmp_path)
    _write_checkpoint(checkpoint_dir, {})
    pretrained = os.path.join(checkpoint_dir, "pretrained_model")
    assert resolve_checkpoint_dir(checkpoint_dir) == pretrained
    assert resolve_checkpoint_dir(pretrained) == pretrained

def test_build_action_separates_position_and_effort():
    action = np.arange(14, dtype=np.float32)
    blocks = build_action(action, 7, ["position", "effort"])
    assert blocks["position"].tolist() == pytest.approx(list(range(7)))
    assert blocks["effort"].tolist() == pytest.approx(list(range(7, 14)))

def _write_checkpoint(path: str, metadata: dict) -> None:
    pretrained = os.path.join(path, "pretrained_model")
    os.makedirs(pretrained, exist_ok=True)
    with open(os.path.join(pretrained, "config.json"), "w", encoding="utf-8") as config_file:
        config_file.write("{}")
    with open(
        os.path.join(pretrained, "training_metadata.json"),
        "w",
        encoding="utf-8",
    ) as metadata_file:
        metadata_file.write(json.dumps(metadata))
