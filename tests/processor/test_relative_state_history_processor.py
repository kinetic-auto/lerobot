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

from lerobot.model.cartesian_quantities import FramePoseRot6D, FrameWrench
from lerobot.processor import RelativeStateHistoryProcessorStep, create_transition
from lerobot.processor.pipeline import TransitionKey
from lerobot.processor.relative_pose import convert_absolute_poses_to_relative_poses
from lerobot.utils.constants import OBS_STATE

STATE_NAMES = [*FramePoseRot6D.feature_names(), "gripper.position.x", *FrameWrench.feature_names()]


def build_random_state_history(batch_size: int, history_length: int) -> torch.Tensor:
    rotations = torch.linalg.qr(torch.randn(batch_size, history_length, 3, 3, dtype=torch.float64)).Q
    rotations = rotations * torch.linalg.det(rotations).sign()[..., None, None]
    return torch.cat(
        [
            torch.randn(batch_size, history_length, 3, dtype=torch.float64),
            rotations[..., :, 0],
            rotations[..., :, 1],
            torch.rand(batch_size, history_length, 1, dtype=torch.float64),
            torch.randn(batch_size, history_length, 6, dtype=torch.float64),
        ],
        dim=-1,
    )


def convert_state_with_step(step: RelativeStateHistoryProcessorStep, state: torch.Tensor) -> torch.Tensor:
    return step(create_transition(observation={OBS_STATE: state}))[TransitionKey.OBSERVATION][OBS_STATE]


def test_se3_state_history_converts_pose_and_gripper_and_keeps_force_and_torque():
    step = RelativeStateHistoryProcessorStep(
        enabled=True, exclude_joints=["force", "torque"], state_names=STATE_NAMES, use_se3_relative_pose=True
    )
    state = build_random_state_history(batch_size=2, history_length=3)
    relative = convert_state_with_step(step, state)

    identity_pose = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float64)
    torch.testing.assert_close(relative[:, -1, :9], identity_pose.expand(2, 9))
    torch.testing.assert_close(
        relative[..., :9], convert_absolute_poses_to_relative_poses(state[..., :9], state[:, -1:, :9])
    )
    torch.testing.assert_close(relative[..., 9], state[..., 9] - state[:, -1:, 9])
    torch.testing.assert_close(relative[..., 10:], state[..., 10:])


def test_state_history_without_se3_is_the_difference_from_the_current_step():
    names = ["joint_a.position", "joint_b.position"]
    step = RelativeStateHistoryProcessorStep(enabled=True, state_names=names)
    state = torch.randn(2, 3, 2)
    torch.testing.assert_close(convert_state_with_step(step, state), state - state[:, -1:])


def test_state_without_a_time_axis_is_returned_unchanged():
    state = torch.randn(2, len(STATE_NAMES), dtype=torch.float64)
    step = RelativeStateHistoryProcessorStep(
        enabled=True, state_names=STATE_NAMES, use_se3_relative_pose=True
    )
    assert convert_state_with_step(step, state) is state


def test_disabled_step_returns_the_state_history_unchanged():
    history = build_random_state_history(batch_size=2, history_length=2)
    assert convert_state_with_step(RelativeStateHistoryProcessorStep(enabled=False), history) is history


def test_se3_state_history_raises_on_state_names_without_a_pose():
    with pytest.raises(ValueError, match="pose channels"):
        RelativeStateHistoryProcessorStep(state_names=["joint_a.position"], use_se3_relative_pose=True)
