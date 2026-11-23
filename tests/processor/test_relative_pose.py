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

import numpy as np
import pytest
import torch

from lerobot.model.cartesian_quantities import FramePoseRot6D, FrameWrench
from lerobot.processor.relative_pose import (
    convert_absolute_poses_to_relative_poses,
    convert_matrix_to_rotation_6d,
    convert_relative_poses_to_absolute_poses,
    convert_rotation_6d_to_matrix,
    find_pose_dimension_indices,
    pose_vectors_to_matrices,
)


def rotation_matrix_about_z_axis(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix_about_x_axis(angle: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )


def build_pose_vector(translation: list[float], rotation: np.ndarray) -> torch.Tensor:
    vector = FramePoseRot6D(translation=np.asarray(translation), rotation=rotation).to_vector()
    return torch.tensor(vector, dtype=torch.float64)


def test_rotation_6d_conversions_match_the_frame_pose_rot6d_layout():
    rotation = rotation_matrix_about_z_axis(0.3) @ rotation_matrix_about_x_axis(-0.7)
    vector = build_pose_vector([1.0, 2.0, 3.0], rotation)
    np.testing.assert_allclose(convert_rotation_6d_to_matrix(vector[3:]).numpy(), rotation, atol=1e-12)
    np.testing.assert_allclose(
        convert_matrix_to_rotation_6d(torch.tensor(rotation)).numpy(), vector[3:].numpy(), atol=1e-12
    )


def test_non_orthonormal_rotation_6d_converts_to_a_valid_rotation_matrix():
    rotation_6d = torch.tensor([2.0, 0.1, 0.0, 0.3, 1.5, 0.0], dtype=torch.float64)
    rotation = convert_rotation_6d_to_matrix(rotation_6d)
    np.testing.assert_allclose((rotation.T @ rotation).numpy(), np.eye(3), atol=1e-12)
    assert torch.linalg.det(rotation).item() == pytest.approx(1.0)


def test_converting_poses_to_relative_and_back_returns_the_original_poses():
    reference = build_pose_vector(
        [0.5, -0.2, 0.9], rotation_matrix_about_z_axis(0.4) @ rotation_matrix_about_x_axis(0.2)
    )
    poses = torch.stack(
        [
            build_pose_vector([0.6, -0.1, 0.8], rotation_matrix_about_z_axis(0.5)),
            build_pose_vector([0.7, 0.0, 0.7], rotation_matrix_about_x_axis(-0.3)),
        ]
    )
    relative = convert_absolute_poses_to_relative_poses(poses, reference.unsqueeze(0))
    restored = convert_relative_poses_to_absolute_poses(relative, reference.unsqueeze(0))
    np.testing.assert_allclose(restored.numpy(), poses.numpy(), atol=1e-12)


def test_reference_pose_converts_to_the_identity_relative_pose():
    reference = build_pose_vector([0.5, -0.2, 0.9], rotation_matrix_about_z_axis(0.4))
    relative = convert_absolute_poses_to_relative_poses(reference, reference)
    np.testing.assert_allclose(relative.numpy(), [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-12)


def test_relative_pose_is_unchanged_when_the_base_frame_moves():
    reference = build_pose_vector([0.5, -0.2, 0.9], rotation_matrix_about_x_axis(0.2))
    pose = build_pose_vector([0.6, 0.1, 0.8], rotation_matrix_about_z_axis(0.5))
    base_rotation = torch.tensor(rotation_matrix_about_z_axis(np.pi / 2))
    rotated_base = torch.eye(4, dtype=torch.float64)
    rotated_base[:3, :3] = base_rotation
    rotated_base[:3, 3] = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

    def express_pose_in_moved_base_frame(vector: torch.Tensor) -> torch.Tensor:
        matrix = rotated_base @ pose_vectors_to_matrices(vector)
        return torch.cat([matrix[:3, 3], convert_matrix_to_rotation_6d(matrix[:3, :3])])

    relative_in_base = convert_absolute_poses_to_relative_poses(pose, reference)
    relative_in_moved_base = convert_absolute_poses_to_relative_poses(
        express_pose_in_moved_base_frame(pose), express_pose_in_moved_base_frame(reference)
    )
    np.testing.assert_allclose(relative_in_moved_base.numpy(), relative_in_base.numpy(), atol=1e-12)


def test_find_pose_dimension_indices_handles_full_missing_and_partial_poses():
    names = ["gripper.position.x", *FramePoseRot6D.feature_names(), *FrameWrench.feature_names()]
    assert find_pose_dimension_indices(names) == list(range(1, 10))
    assert find_pose_dimension_indices(["joint_a.position", "joint_b.position"]) is None
    with pytest.raises(ValueError, match="ee.orientation.rot6d.5"):
        find_pose_dimension_indices(FramePoseRot6D.feature_names()[:-1])
