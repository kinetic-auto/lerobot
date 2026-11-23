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

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.model.cartesian_quantities import FramePoseRot6D

# Pose vector length (3 translation + 6 rotation)
POSE_VECTOR_LENGTH = 9


def convert_absolute_poses_to_relative_poses(
    absolute_pose_vectors: Tensor, reference_pose_vectors: Tensor
) -> Tensor:
    """Convert absolute poses to relative poses with respect to a reference pose.

    Args:
        absolute_pose_vectors (Tensor): Poses in the base frame.
        reference_pose_vectors (Tensor): Reference poses in the base frame, one per sample.

    Returns:
        Tensor: Poses relative to the reference pose.
    """
    absolute_pose_matrices = pose_vectors_to_matrices(absolute_pose_vectors)
    reference_pose_matrices = pose_vectors_to_matrices(reference_pose_vectors)
    return matrices_to_pose_vectors(
        _invert_rigid_transforms(reference_pose_matrices) @ absolute_pose_matrices
    )


def convert_relative_poses_to_absolute_poses(
    relative_pose_vectors: Tensor, reference_pose_vectors: Tensor
) -> Tensor:
    """Convert relative poses to absolute poses in the base frame.

    Args:
        relative_pose_vectors (Tensor): Poses relative to the reference pose.
        reference_pose_vectors (Tensor): Reference poses in the base frame, one per sample.

    Returns:
        Tensor: Poses in the base frame.
    """
    relative_pose_matrices = pose_vectors_to_matrices(relative_pose_vectors)
    reference_pose_matrices = pose_vectors_to_matrices(reference_pose_vectors)
    return matrices_to_pose_vectors(reference_pose_matrices @ relative_pose_matrices)


def find_pose_dimension_indices(feature_names: Sequence[str]) -> list[int] | None:
    """Find the dimension indices of a pose in a list of feature names.

    Args:
        feature_names (Sequence[str]): List of feature names.

    Raises:
        ValueError: If the feature names do not hold a complete pose.

    Returns:
        list[int] | None: List of dimension indices, or None if the feature names do not hold a complete pose.
    """
    required_pose_feature_names = FramePoseRot6D.feature_names()
    found_pose_feature_names = [
        pose_feature_name
        for pose_feature_name in required_pose_feature_names
        if pose_feature_name in feature_names
    ]
    if not found_pose_feature_names:
        return None
    if len(found_pose_feature_names) != len(required_pose_feature_names):
        missing_pose_feature_names = [
            pose_feature_name
            for pose_feature_name in required_pose_feature_names
            if pose_feature_name not in feature_names
        ]
        raise ValueError(f"Feature names hold part of a pose; missing {missing_pose_feature_names}.")
    return [list(feature_names).index(pose_feature_name) for pose_feature_name in required_pose_feature_names]


def pose_vectors_to_matrices(pose_vectors: Tensor) -> Tensor:
    """Convert pose vectors to matrices.

    Args:
        pose_vectors (Tensor): Pose vectors, shape (..., 9).

    Returns:
        Tensor: Transformation matrices, shape (..., 4, 4).
    """
    if pose_vectors.shape[-1] != POSE_VECTOR_LENGTH:
        raise ValueError(
            f"Pose vectors need {POSE_VECTOR_LENGTH} values, got shape {tuple(pose_vectors.shape)}."
        )

    pose_matrices = torch.zeros(
        *pose_vectors.shape[:-1], 4, 4, dtype=pose_vectors.dtype, device=pose_vectors.device
    )
    pose_matrices[..., :3, :3] = convert_rotation_6d_to_matrix(pose_vectors[..., 3:])
    pose_matrices[..., :3, 3] = pose_vectors[..., :3]
    pose_matrices[..., 3, 3] = 1.0
    return pose_matrices


def matrices_to_pose_vectors(matrices: Tensor) -> Tensor:
    """Convert pose matrices to pose vectors.

    Args:
        matrices (Tensor): Transformation matrices, shape (..., 4, 4).

    Returns:
        Tensor: Pose vectors, shape (..., 9).
    """
    return torch.cat([matrices[..., :3, 3], convert_matrix_to_rotation_6d(matrices[..., :3, :3])], dim=-1)


def convert_rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """Convert 6D rotation vectors to rotation matrices.

    Args:
        rotation_6d (Tensor): 6D rotation vectors, shape (..., 6).

    Returns:
        Tensor: Rotation matrices, shape (..., 3, 3).
    """
    # Gram-Schmidt on the two stored columns; the third column is their cross product.
    first_column = F.normalize(rotation_6d[..., :3], dim=-1)
    second_column = rotation_6d[..., 3:]
    second_column = F.normalize(
        second_column - (first_column * second_column).sum(dim=-1, keepdim=True) * first_column, dim=-1
    )
    third_column = torch.cross(first_column, second_column, dim=-1)
    return torch.stack([first_column, second_column, third_column], dim=-1)


def convert_matrix_to_rotation_6d(rotation_matrices: Tensor) -> Tensor:
    """Convert rotation matrices to 6D rotation vectors.

    Args:
        rotation_matrices (Tensor): Rotation matrices, shape (..., 3, 3).

    Returns:
        Tensor: 6D rotation vectors, shape (..., 6).
    """
    return torch.cat([rotation_matrices[..., :, 0], rotation_matrices[..., :, 1]], dim=-1)


def _invert_rigid_transforms(matrices: Tensor) -> Tensor:
    """Invert rigid transformation matrices.

    Args:
        matrices (Tensor): Transformation matrices, shape (..., 4, 4).

    Returns:
        Tensor: Inverted transformation matrices, shape (..., 4, 4).
    """
    inverse_rotations = matrices[..., :3, :3].transpose(-1, -2)
    inverses = torch.zeros_like(matrices)
    inverses[..., :3, :3] = inverse_rotations
    inverses[..., :3, 3] = -(inverse_rotations @ matrices[..., :3, 3:4]).squeeze(-1)
    inverses[..., 3, 3] = 1.0
    return inverses
