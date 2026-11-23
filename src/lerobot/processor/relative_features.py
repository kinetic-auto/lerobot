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
from torch import Tensor

from .relative_pose import (
    convert_absolute_poses_to_relative_poses,
    convert_relative_poses_to_absolute_poses,
)


def build_relative_mask(
    feature_names: Sequence[str] | None, exclude_joints: Sequence[str], dimension: int
) -> list[bool]:
    """Build the mask of dimensions to convert to relative features.

    Args:
        feature_names (Sequence[str] | None): Feature names, one per dimension.
        exclude_joints (Sequence[str]): Name patterns of the dimensions to keep absolute.
        dimension (int): Number of dimensions.

    Returns:
        list[bool]: Mask of the relative dimensions.
    """
    if not exclude_joints or feature_names is None:
        return [True] * dimension

    excluded_joint_names = [str(joint_name).lower() for joint_name in exclude_joints if joint_name]
    if not excluded_joint_names:
        return [True] * dimension

    relative_mask = []
    for feature_name in feature_names[:dimension]:
        normalized_feature_name = str(feature_name).lower()
        is_excluded = any(
            excluded_joint_name == normalized_feature_name or excluded_joint_name in normalized_feature_name
            for excluded_joint_name in excluded_joint_names
        )
        relative_mask.append(not is_excluded)

    if len(relative_mask) < dimension:
        relative_mask.extend([True] * (dimension - len(relative_mask)))

    return relative_mask


def to_se3_relative_features(
    features: Tensor, reference_state: Tensor, mask: Sequence[bool], pose_indices: Sequence[int]
) -> Tensor:
    """Convert absolute features to relative features, using SE(3) for the pose dimensions.

    Args:
        features (Tensor): Absolute features, shape (..., D).
        reference_state (Tensor): Reference state, one per sample.
        mask (Sequence[bool]): Mask of the relative dimensions.
        pose_indices (Sequence[int]): Dimension indices of the pose.

    Returns:
        Tensor: Features relative to the reference state, shape (..., D).
    """
    relative_features = to_relative_features(
        features, reference_state, _mask_without_pose(mask, pose_indices)
    )
    relative_features[..., pose_indices] = convert_absolute_poses_to_relative_poses(
        features[..., pose_indices], _reference_pose_vectors(features, reference_state, pose_indices)
    )
    return relative_features


def to_se3_absolute_features(
    features: Tensor, reference_state: Tensor, mask: Sequence[bool], pose_indices: Sequence[int]
) -> Tensor:
    """Convert relative features to absolute features, using SE(3) for the pose dimensions.

    Args:
        features (Tensor): Features relative to the reference state, shape (..., D).
        reference_state (Tensor): Reference state, one per sample.
        mask (Sequence[bool]): Mask of the relative dimensions.
        pose_indices (Sequence[int]): Dimension indices of the pose.

    Returns:
        Tensor: Absolute features, shape (..., D).
    """
    absolute_features = to_absolute_features(
        features, reference_state, _mask_without_pose(mask, pose_indices)
    )
    absolute_features[..., pose_indices] = convert_relative_poses_to_absolute_poses(
        features[..., pose_indices], _reference_pose_vectors(features, reference_state, pose_indices)
    )
    return absolute_features


def to_relative_features(features: Tensor, reference_state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute features to relative features by subtracting the reference state.

    Args:
        features (Tensor): Absolute features, shape (..., D).
        reference_state (Tensor): Reference state, one per sample.
        mask (Sequence[bool]): Mask of the relative dimensions.

    Returns:
        Tensor: Features relative to the reference state, shape (..., D).
    """
    reference_offset, dimension_count = _masked_reference_offset(features, reference_state, mask)
    relative_features = features.clone()
    relative_features[..., :dimension_count] -= reference_offset
    return relative_features


def to_absolute_features(features: Tensor, reference_state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative features to absolute features by adding the reference state.

    Args:
        features (Tensor): Features relative to the reference state, shape (..., D).
        reference_state (Tensor): Reference state, one per sample.
        mask (Sequence[bool]): Mask of the relative dimensions.

    Returns:
        Tensor: Absolute features, shape (..., D).
    """
    reference_offset, dimension_count = _masked_reference_offset(features, reference_state, mask)
    absolute_features = features.clone()
    absolute_features[..., :dimension_count] += reference_offset
    return absolute_features


def _masked_reference_offset(
    features: Tensor, reference_state: Tensor, mask: Sequence[bool]
) -> tuple[Tensor, int]:
    """Compute the reference state offset of the relative dimensions.

    Args:
        features (Tensor): Features, shape (..., D).
        reference_state (Tensor): Reference state, one per sample.
        mask (Sequence[bool]): Mask of the relative dimensions.

    Returns:
        tuple[Tensor, int]: Offset and the number of dimensions it covers.
    """
    reference_state = reference_state.to(device=features.device, dtype=features.dtype)
    dimension_count = min(features.shape[-1], reference_state.shape[-1], len(mask))
    if any(mask[dimension_count : min(features.shape[-1], len(mask))]):
        raise ValueError(
            f"Relative feature dimensions cannot exceed the reference state width {reference_state.shape[-1]}."
        )
    mask_tensor = torch.tensor(mask[:dimension_count], dtype=features.dtype, device=features.device)
    reference_offset = reference_state[..., :dimension_count] * mask_tensor
    if features.ndim == 3:
        reference_offset = reference_offset.unsqueeze(-2)
    return reference_offset, dimension_count


def _mask_without_pose(mask: Sequence[bool], pose_indices: Sequence[int]) -> list[bool]:
    """Remove the pose dimensions from a mask.

    Args:
        mask (Sequence[bool]): Mask of the relative dimensions.
        pose_indices (Sequence[int]): Dimension indices of the pose.

    Returns:
        list[bool]: Mask of the relative dimensions without the pose.
    """
    return [
        is_relative_dimension and index not in pose_indices
        for index, is_relative_dimension in enumerate(mask)
    ]


def _reference_pose_vectors(features: Tensor, reference_state: Tensor, pose_indices: Sequence[int]) -> Tensor:
    """Select the reference pose vectors from the reference state.

    Args:
        features (Tensor): Features, shape (..., D).
        reference_state (Tensor): Reference state, one per sample.
        pose_indices (Sequence[int]): Dimension indices of the pose.

    Returns:
        Tensor: Reference pose vectors, shape (..., 9).
    """
    reference_poses = reference_state[..., pose_indices].to(device=features.device, dtype=features.dtype)
    if features.ndim == 3:
        reference_poses = reference_poses.unsqueeze(-2)
    return reference_poses
