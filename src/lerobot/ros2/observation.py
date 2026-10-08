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

"""Checkpoint policy state-action layout and observation packing for the ROS 2 policy node.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)

class PolicyStateActionLayout:
    """Joint names and state and action blocks.
    """

    joint_names: tuple[str, ...]
    state_types: tuple[str, ...]
    action_types: tuple[str, ...]

def resolve_checkpoint_dir(checkpoint: str) -> str:
    """Resolve the checkpoint directory.

    Args:
        checkpoint (str): Checkpoint path.

    Returns:
        str: Directory containing config.json.
    """
    # Resolve the checkpoint dir if it starts with ~
    ckpt_dir = os.path.expanduser(str(checkpoint))
    pretrained_dir = os.path.join(ckpt_dir, "pretrained_model")

    if os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        return ckpt_dir
    elif os.path.isfile(os.path.join(pretrained_dir, "config.json")):
        return pretrained_dir
    else:
        raise FileNotFoundError(f"{ckpt_dir} has no config.json or pretrained_model/config.json")

def policy_state_action_layout_from_checkpoint(checkpoint_dir: str) -> PolicyStateActionLayout:
    """Load the checkpoint policy state-action layout.

    Args:
        checkpoint_dir (str): Checkpoint directory.

    Returns:
        PolicyStateActionLayout: Joint names and blocks.
    """
    # Get the checkpoint io layout from the training metadata
    checkpoint_layout = load_checkpoint_metadata(checkpoint_dir).get("io")
    if not isinstance(checkpoint_layout, dict) or not checkpoint_layout:
        raise RuntimeError(f"{checkpoint_dir} training_metadata.json has no io block")

    # Validate the checkpoint io layout
    missing = [
        key
        for key in ("joint_names", "state_blocks", "action_blocks")
        if not checkpoint_layout.get(key)
    ]
    if missing:
        raise ValueError(f"training_metadata.json io block is missing {missing}")

    # Create the policy io layout
    policy_io_layout = PolicyStateActionLayout(
        joint_names=tuple(checkpoint_layout["joint_names"]),
        state_types=tuple(checkpoint_layout["state_blocks"]),
        action_types=tuple(checkpoint_layout["action_blocks"]),
    )
    _validate_policy_state_action_layout(policy_io_layout)

    return policy_io_layout

def load_checkpoint_metadata(checkpoint_dir: str) -> dict:
    """Load checkpoint metadata.

    Args:
        checkpoint_dir (str): Checkpoint directory.

    Returns:
        dict: Metadata.
    """
    training_metadata = os.path.join(checkpoint_dir, "training_metadata.json")

    # Check if the training metadata file exists
    if not os.path.isfile(training_metadata):
        return {}
    
    # Load the training metadata
    with open(training_metadata, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    # Validate the metadata
    if not isinstance(metadata, dict):
        raise ValueError(f"{training_metadata} is not a JSON object")

    return metadata

def _validate_policy_state_action_layout(layout: PolicyStateActionLayout) -> None:
    """Reject duplicate names and unknown blocks."""
    # Check for duplicate joint names
    if len(set(layout.joint_names)) != len(layout.joint_names):
        raise ValueError(f"duplicate joint names {list(layout.joint_names)}")

    # Check for unsupported or duplicate blocks
    allowed = ("position", "velocity", "effort")
    for block_group, blocks in (("state", layout.state_types), ("action", layout.action_types)):
        unknown = [block for block in blocks if block not in allowed]
        if unknown:
            raise ValueError(f"unsupported {block_group} block(s) {unknown}")
        if len(set(blocks)) != len(blocks):
            raise ValueError(f"duplicate {block_group} blocks {list(blocks)}")

def resize_image_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize an image into a target frame.

    Args:
        image (np.ndarray): Source image.
        target_height (int): Target height.
        target_width (int): Target width.

    Returns:
        np.ndarray: Resized image.
    """
    import cv2

    # Return the image if it is already the target size
    original_height, original_width = image.shape[:2]
    if (original_height, original_width) == (target_height, target_width):
        return np.ascontiguousarray(image)

    # Fit the image inside the target frame without stretching it
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = min(target_width, max(1, int(round(original_width * scale))))
    resized_height = min(target_height, max(1, int(round(original_height * scale))))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # Center the resized image on a black frame of the target size
    canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas

def build_observation_state(
    joint_state_values: Mapping[str, Sequence[float]],
    joint_state_types: Sequence[str],
) -> np.ndarray:
    """Build the observation state vector from joint blocks.

    Args:
        joint_state_values (Mapping[str, Sequence[float]]): Joint values for each state type.
        joint_state_types (Sequence[str]): State types from the checkpoint, in order.

    Returns:
        np.ndarray: Observation state vector.
    """
    ordered_joint_state_values = [
        np.asarray(joint_state_values[joint_state_type], dtype=np.float32)
        for joint_state_type in joint_state_types
    ]
    return np.concatenate(ordered_joint_state_values)

def map_joint_names(names: Sequence[str], source_prefix: str, target_prefix: str) -> list[str]:
    """Rename joints by replacing a prefix.

    Args:
        names (Sequence[str]): Joint names.
        source_prefix (str): Prefix to replace.
        target_prefix (str): Replacement prefix.

    Returns:
        list[str]: Renamed joint names.
    """
    mapped = []
    for name in names:
        text = str(name)
        if not text.startswith(source_prefix):
            raise ValueError(f"joint {text!r} does not start with {source_prefix!r}")
        mapped.append(target_prefix + text[len(source_prefix) :])
    return mapped

def build_action(action: np.ndarray, joint_count: int, action_types: Sequence[str]) -> dict[str, np.ndarray]:
    """Build action values by type from a packed action.

    Args:
        action (np.ndarray): Packed action.
        joint_count (int): Joints per type.
        action_types (Sequence[str]): Action type order.

    Returns:
        dict[str, np.ndarray]: Action values by type.
    """
    packed_action = np.asarray(action, dtype=np.float32).reshape(-1)
    expected_action_length = joint_count * len(action_types)

    if packed_action.shape[0] != expected_action_length:
        raise ValueError(
            f"action length {packed_action.shape[0]} does not match {len(action_types)} types of {joint_count}"
        )

    return {
        action_type: packed_action[index * joint_count : (index + 1) * joint_count]
        for index, action_type in enumerate(action_types)
    }
