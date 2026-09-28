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

from collections.abc import Mapping, Sequence
from typing import Any

from lerobot.utils.constants import ACTION, OBS_STATE

VALID_JOINT_FIELDS = ("position", "velocity", "effort")


def packed_feature_names(joint_names: Sequence[str], fields: Sequence[str]) -> list[str]:
    return [f"{joint}.{field}" for field in fields for joint in joint_names]


def feature_channel_names(feature: Mapping[str, Any]) -> list[str]:
    names = feature.get("names")
    if isinstance(names, dict):
        grouped_names: list[str] = []
        for group in names.values():
            if not isinstance(group, (list, tuple)):
                return list(names)
            grouped_names.extend(group)
        return grouped_names
    if isinstance(names, Sequence) and not isinstance(names, str):
        if names and isinstance(names[0], dict):
            return [name for group in names for name in group.get("motor_names", [])]
        return list(names)
    return []


def channel_groups(names: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for field in VALID_JOINT_FIELDS:
        indices = [index for index, name in enumerate(names) if name.endswith(f".{field}")]
        if indices:
            groups[field] = indices
    if not groups:
        groups["position"] = list(range(len(names)))
    return groups


def io_block_layout(names: Sequence[str]) -> tuple[list[str], list[str]]:
    names = list(names)
    if not names:
        raise ValueError("Cannot derive an IO layout from an empty name list.")
    if not any("." in name for name in names):
        return names, ["position"]

    fields: list[str] = []
    joints_per_field: dict[str, list[str]] = {}
    for name in names:
        joint, _, field = name.rpartition(".")
        if field not in VALID_JOINT_FIELDS or not joint:
            raise ValueError(f"Channel {name!r} is not '<joint>.<field>' with a known field.")
        if field not in fields:
            fields.append(field)
        joints_per_field.setdefault(field, []).append(joint)

    joint_names = joints_per_field[fields[0]]
    if any(joints_per_field[field] != joint_names for field in fields):
        raise ValueError(f"Channels are not one contiguous block per field: {names}")
    if names != packed_feature_names(joint_names, fields):
        raise ValueError(f"Channels are not in block order: {names}")
    return joint_names, fields


def io_layout_from_dataset_features(
    features: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    state_names = feature_channel_names(features[OBS_STATE])
    action_names = feature_channel_names(features[ACTION])
    if not state_names:
        raise KeyError(f"Dataset feature {OBS_STATE!r} has no channel names.")
    if not action_names:
        raise KeyError(f"Dataset feature {ACTION!r} has no channel names.")

    state_joints, state_blocks = io_block_layout(state_names)
    action_joints, action_blocks = io_block_layout(action_names)
    if state_joints != action_joints:
        raise ValueError(f"observation.state joints {state_joints} differ from action joints {action_joints}")
    return {
        "joint_names": state_joints,
        "state_blocks": state_blocks,
        "action_blocks": action_blocks,
    }
