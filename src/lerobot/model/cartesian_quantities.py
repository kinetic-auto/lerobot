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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np


class CartesianQuantity(ABC):
    entity_type: str
    channel_names: Any

    @abstractmethod
    def to_vector(self) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_vector(cls, values: np.ndarray) -> Any:
        raise NotImplementedError

    @classmethod
    def feature_names(cls) -> list[str]:
        return list(cls.channel_names)


@dataclass(frozen=True, eq=False)
class FramePoseRot6D(CartesianQuantity):
    translation: np.ndarray
    rotation: np.ndarray
    entity_type: str = "position"
    channel_names: Any = (
        "ee.position.x",
        "ee.position.y",
        "ee.position.z",
        "ee.orientation.rot6d.0",
        "ee.orientation.rot6d.1",
        "ee.orientation.rot6d.2",
        "ee.orientation.rot6d.3",
        "ee.orientation.rot6d.4",
        "ee.orientation.rot6d.5",
    )

    def to_vector(self) -> np.ndarray:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        rot_vec_6d = np.concatenate([rotation[:, 0], rotation[:, 1]])
        return np.concatenate([np.asarray(self.translation, dtype=np.float64).reshape(3), rot_vec_6d])

    @classmethod
    def from_vector(cls, values: np.ndarray) -> Self:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (9,):
            raise ValueError(f"Pose vector must have 9 values, got shape {values.shape}.")

        # Extract the translation
        translation = values[:3]

        # Extract the rotation vector in 6D format
        # Extract the first rotation column
        column_0 = values[3:6]
        norm_0 = np.linalg.norm(column_0)
        if norm_0 == 0:
            raise ValueError("First rotation column has zero length.")
        column_0 = column_0 / norm_0

        # Extract the second rotation column
        column_1 = values[6:9]
        column_1 = column_1 - np.dot(column_0, column_1) * column_0
        norm_1 = np.linalg.norm(column_1)
        if norm_1 == 0:
            raise ValueError("Second rotation column is parallel to the first.")
        column_1 = column_1 / norm_1

        # Extract the third rotation column
        column_2 = np.cross(column_0, column_1)

        # Stack the rotation columns into a rotation matrix
        rotation = np.column_stack([column_0, column_1, column_2])

        return cls(translation=translation, rotation=rotation)


@dataclass(frozen=True, eq=False)
class FrameTwist(CartesianQuantity):
    linear: np.ndarray
    angular: np.ndarray
    entity_type: str = "velocity"
    channel_names: Any = (
        "ee.linear.x",
        "ee.linear.y",
        "ee.linear.z",
        "ee.angular.x",
        "ee.angular.y",
        "ee.angular.z",
    )

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.linear, dtype=np.float64).reshape(3),
                np.asarray(self.angular, dtype=np.float64).reshape(3),
            ]
        )

    @classmethod
    def from_vector(cls, values: np.ndarray) -> Self:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (6,):
            raise ValueError(f"Twist vector must have 6 values, got shape {values.shape}.")
        return cls(linear=values[:3], angular=values[3:])


@dataclass(frozen=True, eq=False)
class FrameWrench(CartesianQuantity):
    force: np.ndarray
    torque: np.ndarray
    entity_type: str = "effort"
    channel_names: Any = (
        "ee.force.x",
        "ee.force.y",
        "ee.force.z",
        "ee.torque.x",
        "ee.torque.y",
        "ee.torque.z",
    )

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.force, dtype=np.float64).reshape(3),
                np.asarray(self.torque, dtype=np.float64).reshape(3),
            ]
        )

    @classmethod
    def from_vector(cls, values: np.ndarray) -> Self:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (6,):
            raise ValueError(f"Wrench vector must have 6 values, got shape {values.shape}.")
        return cls(force=values[:3], torque=values[3:])


@dataclass(frozen=True, eq=False)
class GripperChannels(CartesianQuantity):
    names: Any
    values: np.ndarray
    entity_type: str
    channel_names: Any = ()

    def to_vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64).reshape(-1)

    @classmethod
    def from_vector(cls, values: np.ndarray) -> Self:
        raise ValueError("GripperChannels needs channel names; use the constructor.")

    def channel_feature_names(self) -> list[str]:
        return list(self.names)


_GRIPPER_AXES = ("x", "y", "z")


def gripper_position_channel_names(count: int) -> list[str]:
    names: list[str] = []
    for index in range(count):
        axis = _GRIPPER_AXES[index] if index < len(_GRIPPER_AXES) else str(index)
        names.append(f"gripper.position.{axis}")
    return names


@dataclass(frozen=True, eq=False)
class CartesianFrameSample:
    quantities: Any
    fields: Any

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for field_name in self.fields:
            for quantity in self.quantities:
                if _entity_type(quantity) != field_name:
                    continue
                names.extend(_quantity_feature_names(quantity))
        return names

    def to_vector(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for field_name in self.fields:
            for quantity in self.quantities:
                if _entity_type(quantity) != field_name:
                    continue
                parts.append(np.asarray(quantity.to_vector(), dtype=np.float64).reshape(-1))
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)


def _entity_type(quantity: CartesianQuantity) -> str:
    return quantity.entity_type


def _quantity_feature_names(quantity: CartesianQuantity) -> list[str]:
    if isinstance(quantity, GripperChannels):
        return quantity.channel_feature_names()
    return type(quantity).feature_names()
