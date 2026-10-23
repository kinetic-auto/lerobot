# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import OBS_IMAGE, OBS_PREFIX, OBS_STATE

# Constants for missing labels.
# Missing labels are left out of the losses the same way padded steps are.
MISSING_STAGE_LABEL = -1
MISSING_MASK_CELL_VALUE = 255


@PreTrainedConfig.register_subclass("kinetic_act")
@dataclass
class KineticACTConfig(ACTConfig):
    """Configuration class for ACT with auxiliary per-timestep heads.

    Args:
        object_distance_label_key: Dataset key of the object distance label in meters.
        object_distance_loss_weight: Weight of the object distance loss.
        stage_label_key: Dataset key of the integer stage label.
        stage_class_count: Number of stages.
        stage_loss_weight: Weight of the stage loss.
        object_mask_label_key_per_camera: Camera key to dataset key of the object mask label.
        object_mask_loss_weight: Weight of the object mask loss.
        object_mask_positive_class_weight: Weight of the object cells in the object mask loss.
        use_camera_identity_embedding: Whether to add a learned embedding per camera.
    """

    object_distance_label_key: str | None = None
    object_distance_loss_weight: float = 1.0
    stage_label_key: str | None = None
    stage_class_count: int = 3
    stage_loss_weight: float = 1.0
    object_mask_label_key_per_camera: dict[str, str] = field(default_factory=dict)
    object_mask_loss_weight: float = 1.0
    object_mask_positive_class_weight: float = 1.0
    use_camera_identity_embedding: bool = False

    @property
    def auxiliary_label_keys(self) -> list[str]:
        label_keys = [self.object_distance_label_key, self.stage_label_key]
        label_keys.extend(self.object_mask_label_key_per_camera.values())
        return [label_key for label_key in label_keys if label_key is not None]

    @property
    def auxiliary_label_delta_indices(self) -> dict[str, list[int]]:
        chunked_label_keys = [self.object_distance_label_key, self.stage_label_key]
        return {
            label_key: list(range(self.chunk_size))
            for label_key in chunked_label_keys
            if label_key is not None
        }

    def set_dataset_feature_metadata(self, dataset_features: dict[str, Any]) -> None:
        missing_label_keys = [
            label_key for label_key in self.auxiliary_label_keys if label_key not in dataset_features
        ]
        if missing_label_keys:
            raise ValueError(
                f"Label keys {missing_label_keys} are not features of the dataset. "
                f"Available features: {sorted(dataset_features)}."
            )

        # Auxillary labels are used for training only and are not inputs to the model.
        for label_key in self.auxiliary_label_keys:
            self.input_features.pop(label_key, None)

    def validate_features(self) -> None:
        super().validate_features()

        # Validate the auxiliary label keys.
        self._validate_label_keys()

        if self.stage_label_key is not None and self.stage_class_count < 2:
            raise ValueError(f"`stage_class_count` must be at least 2. Got {self.stage_class_count}.")

        unknown_camera_keys = [
            camera_key
            for camera_key in self.object_mask_label_key_per_camera
            if camera_key not in self.image_features
        ]
        if unknown_camera_keys:
            raise ValueError(
                f"`object_mask_label_key_per_camera` names cameras {unknown_camera_keys} that are not "
                f"image input features. Image features: {list(self.image_features)}."
            )

        # Validate the loss weights.
        loss_weights = {
            "object_distance_loss_weight": self.object_distance_loss_weight,
            "stage_loss_weight": self.stage_loss_weight,
            "object_mask_loss_weight": self.object_mask_loss_weight,
        }
        for weight_name, weight in loss_weights.items():
            if weight < 0:
                raise ValueError(f"`{weight_name}` must be non-negative. Got {weight}.")

        # Validate the object mask positive class weight.
        if self.object_mask_positive_class_weight <= 0:
            raise ValueError(
                "`object_mask_positive_class_weight` must be positive. "
                f"Got {self.object_mask_positive_class_weight}."
            )

    def _validate_label_keys(self) -> None:
        label_keys = self.auxiliary_label_keys

        duplicate_label_keys = sorted({key for key in label_keys if label_keys.count(key) > 1})
        if duplicate_label_keys:
            raise ValueError(f"Label keys must be distinct. Duplicated: {duplicate_label_keys}.")

        invalid_label_keys = [
            key
            for key in label_keys
            if not key.startswith(OBS_PREFIX) or key == OBS_STATE or key.startswith(OBS_IMAGE)
        ]
        if invalid_label_keys:
            raise ValueError(
                f"Label keys {invalid_label_keys} must start with '{OBS_PREFIX}' and must not be "
                f"'{OBS_STATE}' or start with '{OBS_IMAGE}'."
            )
