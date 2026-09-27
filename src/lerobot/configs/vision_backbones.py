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

from enum import Enum


class VisionBackbone(Enum):
    @property
    def backbone_name(self) -> str:
        return self.value[0]

    @property
    def vision_backbone(self) -> str:
        return self.value[1]

    @property
    def pretrained_backbone_weights(self) -> str:
        return self.value[2]

    @property
    def is_dino(self) -> bool:
        return isinstance(self, DinoBackbone)


class ResNetBackbone(VisionBackbone):
    RESNET18 = ("resnet18", "resnet18", "ResNet18_Weights.IMAGENET1K_V1")
    RESNET34 = ("resnet34", "resnet34", "ResNet34_Weights.IMAGENET1K_V1")
    RESNET50 = ("resnet50", "resnet50", "ResNet50_Weights.IMAGENET1K_V1")


class DinoBackbone(VisionBackbone):
    DINOV2 = ("dinov2", "facebook/dinov2-small", "pretrained")
    DINOV2_WITH_REGISTERS = ("dinov2_with_registers", "facebook/dinov2-with-registers-small", "pretrained")
    DINOV3 = ("dinov3", "facebook/dinov3-vits16-pretrain-lvd1689m", "pretrained")
    DINOV3_VIT = ("dinov3_vit", "facebook/dinov3-vits16-pretrain-lvd1689m", "pretrained")


def lookup_vision_backbone(backbone: str) -> VisionBackbone | None:
    backbone_name = backbone.lower()
    for candidate in (*ResNetBackbone, *DinoBackbone):
        if candidate.backbone_name == backbone_name:
            return candidate
    return None


def resolve_vision_backbone(backbone: str) -> VisionBackbone:
    backbone_name = backbone.lower()

    # Resolve ResNet backbones.
    resnet = next(
        (candidate for candidate in ResNetBackbone if candidate.backbone_name == backbone_name),
        None,
    )
    if resnet is not None:
        return resnet

    # Resolve DINO backbones.
    dino = next((candidate for candidate in DinoBackbone if candidate.backbone_name == backbone_name), None)
    if dino is not None:
        return dino

    # Resolve custom DINO backbones.
    if "dino" in backbone_name:
        custom = object.__new__(DinoBackbone)
        custom._name_ = backbone
        custom._value_ = (backbone, backbone, "pretrained")
        return custom

    available_backbones = ", ".join(sorted(candidate.backbone_name for candidate in (*ResNetBackbone, *DinoBackbone)))
    raise ValueError(f"Unknown backbone {backbone!r}. Expected one of {available_backbones}.")
