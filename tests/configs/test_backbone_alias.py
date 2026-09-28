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

from lerobot.configs.vision_backbones import resolve_vision_backbone
from lerobot.policies.act.configuration_act import ACTConfig


def test_resolve_resnet_backbone():
    choice = resolve_vision_backbone("resnet34")
    assert choice.vision_backbone == "resnet34"
    assert choice.pretrained_backbone_weights == "ResNet34_Weights.IMAGENET1K_V1"
    assert choice.is_dino is False


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("dinov2", "facebook/dinov2-small"),
        ("dinov2_with_registers", "facebook/dinov2-with-registers-small"),
        ("dinov3", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        ("org/custom-dino", "org/custom-dino"),
    ],
)
def test_resolve_dino_backbone(alias, expected):
    choice = resolve_vision_backbone(alias)
    assert choice.vision_backbone == expected
    assert choice.pretrained_backbone_weights == "pretrained"
    assert choice.is_dino is True


def test_act_config_accepts_named_backbone():
    config = ACTConfig(vision_backbone="dinov2", device="cpu")
    assert config.vision_backbone == "facebook/dinov2-small"
    assert config.pretrained_backbone_weights == "pretrained"


def test_resolve_unknown_backbone():
    with pytest.raises(ValueError, match="Unknown backbone"):
        resolve_vision_backbone("mobilenet")
