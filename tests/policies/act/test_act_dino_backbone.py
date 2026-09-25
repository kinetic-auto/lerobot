#!/usr/bin/env python

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

# ruff: noqa: E402

import pytest

pytest.importorskip("transformers")

from pathlib import Path

import torch
from transformers import (
    AutoModel,
    CLIPVisionConfig,
    Dinov2Config,
    Dinov2WithRegistersConfig,
    DINOv3ViTConfig,
)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, resize_to_token_budget
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

TINY_DINO_KWARGS = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
}


def make_act_config(vision_backbone: str, **overrides) -> ACTConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 60, 80)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        "vision_backbone": vision_backbone,
        "pretrained_backbone_weights": None,
        "dim_model": 64,
        "dim_feedforward": 128,
        "n_heads": 2,
        "chunk_size": 8,
        "n_action_steps": 8,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "device": "cpu",
        "push_to_hub": False,
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def save_tiny_dino_config(directory: Path, config_cls, **kwargs) -> str:
    config_cls(**TINY_DINO_KWARGS, **kwargs).save_pretrained(directory)
    return str(directory)


def make_train_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, 6),
        f"{OBS_IMAGES}.front": torch.randn(batch_size, 3, 60, 80),
        ACTION: torch.randn(batch_size, 8, 6),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("config_cls", "config_kwargs", "expected_hw"),
    [
        (Dinov2Config, {"patch_size": 14}, (4, 5)),
        (Dinov2WithRegistersConfig, {"patch_size": 14, "num_register_tokens": 4}, (4, 5)),
        (DINOv3ViTConfig, {"patch_size": 16, "num_register_tokens": 4}, (3, 5)),
    ],
)
def test_dino_backbone_feature_map_shape(tmp_path, config_cls, config_kwargs, expected_hw):
    backbone_path = save_tiny_dino_config(tmp_path / "dino-tiny", config_cls, **config_kwargs)
    policy = ACTPolicy(make_act_config(backbone_path, backbone_token_budget=None))
    images = torch.randn(2, 3, 60, 80)

    feature_map = policy.model.backbone(images)["feature_map"]

    assert feature_map.shape == (2, 32, *expected_hw)
    assert policy.model.encoder_img_feat_input_proj.in_channels == 32


def test_dino_backbone_train_and_select_action(tmp_path):
    backbone_path = save_tiny_dino_config(tmp_path / "dinov2-tiny", Dinov2Config, patch_size=14)
    policy = ACTPolicy(make_act_config(backbone_path))
    policy.train()

    loss, _ = policy.forward(make_train_batch())
    loss.backward()

    action = policy.select_action(
        {
            OBS_STATE: torch.randn(2, 6),
            f"{OBS_IMAGES}.front": torch.randn(2, 3, 60, 80),
        }
    )
    assert action.shape == (2, 6)


def test_resize_to_token_budget():
    images_480x640 = torch.randn(1, 3, 480, 640)
    assert resize_to_token_budget(images_480x640, 256, 14).shape[-2:] == (196, 252)
    assert resize_to_token_budget(images_480x640, 196, 16).shape[-2:] == (192, 256)
    assert resize_to_token_budget(images_480x640, 300, 14).shape[-2:] == (210, 280)

    images_224 = torch.randn(1, 3, 224, 224)
    assert resize_to_token_budget(images_224, 256, 14) is images_224
    with pytest.raises(ValueError, match="token_budget"):
        resize_to_token_budget(images_224, 0, 14)


def test_dino_backbone_token_budget_feature_map(tmp_path):
    backbone_path = save_tiny_dino_config(tmp_path / "dinov2-tiny", Dinov2Config, patch_size=14)
    policy = ACTPolicy(make_act_config(backbone_path, backbone_token_budget=4))
    images = torch.randn(2, 3, 60, 80)

    feature_map = policy.model.backbone(images)["feature_map"]

    assert feature_map.shape == (2, 32, 2, 2)


def test_dino_backbone_loads_pretrained_weights_from_local_dir(tmp_path):
    backbone_dir = tmp_path / "dinov2-tiny"
    backbone_config = Dinov2Config(**TINY_DINO_KWARGS, patch_size=14)
    saved_model = AutoModel.from_config(backbone_config)
    saved_model.save_pretrained(backbone_dir)

    policy = ACTPolicy(make_act_config(str(backbone_dir), pretrained_backbone_weights="pretrained"))
    parameter_name, expected = next(saved_model.named_parameters())
    torch.testing.assert_close(dict(policy.model.backbone.model.named_parameters())[parameter_name], expected)


def test_freeze_vision_backbone(tmp_path):
    backbone_path = save_tiny_dino_config(tmp_path / "dinov2-tiny", Dinov2Config, patch_size=14)
    config = make_act_config(backbone_path, freeze_vision_backbone=True)
    policy = ACTPolicy(config)

    backbone_parameters = [
        parameter for name, parameter in policy.named_parameters() if name.startswith("model.backbone")
    ]
    assert backbone_parameters
    assert all(not parameter.requires_grad for parameter in backbone_parameters)
    groups = policy.get_optim_params()
    assert groups[1]["params"] == []
    config.get_optimizer_preset().build(groups)

    policy.train()
    assert policy.model.backbone.training is False
    assert policy.model.encoder.training is True


def test_dino_config_validation(tmp_path):
    with pytest.raises(ValueError, match="replace_final_stride_with_dilation"):
        make_act_config("facebook/dinov2-small", replace_final_stride_with_dilation=True)
    with pytest.raises(ValueError, match="backbone_token_budget"):
        make_act_config("facebook/dinov2-small", backbone_token_budget=0)
    with pytest.raises(ValueError, match="vision_backbone"):
        make_act_config("openai/clip-vit-base-patch16")
    assert make_act_config("resnet18").backbone_token_budget == 300

    fake_dir = tmp_path / "dino-fake"
    CLIPVisionConfig().save_pretrained(fake_dir)
    with pytest.raises(ValueError, match="model_type"):
        ACTPolicy(make_act_config(str(fake_dir), backbone_token_budget=None))
