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

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.kinetic_act.configuration_kinetic_act import (
    MISSING_MASK_CELL_VALUE,
    MISSING_STAGE_LABEL,
    KineticACTConfig,
)
from lerobot.policies.kinetic_act.modeling_kinetic_act import KineticACT, KineticACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CHUNK_SIZE = 8
STATE_DIM = 6
ACTION_DIM = 6
STAGE_CLASS_COUNT = 3
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 96
# ResNet downsamples by 32, so the token grid is 2x3 and the mask label (4x6) is pooled onto it.
TOKEN_GRID_SHAPE = (2, 3)
MASK_LABEL_SHAPE = (4, 6)

RGB_CAMERA_KEY = f"{OBS_IMAGES}.front"
DEPTH_CAMERA_KEY = f"{OBS_IMAGES}.depth"
OBJECT_DISTANCE_LABEL_KEY = "observation.object_distance"
STAGE_LABEL_KEY = "observation.stage"
OBJECT_MASK_LABEL_KEY = "observation.object_mask.front"

ALL_HEAD_OVERRIDES = {
    "object_distance_label_key": OBJECT_DISTANCE_LABEL_KEY,
    "stage_label_key": STAGE_LABEL_KEY,
    "stage_class_count": STAGE_CLASS_COUNT,
    "object_mask_label_key_per_camera": {RGB_CAMERA_KEY: OBJECT_MASK_LABEL_KEY},
    "use_camera_identity_embedding": True,
}


def make_input_features(camera_keys: tuple[str, ...]) -> dict[str, PolicyFeature]:
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
    for camera_key in camera_keys:
        channel_count = 1 if camera_key == DEPTH_CAMERA_KEY else 3
        input_features[camera_key] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(channel_count, IMAGE_HEIGHT, IMAGE_WIDTH)
        )
    return input_features


def make_config_kwargs(camera_keys: tuple[str, ...]) -> dict:
    return {
        "input_features": make_input_features(camera_keys),
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "pretrained_backbone_weights": None,
        "dim_model": 64,
        "dim_feedforward": 128,
        "n_heads": 2,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": CHUNK_SIZE,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "device": "cpu",
        "push_to_hub": False,
    }


def make_kinetic_act_config(
    camera_keys: tuple[str, ...] = (RGB_CAMERA_KEY, DEPTH_CAMERA_KEY), **overrides
) -> KineticACTConfig:
    return KineticACTConfig(**{**make_config_kwargs(camera_keys), **overrides})


def make_train_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, STATE_DIM),
        RGB_CAMERA_KEY: torch.randn(batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
        DEPTH_CAMERA_KEY: torch.randn(batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH),
        ACTION: torch.randn(batch_size, CHUNK_SIZE, ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, CHUNK_SIZE, dtype=torch.bool),
        OBJECT_DISTANCE_LABEL_KEY: torch.rand(batch_size, CHUNK_SIZE),
        f"{OBJECT_DISTANCE_LABEL_KEY}_is_pad": torch.zeros(batch_size, CHUNK_SIZE, dtype=torch.bool),
        STAGE_LABEL_KEY: torch.randint(0, STAGE_CLASS_COUNT, (batch_size, CHUNK_SIZE)),
        f"{STAGE_LABEL_KEY}_is_pad": torch.zeros(batch_size, CHUNK_SIZE, dtype=torch.bool),
        OBJECT_MASK_LABEL_KEY: torch.randint(0, 2, (batch_size, *MASK_LABEL_SHAPE), dtype=torch.uint8),
    }


def make_inference_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, STATE_DIM),
        RGB_CAMERA_KEY: torch.randn(batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
        DEPTH_CAMERA_KEY: torch.randn(batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH),
    }


def parameter_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(parameter.shape) for name, parameter in module.named_parameters()}


def test_default_parameters_match_act():
    act_config = ACTConfig(**make_config_kwargs((RGB_CAMERA_KEY, DEPTH_CAMERA_KEY)))
    kinetic_act = KineticACT(make_kinetic_act_config())

    assert parameter_shapes(kinetic_act) == parameter_shapes(ACT(act_config))


def test_enabled_heads_add_only_their_own_parameters():
    default_shapes = parameter_shapes(KineticACT(make_kinetic_act_config()))
    all_heads_shapes = parameter_shapes(KineticACT(make_kinetic_act_config(**ALL_HEAD_OVERRIDES)))

    added_shapes = {name: shape for name, shape in all_heads_shapes.items() if name not in default_shapes}
    assert added_shapes == {
        "object_distance_head.weight": (1, 64),
        "object_distance_head.bias": (1,),
        "stage_head.weight": (STAGE_CLASS_COUNT, 64),
        "stage_head.bias": (STAGE_CLASS_COUNT,),
        "object_mask_head.weight": (1, 64),
        "object_mask_head.bias": (1,),
        "encoder_camera_identity_embedding.weight": (2, 64),
    }
    assert {name: all_heads_shapes[name] for name in default_shapes} == default_shapes


def test_registration_resolves_config_policy_and_processors():
    config = make_policy_config("kinetic_act", **make_config_kwargs((RGB_CAMERA_KEY, DEPTH_CAMERA_KEY)))

    assert config.type == "kinetic_act"
    assert get_policy_class("kinetic_act") is KineticACTPolicy
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    assert preprocessor is not None
    assert postprocessor is not None


def test_set_dataset_feature_metadata_drops_label_keys_from_inputs():
    config = make_kinetic_act_config(**ALL_HEAD_OVERRIDES)
    config.input_features[OBJECT_DISTANCE_LABEL_KEY] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
    config.input_features[STAGE_LABEL_KEY] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
    config.input_features[OBJECT_MASK_LABEL_KEY] = PolicyFeature(
        type=FeatureType.STATE, shape=MASK_LABEL_SHAPE
    )
    dataset_features = dict.fromkeys(config.input_features) | {ACTION: None}

    config.set_dataset_feature_metadata(dataset_features)

    assert set(config.input_features) == {OBS_STATE, RGB_CAMERA_KEY, DEPTH_CAMERA_KEY}


def test_set_dataset_feature_metadata_raises_on_missing_label_key():
    config = make_kinetic_act_config(**ALL_HEAD_OVERRIDES)

    with pytest.raises(ValueError, match="observation.stage"):
        config.set_dataset_feature_metadata({OBS_STATE: None, OBJECT_DISTANCE_LABEL_KEY: None})


def test_auxiliary_label_delta_indices_are_chunked():
    pytest.importorskip("datasets")
    from lerobot.datasets.factory import resolve_delta_timestamps

    config = make_kinetic_act_config(**ALL_HEAD_OVERRIDES)
    metadata = SimpleNamespace(
        fps=10,
        features=dict.fromkeys(
            [
                OBS_STATE,
                RGB_CAMERA_KEY,
                OBJECT_DISTANCE_LABEL_KEY,
                STAGE_LABEL_KEY,
                OBJECT_MASK_LABEL_KEY,
                ACTION,
            ]
        ),
    )

    delta_timestamps = resolve_delta_timestamps(config, metadata)

    expected_chunk = [index / 10 for index in range(CHUNK_SIZE)]
    assert delta_timestamps[ACTION] == expected_chunk
    assert delta_timestamps[OBJECT_DISTANCE_LABEL_KEY] == expected_chunk
    assert delta_timestamps[STAGE_LABEL_KEY] == expected_chunk
    assert OBJECT_MASK_LABEL_KEY not in delta_timestamps
    assert OBS_STATE not in delta_timestamps
    assert RGB_CAMERA_KEY not in delta_timestamps


def test_train_step_with_all_heads():
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.train()

    loss, loss_dict = policy(make_train_batch())
    loss.backward()

    assert torch.isfinite(loss)
    assert set(loss_dict) == {
        "l1_loss",
        "kld_loss",
        "object_distance_l1_loss",
        "stage_cross_entropy_loss",
        "stage_accuracy",
        "object_mask_binary_cross_entropy_loss",
    }
    assert all(isinstance(value, float) for value in loss_dict.values())
    for head in (policy.model.object_distance_head, policy.model.stage_head, policy.model.object_mask_head):
        assert head.weight.grad is not None
    assert policy.model.encoder_camera_identity_embedding.weight.grad is not None


def test_auxiliary_output_shapes():
    batch_size = 2
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))

    actions, auxiliary_outputs = policy.predict_action_chunk_with_auxiliary_outputs(
        make_inference_batch(batch_size)
    )

    assert actions.shape == (batch_size, CHUNK_SIZE, ACTION_DIM)
    assert auxiliary_outputs["object_distance"].shape == (batch_size, CHUNK_SIZE)
    assert auxiliary_outputs["stage_logits"].shape == (batch_size, CHUNK_SIZE, STAGE_CLASS_COUNT)
    assert auxiliary_outputs[OBJECT_MASK_LABEL_KEY].shape == (batch_size, *TOKEN_GRID_SHAPE)


def test_padded_steps_do_not_contribute_to_the_loss():
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.eval()
    batch = make_train_batch()
    padded_step_count = 3
    for pad_key in ("action_is_pad", f"{OBJECT_DISTANCE_LABEL_KEY}_is_pad", f"{STAGE_LABEL_KEY}_is_pad"):
        batch[pad_key][:, -padded_step_count:] = True
    altered_batch = {key: value.clone() for key, value in batch.items()}
    altered_batch[ACTION][:, -padded_step_count:] += 100.0
    altered_batch[OBJECT_DISTANCE_LABEL_KEY][:, -padded_step_count:] += 100.0
    altered_batch[STAGE_LABEL_KEY][:, -padded_step_count:] = (
        altered_batch[STAGE_LABEL_KEY][:, -padded_step_count:] + 1
    ) % STAGE_CLASS_COUNT

    with torch.no_grad():
        loss, loss_dict = policy(batch)
        altered_loss, altered_loss_dict = policy(altered_batch)

    torch.testing.assert_close(loss, altered_loss)
    assert loss_dict == altered_loss_dict


def evaluate_loss_dict(policy: KineticACTPolicy, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        _, loss_dict = policy(batch)
    return loss_dict


def test_missing_object_distance_is_left_out_like_padding():
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.eval()
    batch = make_train_batch()
    missing_batch = {key: value.clone() for key, value in batch.items()}
    missing_batch[OBJECT_DISTANCE_LABEL_KEY][0, 3] = float("nan")
    padded_batch = {key: value.clone() for key, value in batch.items()}
    padded_batch[OBJECT_DISTANCE_LABEL_KEY][0, 3] = 5.0
    padded_batch[f"{OBJECT_DISTANCE_LABEL_KEY}_is_pad"][0, 3] = True

    missing_loss = evaluate_loss_dict(policy, missing_batch)["object_distance_l1_loss"]
    padded_loss = evaluate_loss_dict(policy, padded_batch)["object_distance_l1_loss"]

    assert missing_loss == pytest.approx(padded_loss)
    assert torch.isfinite(torch.tensor(missing_loss))


def test_missing_stage_is_left_out_like_padding():
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.eval()
    batch = make_train_batch()
    missing_batch = {key: value.clone() for key, value in batch.items()}
    missing_batch[STAGE_LABEL_KEY][1, 2] = MISSING_STAGE_LABEL
    padded_batch = {key: value.clone() for key, value in batch.items()}
    padded_batch[f"{STAGE_LABEL_KEY}_is_pad"][1, 2] = True

    missing_loss_dict = evaluate_loss_dict(policy, missing_batch)
    padded_loss_dict = evaluate_loss_dict(policy, padded_batch)

    assert missing_loss_dict["stage_cross_entropy_loss"] == pytest.approx(
        padded_loss_dict["stage_cross_entropy_loss"]
    )
    assert missing_loss_dict["stage_accuracy"] == pytest.approx(padded_loss_dict["stage_accuracy"])


def test_missing_object_mask_sample_is_left_out_of_the_mask_loss():
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.eval()
    batch = make_train_batch(batch_size=2)
    batch[OBJECT_MASK_LABEL_KEY][0] = MISSING_MASK_CELL_VALUE
    only_valid_sample = {key: value[1:2].clone() for key, value in batch.items()}

    loss_with_missing_sample = evaluate_loss_dict(policy, batch)["object_mask_binary_cross_entropy_loss"]
    loss_of_valid_sample = evaluate_loss_dict(policy, only_valid_sample)[
        "object_mask_binary_cross_entropy_loss"
    ]

    assert loss_with_missing_sample == pytest.approx(loss_of_valid_sample, abs=1e-5)


def test_all_object_masks_missing_gives_zero_mask_loss():
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.eval()
    batch = make_train_batch()
    batch[OBJECT_MASK_LABEL_KEY][:] = MISSING_MASK_CELL_VALUE

    loss_dict = evaluate_loss_dict(policy, batch)

    assert loss_dict["object_mask_binary_cross_entropy_loss"] == 0.0


def test_missing_labels_give_finite_gradients():
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.train()
    batch = make_train_batch()
    batch[OBJECT_DISTANCE_LABEL_KEY][0, :] = float("nan")
    batch[STAGE_LABEL_KEY][1, :4] = MISSING_STAGE_LABEL
    batch[OBJECT_MASK_LABEL_KEY][0] = MISSING_MASK_CELL_VALUE

    loss, _ = policy(batch)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_single_channel_camera_only():
    config = make_kinetic_act_config(camera_keys=(DEPTH_CAMERA_KEY,))
    policy = KineticACTPolicy(config)
    batch = make_inference_batch()
    del batch[RGB_CAMERA_KEY]

    actions = policy.predict_action_chunk(batch)

    assert policy.model.single_channel_camera_keys == {DEPTH_CAMERA_KEY}
    assert actions.shape == (2, CHUNK_SIZE, ACTION_DIM)
    assert torch.isfinite(actions).all()


def test_select_action_returns_one_action_per_batch_element():
    batch_size = 2
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.reset()

    action = policy.select_action(make_inference_batch(batch_size))

    assert action.shape == (batch_size, ACTION_DIM)


def test_temporal_ensembling_returns_one_action_per_step():
    batch_size = 2
    policy = KineticACTPolicy(
        make_kinetic_act_config(temporal_ensemble_coeff=0.01, n_action_steps=1, **ALL_HEAD_OVERRIDES)
    )
    policy.reset()

    first_action = policy.select_action(make_inference_batch(batch_size))
    second_action = policy.select_action(make_inference_batch(batch_size))

    assert first_action.shape == (batch_size, ACTION_DIM)
    assert second_action.shape == (batch_size, ACTION_DIM)


def test_save_and_load_round_trip(tmp_path):
    policy = KineticACTPolicy(make_kinetic_act_config(**ALL_HEAD_OVERRIDES))
    policy.save_pretrained(tmp_path)

    loaded_policy = KineticACTPolicy.from_pretrained(tmp_path)

    assert isinstance(loaded_policy.config, KineticACTConfig)
    assert loaded_policy.config.object_mask_label_key_per_camera == {RGB_CAMERA_KEY: OBJECT_MASK_LABEL_KEY}
    original_state = policy.state_dict()
    for name, value in loaded_policy.state_dict().items():
        torch.testing.assert_close(value, original_state[name])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stage_label_key": STAGE_LABEL_KEY, "stage_class_count": 1}, "stage_class_count"),
        (
            {
                "object_mask_label_key_per_camera": {
                    "observation.images.missing": "observation.object_mask.missing"
                }
            },
            "not image input",
        ),
        ({"object_distance_loss_weight": -1.0}, "object_distance_loss_weight"),
        ({"stage_loss_weight": -0.1}, "stage_loss_weight"),
        ({"object_mask_loss_weight": -2.0}, "object_mask_loss_weight"),
        ({"object_mask_positive_class_weight": 0.0}, "object_mask_positive_class_weight"),
        ({"object_distance_label_key": "object_distance"}, "must start with 'observation.'"),
        ({"stage_label_key": OBS_STATE}, "must start with 'observation.'"),
        ({"stage_label_key": f"{OBS_IMAGES}.stage"}, "must start with 'observation.'"),
        (
            {"object_distance_label_key": STAGE_LABEL_KEY, "stage_label_key": STAGE_LABEL_KEY},
            "must be distinct",
        ),
    ],
)
def test_validate_features_rejects_invalid_values(overrides, message):
    config = make_kinetic_act_config(**overrides)

    with pytest.raises(ValueError, match=message):
        config.validate_features()
