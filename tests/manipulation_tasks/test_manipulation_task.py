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

from pathlib import Path

import numpy as np
import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.manipulation_tasks import (
    KineticManipulationTask,
    KineticManipulationTaskConfig,
    override_task_config,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.kinetic_act.configuration_kinetic_act import KineticACTConfig
from lerobot.policies.kinetic_act.modeling_kinetic_act import KineticACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERA_KEY = "observation.images.front"
STATE_DIMENSION = 6
ACTION_DIMENSION = 6
CHUNK_SIZE = 4
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32


def save_kinetic_act_checkpoint(checkpoint_path: Path) -> Path:
    config = KineticACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIMENSION,)),
            CAMERA_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIMENSION,))},
        pretrained_backbone_weights=None,
        dim_model=32,
        dim_feedforward=64,
        n_heads=2,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        device="cpu",
        push_to_hub=False,
    )
    policy = KineticACTPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    policy.save_pretrained(checkpoint_path)
    preprocessor.save_pretrained(checkpoint_path)
    postprocessor.save_pretrained(checkpoint_path)
    return checkpoint_path


def write_task_config_yaml(task_config_directory: Path, task_name: str, checkpoint_path: Path) -> None:
    task_config_directory.mkdir(parents=True, exist_ok=True)
    (task_config_directory / f"{task_name}.yaml").write_text(
        f"task_name: {task_name}\ntask_description: {task_name} task\npolicy_checkpoint_path: {checkpoint_path}\n"
    )


def build_single_frame_observation() -> dict[str, np.ndarray]:
    return {
        OBS_STATE: np.random.default_rng(0).standard_normal(STATE_DIMENSION).astype(np.float32),
        CAMERA_KEY: np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
    }


@pytest.fixture
def task_config_directory(tmp_path: Path) -> Path:
    checkpoint_path = save_kinetic_act_checkpoint(tmp_path / "checkpoint")
    directory = tmp_path / "tasks"
    write_task_config_yaml(directory, "door_opening", checkpoint_path)
    write_task_config_yaml(directory, "drawer_opening", checkpoint_path)
    return directory


def test_prepared_task_predicts_an_action_chunk(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    task.prepare_task("door_opening")

    action_chunk = task.predict_action_chunk(build_single_frame_observation())

    assert tuple(action_chunk.shape) == (CHUNK_SIZE, ACTION_DIMENSION)


def test_prepared_task_reports_the_task_and_policy_properties(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    task.prepare_task("door_opening")

    assert task.active_task_name == "door_opening"
    assert task.task_description == "door_opening task"
    assert task.policy_type == "kinetic_act"
    assert task.camera_image_shapes == {CAMERA_KEY: (3, IMAGE_HEIGHT, IMAGE_WIDTH)}
    assert (task.state_dimension, task.chunk_size, task.state_history_length) == (
        STATE_DIMENSION,
        CHUNK_SIZE,
        1,
    )


def test_preparing_the_active_task_keeps_the_loaded_policy(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    task.prepare_task("door_opening")
    loaded_policy = task._policy

    task.prepare_task("door_opening")

    assert task._policy is loaded_policy


def test_preparing_the_active_task_resets_the_policy(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    task.prepare_task("door_opening")
    first_action_queue = task._policy._action_queue

    task.prepare_task("door_opening")

    assert task._policy._action_queue is not first_action_queue


def test_preparing_another_task_replaces_the_loaded_policy(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    task.prepare_task("door_opening")
    first_policy = task._policy

    task.prepare_task("drawer_opening")

    assert task.active_task_name == "drawer_opening"
    assert task._policy is not first_policy


def test_unload_task_releases_the_loaded_policy(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    task.prepare_task("door_opening")

    task.unload_task()

    assert task.active_task_name is None
    assert task._policy is None
    with pytest.raises(RuntimeError, match="prepare_task"):
        task.predict_action_chunk(build_single_frame_observation())


def test_predict_action_chunk_raises_before_prepare_task(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    with pytest.raises(RuntimeError, match="prepare_task"):
        task.predict_action_chunk(build_single_frame_observation())


def test_prepare_task_raises_on_an_unknown_task(task_config_directory):
    task = KineticManipulationTask(task_config_directory, device="cpu")
    with pytest.raises(ValueError, match="charging_port"):
        task.prepare_task("charging_port")


def test_load_from_yaml_loads_the_packaged_door_opening_config():
    task_config = KineticManipulationTaskConfig.load_from_yaml("door_opening")
    assert task_config.task_description == "open the door"


def test_override_task_config_sets_the_given_fields():
    task_config = KineticManipulationTaskConfig.load_from_yaml("door_opening")
    override_task_config(task_config, {"policy_checkpoint_path": "/other/checkpoint"})
    assert task_config.policy_checkpoint_path == "/other/checkpoint"
    assert task_config.task_description == "open the door"


def test_override_task_config_warns_and_skips_an_unknown_field(caplog):
    task_config = KineticManipulationTaskConfig.load_from_yaml("door_opening")
    with caplog.at_level("WARNING"):
        override_task_config(task_config, {"checkpoint": "/other/checkpoint"})
    assert "checkpoint" in caplog.text
    assert not hasattr(task_config, "checkpoint")


def test_prepare_task_loads_the_overridden_checkpoint(task_config_directory, tmp_path):
    other_checkpoint_path = save_kinetic_act_checkpoint(tmp_path / "other_checkpoint")
    task = KineticManipulationTask(task_config_directory, device="cpu")
    task.prepare_task("door_opening")
    first_policy = task._policy

    task.prepare_task("door_opening", {"policy_checkpoint_path": str(other_checkpoint_path)})

    assert task._task_config.policy_checkpoint_path == str(other_checkpoint_path)
    assert task._policy is not first_policy
