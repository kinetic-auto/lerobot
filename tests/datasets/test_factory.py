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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import (
    _resolve_episode_split,
    _uses_recomputed_relative_action_stats,
    apply_relative_state_history_stats,
    make_dataset,
    make_train_eval_datasets,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import EpisodeSplit
from lerobot.model.cartesian_quantities import FramePoseRot6D, FrameWrench
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.kinetic_act.configuration_kinetic_act import KineticACTConfig
from lerobot.utils.constants import OBS_STATE, PRETRAINED_MODEL_DIR, SPLIT_INFO
from lerobot.utils.io_utils import write_json


def test_make_three_way_episode_split(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    dataset = lerobot_dataset_factory(
        root=root,
        total_episodes=10,
        total_frames=100,
        use_videos=False,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset.repo_id,
            root=str(root),
            split_ratio=[8, 1, 1],
        ),
        policy=ACTConfig(device="cpu"),
        seed=42,
    )

    result = make_train_eval_datasets(cfg)

    assert result.eval is not None
    assert result.test is not None
    train = set(result.train.episodes)
    validation = set(result.eval.episodes)
    test = set(result.test.episodes)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(range(10))
    assert result.eval.image_transforms is None
    assert result.test.image_transforms is None


OBJECT_MASK_KEY_PER_CAMERA = {"observation.images.laptop": "observation.object_mask.laptop"}


@pytest.mark.parametrize(
    ("policy_config", "expected_mapping"),
    [
        (ACTConfig(device="cpu"), None),
        (
            KineticACTConfig(device="cpu", object_mask_label_key_per_camera=OBJECT_MASK_KEY_PER_CAMERA),
            OBJECT_MASK_KEY_PER_CAMERA,
        ),
    ],
    ids=["act", "kinetic_act"],
)
def test_datasets_get_the_policy_mask_label_keys(
    tmp_path, lerobot_dataset_factory, policy_config, expected_mapping
):
    root = tmp_path / "dataset"
    dataset = lerobot_dataset_factory(root=root, total_episodes=4, total_frames=40, use_videos=False)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=dataset.repo_id, root=str(root), split_ratio=[2, 1, 1]),
        policy=policy_config,
        seed=42,
    )

    full_dataset = make_dataset(cfg)
    split_datasets = make_train_eval_datasets(cfg)

    for split_dataset in (full_dataset, split_datasets.train, split_datasets.eval, split_datasets.test):
        assert split_dataset.mask_label_key_per_camera == expected_mapping


def test_make_eval_split_has_no_test_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    dataset = lerobot_dataset_factory(
        root=root,
        total_episodes=4,
        total_frames=40,
        use_videos=False,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset.repo_id,
            root=str(root),
            eval_split=0.25,
        ),
        policy=ACTConfig(device="cpu"),
    )

    result = make_train_eval_datasets(cfg)

    assert result.eval is not None
    assert result.test is None
    assert result.episode_split is None


def test_resume_split_rejects_changed_episode_selection(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    pretrained = checkpoint / PRETRAINED_MODEL_DIR
    pretrained.mkdir(parents=True)
    write_json(
        EpisodeSplit([8, 1, 1], 3, [0], [1], [2], seed=42).to_dict(),
        pretrained / SPLIT_INFO,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/dataset", split_ratio=[8, 1, 1]),
        policy=ACTConfig(device="cpu"),
        resume=True,
        seed=42,
    )
    cfg.checkpoint_path = checkpoint

    with pytest.raises(ValueError, match="do not match"):
        _resolve_episode_split(cfg, [1, 2, 3])


def test_relative_stats_leave_existing_relative_policies_unchanged():
    assert not _uses_recomputed_relative_action_stats(SimpleNamespace(use_relative_actions=True))


def test_relative_state_history_stats_keep_rotation_unscaled_and_wrench_from_the_dataset(tmp_path):
    state_names = [*FramePoseRot6D.feature_names(), "gripper.position.x", *FrameWrench.feature_names()]
    dataset = LeRobotDataset.create(
        repo_id="local/cartesian",
        fps=10,
        root=tmp_path / "dataset",
        features={OBS_STATE: {"dtype": "float32", "shape": (len(state_names),), "names": state_names}},
        use_videos=False,
    )
    for frame_index in range(6):
        angle = 0.1 * frame_index
        state = np.array(
            [
                0.1 * frame_index,
                0.0,
                0.5,
                np.cos(angle),
                np.sin(angle),
                0.0,
                -np.sin(angle),
                np.cos(angle),
                0.0,
                0.02,
            ]
            + [float(frame_index)] * 6,
            dtype=np.float32,
        )
        dataset.add_frame({"task": "open the door", OBS_STATE: state})
    dataset.save_episode()
    dataset.finalize()
    dataset = LeRobotDataset("local/cartesian", root=tmp_path / "dataset")
    dataset_wrench_mean = np.asarray(dataset.meta.stats[OBS_STATE]["mean"])[10:]
    policy_config = SimpleNamespace(
        relative_exclude_joints=["force", "torque"],
        use_se3_relative_pose=True,
        state_observation_delta_indices=[-1, 0],
    )

    apply_relative_state_history_stats(dataset, policy_config)

    stats = dataset.meta.stats[OBS_STATE]
    torch.testing.assert_close(stats["mean"][3:9], torch.zeros(6))
    torch.testing.assert_close(stats["std"][3:9], torch.ones(6))
    torch.testing.assert_close(stats["min"][3:9], -torch.ones(6))
    torch.testing.assert_close(stats["q99"][3:9], torch.ones(6))
    np.testing.assert_allclose(stats["mean"][10:].numpy(), dataset_wrench_mean, atol=1e-6)
    assert stats["mean"][0] < 0
