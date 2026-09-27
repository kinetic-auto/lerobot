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

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import (
    _resolve_episode_split,
    _uses_recomputed_relative_action_stats,
    make_train_eval_datasets,
)
from lerobot.datasets.utils import EpisodeSplit
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, SPLIT_INFO
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
