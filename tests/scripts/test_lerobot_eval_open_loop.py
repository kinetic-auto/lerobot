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

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig
from lerobot.configs.eval import OpenLoopEvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import EpisodeSplit
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval_open_loop import (
    EpisodeReplay,
    aggregate_metrics,
    compute_episode_metrics,
    evaluate_open_loop,
    load_episode_split,
    observation_delta_timestamps,
    select_episodes_to_evaluate,
    write_metrics_per_episode,
    write_metrics_summary,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, SPLIT_INFO
from lerobot.utils.io_utils import load_json, write_json
from tests.fixtures.constants import DUMMY_CAMERA_FEATURES, DUMMY_MOTOR_FEATURES, DUMMY_REPO_ID

JOINT_NAMES = ["shoulder.position", "elbow.position", "shoulder.effort", "elbow.effort"]


def make_config(tmp_path: Path, **overrides) -> OpenLoopEvalConfig:
    kwargs = {
        "dataset": DatasetConfig(root=str(tmp_path)),
        "policy": ACTConfig(device="cpu", push_to_hub=False),
        "output_dir": tmp_path / "eval",
    }
    kwargs.update(overrides)
    return OpenLoopEvalConfig(**kwargs)


def make_split() -> EpisodeSplit:
    return EpisodeSplit(
        split_ratio=[0.6, 0.2, 0.2],
        total_episodes=10,
        train_episodes=[0, 1, 2, 3, 4, 5],
        val_episodes=[6, 7],
        test_episodes=[8, 9],
        seed=1,
    )


def make_replay(predicted: np.ndarray, ground_truth: np.ndarray) -> EpisodeReplay:
    return EpisodeReplay(
        episode_index=0,
        split_label="val",
        predicted=predicted,
        ground_truth=ground_truth,
        observed_state=None,
    )


def test_config_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split"):
        make_config(tmp_path, split="holdout")


def test_config_rejects_streaming(tmp_path):
    with pytest.raises(ValueError, match="streaming"):
        make_config(tmp_path, dataset=DatasetConfig(root=str(tmp_path), streaming=True))


def test_config_requires_policy(tmp_path):
    with pytest.raises(ValueError, match="policy.path"):
        make_config(tmp_path, policy=None)


def test_config_default_output_dir_sits_next_to_local_checkpoint(tmp_path):
    pretrained_dir = tmp_path / "checkpoints" / "000100" / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    policy = ACTConfig(device="cpu", push_to_hub=False)
    policy.pretrained_path = pretrained_dir
    cfg = make_config(tmp_path, policy=policy, output_dir=None)
    assert cfg.output_dir == pretrained_dir.parent / "eval_open_loop"


def test_select_episodes_caps_each_split_deterministically(tmp_path):
    cfg = make_config(tmp_path, episodes_per_split=1, seed=3)
    first = select_episodes_to_evaluate(cfg, make_split(), total_episodes=10)
    second = select_episodes_to_evaluate(cfg, make_split(), total_episodes=10)
    assert first == second
    assert [label for _, label in first] == ["train", "val", "test"]
    assert first == sorted(first)


def test_select_episodes_single_split_and_no_cap(tmp_path):
    cfg = make_config(tmp_path, split="val", episodes_per_split=0)
    assert select_episodes_to_evaluate(cfg, make_split(), total_episodes=10) == [(6, "val"), (7, "val")]


def test_select_episodes_without_split_file_labels_all(tmp_path):
    cfg = make_config(tmp_path, episodes_per_split=0)
    selected = select_episodes_to_evaluate(cfg, None, total_episodes=3)
    assert selected == [(0, "all"), (1, "all"), (2, "all")]


def test_select_episodes_named_split_needs_split_file(tmp_path):
    cfg = make_config(tmp_path, split="val")
    with pytest.raises(ValueError, match=SPLIT_INFO):
        select_episodes_to_evaluate(cfg, None, total_episodes=3)


def test_select_episodes_drops_split_episodes_outside_dataset(tmp_path, caplog):
    cfg = make_config(tmp_path, split="test", episodes_per_split=0)
    with caplog.at_level("WARNING"):
        selected = select_episodes_to_evaluate(cfg, make_split(), total_episodes=9)
    assert selected == [(8, "test")]
    assert "outside the dataset range" in caplog.text


def test_select_episodes_empty_split_raises(tmp_path):
    cfg = make_config(tmp_path, split="test")
    split = make_split()
    split.test_episodes = []
    with pytest.raises(ValueError, match="No episodes"):
        select_episodes_to_evaluate(cfg, split, total_episodes=10)


def test_select_episodes_explicit_list_keeps_split_labels(tmp_path):
    cfg = make_config(tmp_path, dataset=DatasetConfig(root=str(tmp_path), episodes=[9, 2, 6]))
    assert select_episodes_to_evaluate(cfg, make_split(), total_episodes=10) == [
        (2, "train"),
        (6, "val"),
        (9, "test"),
    ]
    assert select_episodes_to_evaluate(cfg, None, total_episodes=10) == [
        (2, "manual"),
        (6, "manual"),
        (9, "manual"),
    ]


def test_select_episodes_explicit_list_out_of_range(tmp_path):
    cfg = make_config(tmp_path, dataset=DatasetConfig(root=str(tmp_path), episodes=[12]))
    with pytest.raises(ValueError, match="outside"):
        select_episodes_to_evaluate(cfg, make_split(), total_episodes=10)


def test_load_episode_split_round_trip(tmp_path):
    write_json(make_split().to_dict(), tmp_path / SPLIT_INFO)
    assert load_episode_split(tmp_path) == make_split()
    assert load_episode_split(tmp_path / "missing") is None
    assert load_episode_split(None) is None


def test_observation_delta_timestamps_drops_action_window():
    metadata = SimpleNamespace(
        features={"observation.state": {}, "observation.images.cam": {}, "action": {}}, fps=10
    )
    delta_timestamps = observation_delta_timestamps(DiffusionConfig(), metadata, rename_map={})
    assert delta_timestamps is not None
    assert "action" not in delta_timestamps
    assert delta_timestamps["observation.state"] == [-0.1, 0.0]
    assert observation_delta_timestamps(ACTConfig(), metadata, rename_map={}) is None


def test_compute_episode_metrics_matches_hand_computation():
    ground_truth = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]])
    predicted = ground_truth + np.array([[0.1, -0.2, 0.5, 0.0], [0.1, -0.2, 0.5, 0.0], [0.1, -0.2, 0.5, 0.0]])
    metrics = compute_episode_metrics(make_replay(predicted, ground_truth), JOINT_NAMES)

    assert metrics.num_frames == 3
    assert metrics.mae == pytest.approx(0.15)
    assert metrics.mse == pytest.approx((0.01 + 0.04) / 2)
    assert metrics.rmse == pytest.approx(np.sqrt((0.01 + 0.04) / 2))
    assert metrics.block_mae == {"position": pytest.approx(0.15), "effort": pytest.approx(0.25)}
    assert metrics.per_joint_mae["shoulder.effort"] == pytest.approx(0.5)
    assert metrics.per_joint_rmse["elbow.position"] == pytest.approx(0.2)
    assert metrics.max_abs_error == pytest.approx(0.5)
    assert metrics.max_abs_error_joint == "shoulder.effort"
    assert metrics.ground_truth_smoothness_mean == pytest.approx(np.sqrt(2.0))
    assert metrics.predicted_smoothness_mean == pytest.approx(np.sqrt(2.0))
    assert metrics.ground_truth_smoothness_std == pytest.approx(0.0)


def test_compute_episode_metrics_perfect_prediction():
    ground_truth = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    metrics = compute_episode_metrics(make_replay(ground_truth.copy(), ground_truth), JOINT_NAMES)
    assert metrics.mae == 0.0
    assert metrics.cosine_similarity == pytest.approx(1.0)
    assert metrics.block_cosine["effort"] == pytest.approx(1.0)


def test_reports_aggregate_and_write(tmp_path):
    ground_truth = np.zeros((4, 4))
    metrics = [
        compute_episode_metrics(make_replay(ground_truth + 0.2, ground_truth), JOINT_NAMES),
        compute_episode_metrics(make_replay(ground_truth + 0.4, ground_truth), JOINT_NAMES),
    ]
    metrics[1].split_label = "test"

    aggregate = aggregate_metrics(metrics)
    assert aggregate["count"] == 2
    assert aggregate["mean_mae"] == pytest.approx(0.3)
    assert aggregate["per_joint_mae"]["elbow.effort"] == pytest.approx(0.3)
    assert aggregate_metrics([]) == {}

    write_metrics_summary(metrics, tmp_path / "metrics_summary.json")
    summary = load_json(tmp_path / "metrics_summary.json")
    assert summary["overall"]["count"] == 2
    assert set(summary["by_split"]) == {"val", "test"}
    assert summary["by_split"]["test"]["mean_mae"] == pytest.approx(0.4)

    write_metrics_per_episode(metrics, tmp_path / "metrics_per_episode.csv")
    with (tmp_path / "metrics_per_episode.csv").open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["split_label"] for row in rows] == ["val", "test"]
    assert float(rows[0]["mae_shoulder.position"]) == pytest.approx(0.2)


def make_checkpoint(dataset, checkpoint_dir: Path, **config_overrides) -> Path:
    config = ACTConfig(
        dim_model=64,
        dim_feedforward=128,
        n_heads=2,
        chunk_size=4,
        n_action_steps=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
        push_to_hub=False,
        **config_overrides,
    )
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    pretrained_dir = checkpoint_dir / "pretrained_model"
    policy.save_pretrained(pretrained_dir)
    preprocessor.save_pretrained(pretrained_dir)
    postprocessor.save_pretrained(pretrained_dir)
    write_json(
        EpisodeSplit(
            split_ratio=[0.5, 0.5, 0.0],
            total_episodes=2,
            train_episodes=[0],
            val_episodes=[1],
            test_episodes=[],
            seed=0,
        ).to_dict(),
        pretrained_dir / SPLIT_INFO,
    )
    return pretrained_dir


@pytest.mark.parametrize("relative", [False, True])
def test_evaluate_open_loop_end_to_end(tmp_path, lerobot_dataset_factory, info_factory, relative):
    pytest.importorskip("matplotlib", reason="matplotlib is required (install lerobot[dataset_viz])")
    info = info_factory(
        total_episodes=2,
        total_frames=12,
        total_tasks=1,
        use_videos=False,
        motor_features={ACTION: DUMMY_MOTOR_FEATURES[ACTION], OBS_STATE: DUMMY_MOTOR_FEATURES["state"]},
        camera_features={f"{OBS_IMAGES}.laptop": DUMMY_CAMERA_FEATURES["laptop"]},
    )
    dataset = lerobot_dataset_factory(root=tmp_path / "dataset", info=info)
    config_overrides = (
        {"use_relative_actions": True, "relative_action_mode": "sequential"} if relative else {}
    )
    pretrained_dir = make_checkpoint(dataset, tmp_path / "checkpoints" / "000010", **config_overrides)

    policy_config = PreTrainedConfig.from_pretrained(pretrained_dir)
    policy_config.pretrained_path = pretrained_dir
    cfg = OpenLoopEvalConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=str(dataset.root)),
        policy=policy_config,
        episodes_per_split=0,
    )
    evaluate_open_loop(cfg)

    output_dir = pretrained_dir.parent / "eval_open_loop"
    summary = load_json(output_dir / "metrics_summary.json")
    assert summary["overall"]["count"] == 2
    assert set(summary["by_split"]) == {"train", "val"}
    assert set(summary["overall"]["per_joint_mae"]) == set(dataset.features[ACTION]["names"])
    with (output_dir / "metrics_per_episode.csv").open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert sum(int(row["num_frames"]) for row in rows) == info.total_frames
    assert (output_dir / "plots" / "episode_0000_train.png").is_file()
    assert (output_dir / "plots" / "episode_0001_val.png").is_file()
    assert (output_dir / "plots" / "per_joint_mae_by_split.png").is_file()
    assert all(np.isfinite(float(row["mae"])) for row in rows)
