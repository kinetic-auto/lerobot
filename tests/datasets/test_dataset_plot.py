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
import shutil
import sys

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("matplotlib", reason="matplotlib is required (install lerobot[dataset_viz])")

import numpy as np

from lerobot.scripts.lerobot_dataset_plot import (
    main,
    select_midpoint_timestamps,
    resolve_image_observation_keys,
    split_episode_timestamps,
    snap_to_frame_timestamps,
    validate_episode_indices,
    validate_state_observation_keys,
)


def test_split_episode_timestamps_count():
    timestamps = np.array([0.0, 1.0, 2.0])
    boundaries = split_episode_timestamps(timestamps, 3)
    assert len(boundaries) == 4
    assert boundaries[0] == pytest.approx(0.0)
    assert boundaries[-1] == pytest.approx(2.0)


def test_split_episode_timestamps_rejects_zero():
    with pytest.raises(ValueError, match="at least 1"):
        split_episode_timestamps(np.array([0.0, 1.0]), 0)


def test_select_midpoint_timestamps():
    np.testing.assert_allclose(select_midpoint_timestamps(np.array([0.0, 2.0, 4.0])), [1.0, 3.0])


def test_snap_to_frame_timestamps():
    frame_timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    snapped = snap_to_frame_timestamps(np.array([0.04, 0.26]), frame_timestamps)
    np.testing.assert_allclose(snapped, [0.0, 0.3])


def test_validate_state_observation_keys_missing(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(root=tmp_path, use_videos=False)
    with pytest.raises(ValueError, match="observation.velocity"):
        validate_state_observation_keys(dataset, ["observation.velocity"])


def test_validate_state_observation_keys_rejects_image(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(root=tmp_path, use_videos=False)
    image_key = dataset.meta.camera_keys[0]
    with pytest.raises(ValueError, match=image_key):
        validate_state_observation_keys(dataset, [image_key])


def test_resolve_image_observation_keys_default(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(root=tmp_path, use_videos=True)
    assert resolve_image_observation_keys(dataset, None) == [
        key
        for key in dataset.meta.camera_keys
        if key in dataset.meta.video_keys and key not in dataset.meta.depth_keys
    ]


def test_resolve_image_observation_keys_unknown(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(root=tmp_path, use_videos=True)
    with pytest.raises(ValueError, match="observation.images.missing"):
        resolve_image_observation_keys(dataset, ["observation.images.missing"])


def test_resolve_image_observation_keys_preserves_order(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(root=tmp_path, use_videos=True)
    keys = list(reversed(dataset.meta.camera_keys))
    assert resolve_image_observation_keys(dataset, keys) == keys


def test_validate_episode_indices_out_of_range():
    with pytest.raises(ValueError, match="5"):
        validate_episode_indices([0, 5], total_episodes=3)


def test_validate_episode_indices_none_and_valid():
    validate_episode_indices(None, total_episodes=3)
    validate_episode_indices([0, 2], total_episodes=3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_plot_all_episodes_default_output(tmp_path, lerobot_dataset_factory, monkeypatch):
    root = tmp_path / "dataset"
    lerobot_dataset_factory(root=root, use_videos=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-dataset-plot",
            "--dataset-path",
            str(root),
            "--num-image-samples",
            "3",
            "--state-observations",
            "state",
        ],
    )
    main()
    for episode_index in range(3):
        output_path = root / "episode_plots" / f"episode_{episode_index}.png"
        assert output_path.is_file()
        assert output_path.stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_plot_single_episode_custom_output(tmp_path, lerobot_dataset_factory, monkeypatch):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "plots"
    lerobot_dataset_factory(root=root, use_videos=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-dataset-plot",
            "--dataset-path",
            str(root),
            "--episodes",
            "1",
            "--num-image-samples",
            "3",
            "--state-observations",
            "state",
            "--output-dir",
            str(output_dir),
        ],
    )
    main()
    written = list(output_dir.glob("*.png"))
    assert [path.name for path in written] == ["episode_1.png"]
    assert written[0].stat().st_size > 0
