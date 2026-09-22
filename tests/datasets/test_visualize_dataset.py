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
from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import numpy as np

from lerobot.scripts.lerobot_dataset_viz import (
    group_feature_dims,
    infer_repo_id,
    resolve_dataset_root,
    scalar_entity_path,
    select_video_frames,
    visualize_dataset,
)


def test_infer_repo_id_from_root():
    assert infer_repo_id(None, "/data/car-door-opening-20260910") == "car-door-opening-20260910"


def test_infer_repo_id_prefers_explicit():
    assert infer_repo_id("me/ds", "/data/foo") == "me/ds"


def test_resolve_dataset_root_requires_info_json(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_dataset_root(str(tmp_path))


def test_resolve_dataset_root_accepts_valid_local(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    assert Path(resolve_dataset_root(str(tmp_path))) == tmp_path.resolve()


def test_resolve_dataset_root_keeps_remote_uri():
    uri = "s3://bucket/dataset"
    assert resolve_dataset_root(uri) == uri


def test_group_feature_dims_packed_joints():
    names = [
        "right_finger_joint1.position",
        "right_joint1.position",
        "right_finger_joint1.velocity",
        "right_joint1.velocity",
        "right_finger_joint1.effort",
        "right_joint1.effort",
    ]
    groups = group_feature_dims(names)
    assert [group[0] for group in groups] == ["position", "velocity", "effort"]
    assert groups[0][1] == [0, 1]
    assert groups[1][2] == ["right_finger_joint1.velocity", "right_joint1.velocity"]


def test_group_feature_dims_flat_names():
    names = ["joint_0", "joint_1"]
    groups = group_feature_dims(names)
    assert groups == [("", [0, 1], ["joint_0", "joint_1"])]


def test_scalar_entity_path():
    assert scalar_entity_path("action", "position") == "action/position"
    assert scalar_entity_path("state", "") == "state"


def test_select_video_frames_offsets_to_episode_start():
    fps = 60
    frame_ns = (np.arange(0, 180) * 1e9 / fps).astype(np.int64)
    selected_ns, relative_s = select_video_frames(frame_ns, from_s=1.0, to_s=2.0)
    assert 60 <= selected_ns.size <= 61
    assert relative_s[0] == pytest.approx(0.0, abs=1e-3)
    assert relative_s[-1] == pytest.approx(1.0, abs=2 / fps)


@pytest.mark.skip("TODO: add dummy videos")
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(root=root)
    rrd_path = visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
    )
    assert rrd_path.exists()
