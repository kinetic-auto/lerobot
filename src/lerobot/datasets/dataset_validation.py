#!/usr/bin/env python

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

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import DEPTH_MILLIMETER_UNIT
from lerobot.utils.constants import ACTION, OBS_STATE
from .lerobot_dataset import LeRobotDataset

CONVERSION_MANIFEST_FILE_NAME = "mcap_conversion_manifest.json"
BATCH_READ_SAMPLE_COUNT = 8
TIMESTAMP_TOLERANCE_SECONDS = 1e-4


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def validate_dataset(root: Path, repo_id: str) -> list[CheckResult]:
    # Load the dataset and check if it can be loaded
    try:
        dataset = LeRobotDataset(repo_id, root=root, depth_output_unit=DEPTH_MILLIMETER_UNIT)
    except Exception as error:
        return [CheckResult("load dataset", False, f"{type(error).__name__}: {error}")]

    results = [
        CheckResult(
            "load dataset",
            True,
            f"{dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames "
            f"at {dataset.fps} fps",
        )
    ]

    # Run the dataset sanity checks
    checks: list[tuple[str, Callable[[LeRobotDataset, Path], str]]] = [
        ("features", check_features),
        ("first frame", check_first_frame),
        ("batch read", check_batch_read),
        ("timestamps and finite values", check_timestamps_and_finite_values),
        ("conversion manifest", check_manifest),
    ]

    for check_name, check in checks:
        results.append(run_check(check_name, check, dataset, root))
    return results


def run_check(
    check_name: str, check: Callable[[LeRobotDataset, Path], str], dataset: LeRobotDataset, root: Path
) -> CheckResult:
    try:
        return CheckResult(check_name, True, check(dataset, root))
    except Exception as error:
        return CheckResult(check_name, False, f"{type(error).__name__}: {error}")


def check_features(dataset: LeRobotDataset, root: Path) -> str:
    # Get the features from dataset meta
    features = dataset.meta.features

    # Check if the required features are present
    for required_key in (OBS_STATE, ACTION):
        if required_key not in features:
            raise ValueError(f"feature {required_key!r} is missing")

    # Check if the state and action shapes are the same
    # NOTE: This is a special case where actions are fetched from future observations.
    if features[OBS_STATE]["shape"] != features[ACTION]["shape"]:
        raise ValueError(
            f"state shape {features[OBS_STATE]['shape']} differs from action shape "
            f"{features[ACTION]['shape']}"
        )

    # Check if the state names match the state dimension
    state_names = features[OBS_STATE]["names"]
    if len(state_names) != features[OBS_STATE]["shape"][0]:
        raise ValueError("state names do not match the state dimension")
    detail = f"{len(state_names)}-dim state and action"

    # Check if the dataset has any camera features
    if dataset.meta.video_keys:
        detail += f", cameras {dataset.meta.video_keys}"
    return detail


def check_first_frame(dataset: LeRobotDataset, root: Path) -> str:
    first_item = dataset[0]

    # Check if the first frame has the expected shape for each camera feature
    for camera_key in dataset.meta.video_keys:
        expected_channels = 1 if camera_key in dataset.meta.depth_keys else 3
        if first_item[camera_key].shape[0] != expected_channels:
            raise ValueError(f"{camera_key} has shape {tuple(first_item[camera_key].shape)}")

    for vector_key in (OBS_STATE, ACTION):
        if not torch.isfinite(first_item[vector_key]).all():
            raise ValueError(f"{vector_key} has non-finite values in the first frame")

    if dataset.meta.video_keys:
        return "state, action and all camera frames read"

    return "state and action read"


def check_batch_read(dataset: LeRobotDataset, root: Path) -> str:
    sample_count = min(BATCH_READ_SAMPLE_COUNT, len(dataset))

    # Sample a batch of indices from the dataset
    indices = np.linspace(0, len(dataset) - 1, num=sample_count).astype(int)

    # Check if the batch of indices can be read
    for index in indices:
        dataset[int(index)]

    return f"read {sample_count} frames spread across the dataset"


def check_timestamps_and_finite_values(dataset: LeRobotDataset, root: Path) -> str:
    columns = dataset.hf_dataset.with_format("numpy")
    episode_indices = np.asarray(columns["episode_index"])
    timestamps = np.asarray(columns["timestamp"], dtype=np.float64)
    expected_step = 1.0 / dataset.fps

    # Check if the timestamps are evenly spaced
    for episode_index in np.unique(episode_indices):
        episode_timestamps = timestamps[episode_indices == episode_index]
        steps = np.diff(episode_timestamps)
        if steps.size and np.abs(steps - expected_step).max() > TIMESTAMP_TOLERANCE_SECONDS:
            raise ValueError(f"episode {episode_index} timestamps are not spaced by 1/{dataset.fps}s")

    # Check if the state and action contain only finite values
    for vector_key in (OBS_STATE, ACTION):
        if not np.isfinite(np.asarray(columns[vector_key])).all():
            raise ValueError(f"{vector_key} contains non-finite values")

    return f"{len(np.unique(episode_indices))} episodes evenly spaced and finite"


def check_manifest(dataset: LeRobotDataset, root: Path) -> str:
    # Check if the mcap conversion manifest exists
    manifest_path = root / CONVERSION_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return f"no {CONVERSION_MANIFEST_FILE_NAME}, cross-check skipped"

    # Check if the episode count and manifest conversion episode count match
    converted_episodes = json.loads(manifest_path.read_text())["converted"]
    if len(converted_episodes) != dataset.meta.total_episodes:
        raise ValueError(
            f"manifest lists {len(converted_episodes)} episodes but the dataset has "
            f"{dataset.meta.total_episodes}"
        )

    # Check if the frame count and manifest conversion frame count match
    manifest_frames = sum(episode["frame_count"] for episode in converted_episodes)
    if manifest_frames != dataset.meta.total_frames:
        raise ValueError(
            f"manifest lists {manifest_frames} frames but the dataset has {dataset.meta.total_frames}"
        )

    return "episode and frame counts match the conversion manifest"
