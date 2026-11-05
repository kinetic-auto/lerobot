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

import dataclasses
import itertools
import json
import logging
import shutil
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")

import numpy as np
import torch

import lerobot.scripts.kinetic_label_object_task as label_script
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import load_stats
from lerobot.policies.kinetic_act.configuration_kinetic_act import MISSING_MASK_CELL_VALUE
from lerobot.scripts.kinetic_label_object_task import (
    APPROACH_STAGE,
    GRASP_STAGE,
    PULL_STAGE,
    LabelObjectTaskConfig,
    Sam3Segmenter,
    StageThresholds,
    assign_stage_labels,
    downsample_mask,
    fill_missing_distances,
    find_position_channel_indices,
    get_image_height_and_width,
    label_dataset,
    measure_objects_in_frames,
    median_depth_under_mask,
    median_filter_distances,
    rotate_clockwise,
    segment_object_mask,
    segment_object_masks,
    to_uint8_rgb_image,
    validate_config,
    validate_dataset_features,
)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 96
FRAMES_PER_EPISODE = 12
EPISODE_COUNT = 2
FPS = 10
GRIPPER_CLOSES_AT_FRAME = 6
ARM_STARTS_MOVING_AT_FRAME = 8
# Depth in millimeters is 2000 - 100 * frame, so the object distance in meters is 2.0 - 0.1 * frame.
GRASP_DISTANCE_METERS = 1.5
THRESHOLDS = StageThresholds(
    gripper_closed_threshold=0.02,
    grasp_distance_meters=GRASP_DISTANCE_METERS,
    pull_joint_displacement_threshold=0.15,
)


def make_config(tmp_path, **overrides) -> LabelObjectTaskConfig:
    kwargs = {
        "repo_id": "local/source",
        "root": str(tmp_path / "source"),
        "new_repo_id": "local/labelled",
        "new_root": str(tmp_path / "labelled"),
        "gripper_closed_threshold": THRESHOLDS.gripper_closed_threshold,
        "grasp_distance_meters": THRESHOLDS.grasp_distance_meters,
        "pull_joint_displacement_threshold": THRESHOLDS.pull_joint_displacement_threshold,
        "distance_median_filter_frames": 1,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return LabelObjectTaskConfig(**kwargs)


def write_source_dataset(root) -> None:
    channel_names = ["joint1.position", "gripper.position.x"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": channel_names},
        "action": {"dtype": "float32", "shape": (2,), "names": channel_names},
        "observation.images.realsense": {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.realsense_depth": {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
    }
    dataset = LeRobotDataset.create("local/source", FPS, features, root=root, use_videos=False)
    for _ in range(EPISODE_COUNT):
        for frame in range(FRAMES_PER_EPISODE):
            gripper = 0.044 if frame < GRIPPER_CLOSES_AT_FRAME else 0.0
            joint = 0.1 * max(0, frame - ARM_STARTS_MOVING_AT_FRAME)
            dataset.add_frame(
                {
                    "observation.state": np.array([joint, gripper], dtype=np.float32),
                    "action": np.array([joint, gripper], dtype=np.float32),
                    "observation.images.realsense": np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, np.uint8),
                    "observation.images.realsense_depth": np.full(
                        (IMAGE_HEIGHT, IMAGE_WIDTH, 1), 2000 - 100 * frame, np.uint16
                    ),
                    "task": "pull handle",
                }
            )
        dataset.save_episode()
    dataset.finalize()


def fake_segment_object_mask(segmenter, rgb_image, object_text_prompt, mask_score_threshold):
    mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    mask[8:24, 16:48] = True
    return mask


def fake_segment_object_masks(segmenter, rgb_images, object_text_prompt, mask_score_threshold):
    return [
        fake_segment_object_mask(segmenter, rgb_image, object_text_prompt, mask_score_threshold)
        for rgb_image in rgb_images
    ]


def test_label_dataset_writes_distance_stage_and_mask_features(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)

    label_dataset(make_config(tmp_path))

    labelled = LeRobotDataset("local/labelled", root=tmp_path / "labelled")
    frame_count = FRAMES_PER_EPISODE * EPISODE_COUNT
    expected_distance = np.tile(2.0 - 0.1 * np.arange(FRAMES_PER_EPISODE), EPISODE_COUNT)
    expected_stage = np.tile([0] * 6 + [1] * 4 + [2] * 2, EPISODE_COUNT)
    expected_mask = np.zeros((4, 6), dtype=np.uint8)
    expected_mask[0:2, 1:3] = 1

    assert tuple(labelled.meta.features["observation.object_distance"]["shape"]) == (1,)
    assert labelled.meta.features["observation.stage"]["dtype"] == "int64"
    assert tuple(labelled.meta.features["observation.object_mask.realsense"]["shape"]) == (4, 6)
    np.testing.assert_allclose(
        np.asarray(labelled.hf_dataset["observation.object_distance"]), expected_distance, atol=1e-3
    )
    np.testing.assert_array_equal(np.asarray(labelled.hf_dataset["observation.stage"]), expected_stage)
    masks = np.asarray(labelled.hf_dataset["observation.object_mask.realsense"])
    assert masks.shape == (frame_count, 4, 6)
    np.testing.assert_array_equal(masks[0], expected_mask)
    for key in ("observation.object_distance", "observation.stage", "observation.object_mask.realsense"):
        assert key in labelled.meta.stats


def test_label_dataset_keeps_the_depth_statistics_of_the_source(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)

    label_dataset(make_config(tmp_path))

    depth_key = "observation.images.realsense_depth"
    source_depth_stats = load_stats(tmp_path / "source")[depth_key]
    labelled_depth_stats = load_stats(tmp_path / "labelled")[depth_key]
    # The source depth is in millimeters, so a copy rescaled to meters would have a mean below 2.
    assert source_depth_stats["mean"].max() > 100
    for stat_name, source_value in source_depth_stats.items():
        np.testing.assert_allclose(labelled_depth_stats[stat_name], source_value)


def test_label_dataset_writes_debug_overlays_and_plots(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)
    debug_directory = tmp_path / "debug"

    label_dataset(
        make_config(tmp_path, debug_output_directory=str(debug_directory), debug_frames_per_episode=3)
    )

    assert len(list(debug_directory.glob("episode_*_frame_*.png"))) == 3 * EPISODE_COUNT
    assert len(list(debug_directory.glob("episode_*_labels.png"))) == EPISODE_COUNT


def stub_segmentation(monkeypatch) -> None:
    monkeypatch.setattr(label_script, "load_sam3_segmenter", lambda model_name_or_path, device: None)
    monkeypatch.setattr(label_script, "segment_object_masks", fake_segment_object_masks)


def load_labelled_dataset(tmp_path) -> LeRobotDataset:
    return LeRobotDataset("local/labelled", root=tmp_path / "labelled")


def label_feature_keys(dataset: LeRobotDataset) -> set[str]:
    label_keys = {"observation.object_distance", "observation.stage", "observation.object_mask.realsense"}
    return label_keys & set(dataset.meta.features)


def test_label_dataset_writes_only_the_object_distance_without_stage_thresholds(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)
    config = make_config(
        tmp_path,
        labels_to_write=["object_distance"],
        gripper_closed_threshold=None,
        grasp_distance_meters=None,
        pull_joint_displacement_threshold=None,
    )

    label_dataset(config)

    labelled = load_labelled_dataset(tmp_path)
    assert label_feature_keys(labelled) == {"observation.object_distance"}
    expected_distance = np.tile(2.0 - 0.1 * np.arange(FRAMES_PER_EPISODE), EPISODE_COUNT)
    np.testing.assert_allclose(
        np.asarray(labelled.hf_dataset["observation.object_distance"]), expected_distance, atol=1e-3
    )
    assert "observation.object_distance" in labelled.meta.stats
    assert "observation.stage" not in labelled.meta.stats


def test_label_dataset_writes_only_the_stage_and_still_uses_the_depth(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)

    label_dataset(make_config(tmp_path, labels_to_write=["stage"]))

    labelled = load_labelled_dataset(tmp_path)
    assert label_feature_keys(labelled) == {"observation.stage"}
    expected_stage = np.tile([0] * 6 + [1] * 4 + [2] * 2, EPISODE_COUNT)
    np.testing.assert_array_equal(np.asarray(labelled.hf_dataset["observation.stage"]), expected_stage)


def test_label_dataset_writes_only_the_object_mask_without_a_depth_feature(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)
    config = make_config(
        tmp_path,
        labels_to_write=["object_mask"],
        depth_image_key="observation.images.missing_depth",
        gripper_action_channel_name="missing",
    )

    label_dataset(config)

    labelled = load_labelled_dataset(tmp_path)
    assert label_feature_keys(labelled) == {"observation.object_mask.realsense"}


def test_label_dataset_logs_the_progress_of_every_episode(tmp_path, monkeypatch, caplog):
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)
    caplog.set_level(logging.INFO)

    label_dataset(make_config(tmp_path, labels_to_write=["object_mask"]))

    messages = [record.getMessage() for record in caplog.records]
    assert any("Labelling ['object_mask'] for 2 episodes (24 frames)" in message for message in messages)
    for episode_number in range(1, EPISODE_COUNT + 1):
        assert (
            f"Episode {episode_number}/{EPISODE_COUNT}: {FRAMES_PER_EPISODE} frames, 0 without an object detection"
            in messages
        )
    assert any(message.startswith("Writing ['observation.object_mask.realsense']") for message in messages)
    assert any(message.startswith("Wrote ['observation.object_mask.realsense']") for message in messages)


def make_segmentation_missing_on_calls(missing_calls: set[int]):
    # The frames are segmented in dataset order, so the call number is the global frame number.
    call_counter = itertools.count()

    def segment(segmenter, rgb_images, object_text_prompt, mask_score_threshold):
        masks = []
        for rgb_image in rgb_images:
            if next(call_counter) in missing_calls:
                masks.append(None)
            else:
                masks.append(
                    fake_segment_object_mask(segmenter, rgb_image, object_text_prompt, mask_score_threshold)
                )
        return masks

    return segment


def test_label_dataset_marks_the_frames_without_a_detection_as_missing(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    monkeypatch.setattr(label_script, "load_sam3_segmenter", lambda model_name_or_path, device: None)
    monkeypatch.setattr(label_script, "segment_object_masks", make_segmentation_missing_on_calls({3, 4}))

    label_dataset(make_config(tmp_path))

    labelled = load_labelled_dataset(tmp_path)
    expected_distance = np.tile(2.0 - 0.1 * np.arange(FRAMES_PER_EPISODE), EPISODE_COUNT)
    expected_distance[[3, 4]] = np.nan
    np.testing.assert_allclose(
        np.asarray(labelled.hf_dataset["observation.object_distance"]).reshape(-1),
        expected_distance,
        atol=1e-3,
        equal_nan=True,
    )
    masks = np.asarray(labelled.hf_dataset["observation.object_mask.realsense"])
    assert (masks[[3, 4]] == MISSING_MASK_CELL_VALUE).all()
    assert (masks[[0, 5]] != MISSING_MASK_CELL_VALUE).all()
    expected_stage = np.tile([0] * 6 + [1] * 4 + [2] * 2, EPISODE_COUNT)
    np.testing.assert_array_equal(
        np.asarray(labelled.hf_dataset["observation.stage"]).reshape(-1), expected_stage
    )
    frame_count = FRAMES_PER_EPISODE * EPISODE_COUNT
    distance_stats = labelled.meta.stats["observation.object_distance"]
    assert distance_stats["count"][0] == frame_count - 2
    assert np.isfinite(distance_stats["mean"]).all()
    assert labelled.meta.stats["observation.object_mask.realsense"]["max"].max() <= 1


def test_label_dataset_fills_the_missing_frames_when_not_marking_them(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    monkeypatch.setattr(label_script, "load_sam3_segmenter", lambda model_name_or_path, device: None)
    monkeypatch.setattr(label_script, "segment_object_masks", make_segmentation_missing_on_calls({3, 4}))

    label_dataset(make_config(tmp_path, mark_missing_labels=False))

    labelled = load_labelled_dataset(tmp_path)
    distance = np.asarray(labelled.hf_dataset["observation.object_distance"]).reshape(-1)
    np.testing.assert_allclose(distance[[3, 4]], distance[2], atol=1e-3)
    assert np.isfinite(distance).all()
    masks = np.asarray(labelled.hf_dataset["observation.object_mask.realsense"])
    assert (masks[[3, 4]] == 0).all()


def test_label_dataset_writes_a_distance_only_debug_plot(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)
    debug_directory = tmp_path / "debug"

    label_dataset(
        make_config(
            tmp_path,
            labels_to_write=["object_distance"],
            debug_output_directory=str(debug_directory),
            debug_frames_per_episode=2,
        )
    )

    assert len(list(debug_directory.glob("episode_*_labels.png"))) == EPISODE_COUNT


class FakeSegmentation:
    """Stands in for SAM 3 and counts the segmentations, the model loads and, optionally, fails on one."""

    def __init__(self, monkeypatch, fail_at_image: int | None = None):
        self.segmented_image_count = 0
        self.segmenter_load_count = 0
        self.fail_at_image = fail_at_image
        monkeypatch.setattr(label_script, "load_sam3_segmenter", self.load_segmenter)
        monkeypatch.setattr(label_script, "segment_object_masks", self.segment)

    def load_segmenter(self, model_name_or_path, device):
        self.segmenter_load_count += 1

    def segment(self, segmenter, rgb_images, object_text_prompt, mask_score_threshold):
        masks = []
        for rgb_image in rgb_images:
            if self.segmented_image_count == self.fail_at_image:
                raise RuntimeError("GPU lost")
            self.segmented_image_count += 1
            masks.append(
                fake_segment_object_mask(segmenter, rgb_image, object_text_prompt, mask_score_threshold)
            )
        return masks


def label_again(tmp_path, **overrides) -> dict[str, np.ndarray]:
    # The writer refuses an output path that exists, so a partial output has to be removed first.
    shutil.rmtree(tmp_path / "labelled", ignore_errors=True)
    label_dataset(make_config(tmp_path, **overrides))
    labelled = load_labelled_dataset(tmp_path)
    return {key: np.asarray(labelled.hf_dataset[key]) for key in sorted(label_feature_keys(labelled))}


def test_label_dataset_caches_each_episode_beside_the_dataset(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    FakeSegmentation(monkeypatch)

    label_dataset(make_config(tmp_path))

    cache_directory = tmp_path / "label_cache"
    assert sorted(path.name for path in cache_directory.iterdir()) == [
        "config.json",
        "episode_000.npz",
        "episode_001.npz",
    ]
    settings = json.loads((cache_directory / "config.json").read_text())
    assert settings["object_text_prompt"] == "car door handle"
    assert settings["sam_input_rotation_degrees"] == 0
    assert settings["mask_score_threshold"] == 0.2
    assert settings["mask_downsample_factor"] == 16
    assert settings["sam3_model_name_or_path"] == "facebook/sam3"


def test_label_dataset_writes_the_cache_to_the_given_directory(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    FakeSegmentation(monkeypatch)

    label_dataset(make_config(tmp_path, label_cache_directory=str(tmp_path / "elsewhere")))

    assert (tmp_path / "elsewhere" / "episode_000.npz").exists()
    assert not (tmp_path / "label_cache").exists()


def test_label_dataset_reruns_without_segmenting_when_every_episode_is_cached(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch)
    first_labels = label_again(tmp_path)
    segmented_image_count = fake_segmentation.segmented_image_count

    second_labels = label_again(tmp_path)

    assert segmented_image_count == FRAMES_PER_EPISODE * EPISODE_COUNT
    assert fake_segmentation.segmented_image_count == segmented_image_count
    assert fake_segmentation.segmenter_load_count == 1
    assert first_labels.keys() == second_labels.keys()
    for key in first_labels:
        np.testing.assert_array_equal(first_labels[key], second_labels[key])


def test_changing_a_stage_threshold_reuses_the_cached_segmentation(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch)
    first_labels = label_again(tmp_path)
    segmented_image_count = fake_segmentation.segmented_image_count

    second_labels = label_again(tmp_path, grasp_distance_meters=1.1)

    assert fake_segmentation.segmented_image_count == segmented_image_count
    np.testing.assert_array_equal(
        first_labels["observation.object_distance"], second_labels["observation.object_distance"]
    )
    np.testing.assert_array_equal(
        first_labels["observation.object_mask.realsense"], second_labels["observation.object_mask.realsense"]
    )
    assert not np.array_equal(first_labels["observation.stage"], second_labels["observation.stage"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"object_text_prompt": "latch"},
        {"sam_input_rotation_degrees": 90},
        {"mask_score_threshold": 0.3},
        {"mask_downsample_factor": 8},
        {"sam3_model_name_or_path": "other/sam3"},
        {"depth_image_key": "observation.images.realsense"},
    ],
)
def test_changing_a_segmentation_setting_segments_again(tmp_path, monkeypatch, overrides):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch)
    label_again(tmp_path, labels_to_write=["object_mask"])
    segmented_image_count = fake_segmentation.segmented_image_count

    label_again(tmp_path, labels_to_write=["object_mask"], **overrides)

    assert fake_segmentation.segmented_image_count == 2 * segmented_image_count


def test_a_crash_keeps_the_finished_episodes_cached(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch, fail_at_image=FRAMES_PER_EPISODE + 2)
    with pytest.raises(RuntimeError, match="GPU lost"):
        label_dataset(make_config(tmp_path))
    cache_directory = tmp_path / "label_cache"
    assert (cache_directory / "episode_000.npz").exists()
    assert not (cache_directory / "episode_001.npz").exists()
    assert not list(cache_directory.glob("*.tmp"))

    fake_segmentation.fail_at_image = None
    segmented_image_count = fake_segmentation.segmented_image_count
    labels = label_again(tmp_path)

    assert fake_segmentation.segmented_image_count - segmented_image_count == FRAMES_PER_EPISODE
    assert len(labels["observation.stage"]) == FRAMES_PER_EPISODE * EPISODE_COUNT


def test_an_unreadable_cache_file_is_segmented_again(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch)
    first_labels = label_again(tmp_path)
    (tmp_path / "label_cache" / "episode_000.npz").write_bytes(b"not a zip file")
    segmented_image_count = fake_segmentation.segmented_image_count

    second_labels = label_again(tmp_path)

    assert fake_segmentation.segmented_image_count - segmented_image_count == FRAMES_PER_EPISODE
    for key in first_labels:
        np.testing.assert_array_equal(first_labels[key], second_labels[key])


def test_a_cache_of_another_frame_count_is_segmented_again(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch)
    label_again(tmp_path)
    cache = label_script.SegmentationCache(tmp_path / "label_cache", make_config(tmp_path))
    cache.save(
        0,
        label_script.EpisodeSegmentation(
            raw_distances=np.zeros(FRAMES_PER_EPISODE + 1),
            object_masks=np.zeros((FRAMES_PER_EPISODE + 1, 4, 6), dtype=np.uint8),
            object_found=np.ones(FRAMES_PER_EPISODE + 1, dtype=bool),
            distances_measured=True,
        ),
    )
    segmented_image_count = fake_segmentation.segmented_image_count

    label_again(tmp_path)

    assert fake_segmentation.segmented_image_count - segmented_image_count == FRAMES_PER_EPISODE


def test_a_cache_without_distances_is_segmented_again_when_the_distances_are_needed(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch)
    label_again(tmp_path, labels_to_write=["object_mask"])
    segmented_image_count = fake_segmentation.segmented_image_count

    labels = label_again(tmp_path)

    assert fake_segmentation.segmented_image_count == 2 * segmented_image_count
    assert "observation.object_distance" in labels


def test_a_cache_with_distances_serves_a_mask_only_run(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    fake_segmentation = FakeSegmentation(monkeypatch)
    label_again(tmp_path)
    segmented_image_count = fake_segmentation.segmented_image_count

    labels = label_again(tmp_path, labels_to_write=["object_mask"])

    assert fake_segmentation.segmented_image_count == segmented_image_count
    assert set(labels) == {"observation.object_mask.realsense"}


def test_the_cached_missing_detections_are_marked_or_filled_as_configured(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    monkeypatch.setattr(label_script, "load_sam3_segmenter", lambda model_name_or_path, device: None)
    monkeypatch.setattr(label_script, "segment_object_masks", make_segmentation_missing_on_calls({3, 4}))
    marked = label_again(tmp_path)
    monkeypatch.setattr(label_script, "segment_object_masks", None)

    filled = label_again(tmp_path, mark_missing_labels=False)

    assert (marked["observation.object_mask.realsense"][[3, 4]] == MISSING_MASK_CELL_VALUE).all()
    assert np.isnan(marked["observation.object_distance"][[3, 4]]).all()
    assert (filled["observation.object_mask.realsense"][[3, 4]] == 0).all()
    assert np.isfinite(filled["observation.object_distance"]).all()


def test_label_dataset_raises_when_no_frame_has_the_object(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    monkeypatch.setattr(label_script, "load_sam3_segmenter", lambda model_name_or_path, device: None)
    monkeypatch.setattr(
        label_script,
        "segment_object_masks",
        lambda segmenter, rgb_images, object_text_prompt, mask_score_threshold: [None] * len(rgb_images),
    )

    with pytest.raises(ValueError, match="Episode 0"):
        label_dataset(make_config(tmp_path))


def test_label_dataset_rejects_unknown_gripper_channel(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    monkeypatch.setattr(label_script, "load_sam3_segmenter", lambda model_name_or_path, device: None)

    with pytest.raises(ValueError, match="gripper_action_channel_name"):
        label_dataset(make_config(tmp_path, gripper_action_channel_name="missing"))


@pytest.mark.parametrize(
    ("shape", "names", "expected"),
    [
        ([480, 640, 3], ["height", "width", "channels"], (480, 640)),
        ([480, 640, 1], ["height", "width", "channel"], (480, 640)),
        ([3, 480, 640], ["channel", "height", "width"], (480, 640)),
        ([1, 256, 320], ["channel", "height", "width"], (256, 320)),
    ],
)
def test_get_image_height_and_width_supports_both_layouts(shape, names, expected):
    assert get_image_height_and_width({"shape": shape, "names": names}) == expected


def make_dataset_with_image_features(rgb_feature: dict, depth_feature: dict) -> SimpleNamespace:
    action_feature = {"shape": [2], "names": ["joint1.position", "gripper.position.x"]}
    return SimpleNamespace(
        meta=SimpleNamespace(
            features={"rgb": rgb_feature, "depth": depth_feature, "action": action_feature},
            depth_keys=["depth"],
            camera_keys=["rgb", "depth"],
        )
    )


def test_validate_dataset_features_accepts_channel_first_cameras(tmp_path):
    channel_first = ["channel", "height", "width"]
    dataset = make_dataset_with_image_features(
        {"shape": [3, 480, 640], "names": channel_first}, {"shape": [1, 480, 640], "names": channel_first}
    )
    config = make_config(tmp_path, rgb_image_key="rgb", depth_image_key="depth")

    validate_dataset_features(dataset, config)


def test_validate_dataset_features_rejects_a_missing_gripper_state_channel(tmp_path):
    channel_first = ["channel", "height", "width"]
    dataset = make_dataset_with_image_features(
        {"shape": [3, 480, 640], "names": channel_first}, {"shape": [1, 480, 640], "names": channel_first}
    )
    dataset.meta.features["observation.state"] = {"shape": [1], "names": ["joint1.position"]}
    config = make_config(
        tmp_path,
        rgb_image_key="rgb",
        depth_image_key="depth",
        gripper_still_frames=3,
        gripper_still_speed_threshold=0.001,
    )

    with pytest.raises(ValueError, match="gripper_state_channel_name"):
        validate_dataset_features(dataset, config)


def test_label_dataset_starts_the_grasp_when_the_gripper_has_stopped(tmp_path, monkeypatch):
    write_source_dataset(tmp_path / "source")
    stub_segmentation(monkeypatch)

    label_dataset(make_config(tmp_path, gripper_still_frames=3, gripper_still_speed_threshold=0.001))

    labelled = load_labelled_dataset(tmp_path)
    # The measured gripper steps from open to closed at frame 6 and is still from frame 7 on.
    expected_stage = np.tile([0] * 7 + [1] * 3 + [2] * 2, EPISODE_COUNT)
    np.testing.assert_array_equal(
        np.asarray(labelled.hf_dataset["observation.stage"]).reshape(-1), expected_stage
    )


def test_validate_dataset_features_rejects_depth_of_another_size(tmp_path):
    channel_first = ["channel", "height", "width"]
    dataset = make_dataset_with_image_features(
        {"shape": [3, 480, 640], "names": channel_first}, {"shape": [1, 240, 320], "names": channel_first}
    )
    config = make_config(tmp_path, rgb_image_key="rgb", depth_image_key="depth")

    with pytest.raises(ValueError, match="aligned"):
        validate_dataset_features(dataset, config)


def test_median_depth_under_mask_ignores_invalid_depth():
    depth = np.array([[1.0, 0.0, np.nan], [3.0, 5.0, 100.0]])
    mask = np.array([[True, True, True], [True, True, False]])

    assert median_depth_under_mask(depth, mask) == 3.0


def test_median_depth_under_mask_without_valid_depth_is_none():
    depth = np.zeros((2, 2))

    assert median_depth_under_mask(depth, np.ones((2, 2), dtype=bool)) is None


def test_fill_missing_distances_fills_forward_and_back_fills_leading_gap():
    distances = np.array([np.nan, np.nan, 2.0, np.nan, 3.0, np.nan])

    np.testing.assert_array_equal(fill_missing_distances(distances), [2.0, 2.0, 2.0, 2.0, 3.0, 3.0])


def test_fill_missing_distances_requires_a_valid_distance():
    with pytest.raises(ValueError, match="At least one valid distance"):
        fill_missing_distances(np.full(3, np.nan))


def test_median_filter_distances_removes_spikes_and_keeps_length():
    distances = np.array([1.0, 1.0, 9.0, 1.0, 1.0])

    np.testing.assert_array_equal(median_filter_distances(distances, 3), np.ones(5))
    np.testing.assert_array_equal(median_filter_distances(distances, 1), distances)


def test_downsample_mask_uses_block_majority_and_pads_partial_blocks():
    mask = np.zeros((5, 4), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4, 0:4] = True

    downsampled = downsample_mask(mask, 2)

    assert downsampled.shape == (3, 2)
    np.testing.assert_array_equal(downsampled, [[1, 0], [0, 0], [1, 1]])


def test_assign_stage_labels_follows_grasp_then_pull():
    gripper_commands = np.array([0.04, 0.04, 0.0, 0.0, 0.0, 0.0])
    object_distances = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    joint_positions = np.array([[0.0], [0.0], [0.0], [0.1], [0.3], [0.3]])

    stage = assign_stage_labels(gripper_commands, object_distances, joint_positions, THRESHOLDS)

    np.testing.assert_array_equal(
        stage, [APPROACH_STAGE, APPROACH_STAGE, GRASP_STAGE, GRASP_STAGE, PULL_STAGE, PULL_STAGE]
    )


STILL_GRIPPER_THRESHOLDS = StageThresholds(
    gripper_closed_threshold=0.03,
    grasp_distance_meters=GRASP_DISTANCE_METERS,
    pull_joint_displacement_threshold=0.15,
    gripper_still_frames=3,
    gripper_still_speed_threshold=0.0005,
)


def test_assign_stage_labels_starts_the_grasp_when_the_gripper_has_stopped_closing():
    gripper_commands = np.array([0.044] * 3 + [0.026] * 7)
    gripper_positions = np.array([0.044, 0.044, 0.040, 0.034, 0.028, 0.0257, 0.0257, 0.0257, 0.0257, 0.0257])
    object_distances = np.full(10, 1.0)
    joint_positions = np.zeros((10, 1))

    still_stage = assign_stage_labels(
        gripper_commands, object_distances, joint_positions, STILL_GRIPPER_THRESHOLDS, gripper_positions
    )
    command_stage = assign_stage_labels(
        gripper_commands,
        object_distances,
        joint_positions,
        dataclasses.replace(STILL_GRIPPER_THRESHOLDS, gripper_still_frames=0),
    )

    np.testing.assert_array_equal(still_stage, [APPROACH_STAGE] * 6 + [GRASP_STAGE] * 4)
    np.testing.assert_array_equal(command_stage, [APPROACH_STAGE] * 3 + [GRASP_STAGE] * 7)


def test_assign_stage_labels_without_a_still_gripper_stays_in_approach():
    gripper_positions = np.linspace(0.044, 0.020, 12)
    gripper_commands = np.full(12, 0.02)

    stage = assign_stage_labels(
        gripper_commands, np.full(12, 1.0), np.zeros((12, 1)), STILL_GRIPPER_THRESHOLDS, gripper_positions
    )

    np.testing.assert_array_equal(stage, [APPROACH_STAGE] * 12)


def test_assign_stage_labels_with_a_still_gripper_still_requires_the_object_to_be_near():
    gripper_positions = np.array([0.044] * 3 + [0.0257] * 7)

    stage = assign_stage_labels(
        np.full(10, 0.02),
        np.full(10, GRASP_DISTANCE_METERS + 1.0),
        np.zeros((10, 1)),
        STILL_GRIPPER_THRESHOLDS,
        gripper_positions,
    )

    np.testing.assert_array_equal(stage, [APPROACH_STAGE] * 10)


def test_assign_stage_labels_latch_when_the_gripper_reopens():
    gripper_commands = np.array([0.0, 0.0, 0.04, 0.04])
    object_distances = np.full(4, 1.0)

    stage = assign_stage_labels(gripper_commands, object_distances, np.zeros((4, 1)), THRESHOLDS)

    np.testing.assert_array_equal(stage, [GRASP_STAGE] * 4)


def test_assign_stage_labels_without_grasp_is_all_approach():
    gripper_commands = np.full(4, 0.04)

    stage = assign_stage_labels(gripper_commands, np.full(4, 1.0), np.zeros((4, 1)), THRESHOLDS)

    np.testing.assert_array_equal(stage, [APPROACH_STAGE] * 4)


def test_find_position_channel_indices_prefers_position_channels():
    names = ["a.position", "a.velocity", "b.position", "b.effort"]

    assert find_position_channel_indices(names) == [0, 2]
    assert find_position_channel_indices(["x", "y"]) == [0, 1]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"distance_median_filter_frames": 4}, "distance_median_filter_frames"),
        ({"mask_downsample_factor": 0}, "mask_downsample_factor"),
        ({"mask_score_threshold": 1.5}, "mask_score_threshold"),
        ({"sam_input_rotation_degrees": 45}, "sam_input_rotation_degrees"),
        ({"sam_batch_size": 0}, "sam_batch_size"),
        ({"debug_output_directory": "d", "debug_frames_per_episode": 0}, "debug_frames_per_episode"),
        ({"gripper_still_frames": 3}, "gripper_still_speed_threshold"),
        ({"gripper_still_frames": -1}, "gripper_still_frames"),
        ({"labels_to_write": []}, "labels_to_write"),
        ({"labels_to_write": ["object_distance", "object_distance"]}, "labels_to_write"),
        ({"labels_to_write": ["object_distance", "depth"]}, "labels_to_write"),
        ({"labels_to_write": ["stage"], "grasp_distance_meters": None}, "grasp_distance_meters"),
    ],
)
def test_validate_config_rejects_invalid_values(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_config(make_config(tmp_path, **overrides))


def test_rotate_clockwise_turns_a_quarter_and_a_negative_angle_turns_back():
    image = np.arange(6).reshape(2, 3)
    rotated = rotate_clockwise(image, 90)
    np.testing.assert_array_equal(rotated, [[3, 0], [4, 1], [5, 2]])
    np.testing.assert_array_equal(rotate_clockwise(rotated, -90), image)
    np.testing.assert_array_equal(rotate_clockwise(image, 0), image)
    with pytest.raises(ValueError, match="multiple of 90"):
        rotate_clockwise(image, 45)


def test_measure_objects_in_frames_segments_the_turned_image_and_turns_the_mask_back(tmp_path, monkeypatch):
    config = make_config(tmp_path, labels_to_write=["object_mask"], sam_input_rotation_degrees=90)
    image = torch.rand(3, IMAGE_HEIGHT, IMAGE_WIDTH)
    turned_mask = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT), dtype=bool)
    turned_mask[:10, :20] = True
    segmented_images = []

    def fake_segment_turned_images(segmenter, rgb_images, object_text_prompt, mask_score_threshold):
        segmented_images.extend(rgb_images)
        return [turned_mask]

    monkeypatch.setattr(label_script, "segment_object_masks", fake_segment_turned_images)
    (measurement,) = measure_objects_in_frames(None, [{config.rgb_image_key: image}], config)

    np.testing.assert_array_equal(segmented_images[0], np.rot90(to_uint8_rgb_image(image), k=-1))
    np.testing.assert_array_equal(measurement.mask, np.rot90(turned_mask, k=1))
    assert measurement.mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert measurement.object_found


class FakeInputs(dict):
    def to(self, device):
        return self


class RecordingModel:
    def __init__(self):
        self.call_count = 0

    def __call__(self, **inputs):
        self.call_count += 1
        return None


class FakeProcessor:
    def __init__(self, scores: torch.Tensor, masks: torch.Tensor):
        self.scores = scores
        self.masks = masks
        self.last_text = None

    def __call__(self, images, text, return_tensors):
        batch_images = images if isinstance(images, list) else [images]
        self.last_text = text
        return FakeInputs(
            original_sizes=torch.tensor([[image.height, image.width] for image in batch_images])
        )

    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        return [{"scores": self.scores, "masks": self.masks} for _ in target_sizes]


def test_segment_object_mask_returns_the_highest_scoring_instance():
    masks = torch.zeros(2, 4, 4, dtype=torch.bool)
    masks[1, 1:3, 1:3] = True
    segmenter = Sam3Segmenter(
        model=lambda **inputs: None,
        processor=FakeProcessor(torch.tensor([0.5, 0.9]), masks),
        device="cpu",
    )

    mask = segment_object_mask(segmenter, np.zeros((4, 4, 3), dtype=np.uint8), "door handle", 0.4)

    np.testing.assert_array_equal(mask, masks[1].numpy())


def test_segment_object_mask_without_instances_is_none():
    segmenter = Sam3Segmenter(
        model=lambda **inputs: None,
        processor=FakeProcessor(torch.zeros(0), torch.zeros(0, 4, 4, dtype=torch.bool)),
        device="cpu",
    )

    assert segment_object_mask(segmenter, np.zeros((4, 4, 3), dtype=np.uint8), "door handle", 0.4) is None


def test_segment_object_masks_forwards_the_batch_once():
    masks = torch.zeros(1, 4, 4, dtype=torch.bool)
    masks[0, 0, 0] = True
    model = RecordingModel()
    processor = FakeProcessor(torch.tensor([0.9]), masks)
    segmenter = Sam3Segmenter(model=model, processor=processor, device="cpu")
    images = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]

    results = segment_object_masks(segmenter, images, "door handle", 0.4)

    assert model.call_count == 1
    assert processor.last_text == ["door handle", "door handle"]
    assert len(results) == 2
    np.testing.assert_array_equal(results[0], masks[0].numpy())
    np.testing.assert_array_equal(results[1], masks[0].numpy())
