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
"""Label a LeRobot dataset with the object distance, stage and object mask used by `kinetic_act`.

Requires: pip install 'lerobot[transformers-dep]' and, for the distance and stage labels, a depth feature
aligned to the RGB image. Choose the labels with `--labels_to_write`; the stage thresholds are needed
only for the stage label. Frames where SAM 3 finds no object are marked as missing labels that `kinetic_act`
leaves out of its losses, unless `--mark_missing_labels false` fills them instead. `--sam_input_rotation_degrees`
turns the image of a sideways-mounted camera upright before SAM 3 segments it.

The SAM 3 result of every finished episode is cached beside the dataset, so a rerun after a crash or after
changing a stage threshold skips the episodes whose cache matches the segmentation settings.

Example:
```
lerobot-label-object-task \
    --repo_id local/car_door \
    --root /path/to/car_door \
    --new_repo_id local/car_door_labelled \
    --gripper_closed_threshold 0.02 \
    --grasp_distance_meters 0.5 \
    --pull_joint_displacement_threshold 0.05 \
    --debug_output_directory /tmp/label_debug
```
"""

import functools
import json
import logging
import math
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.joint_layout import feature_channel_names
from lerobot.policies.kinetic_act.configuration_kinetic_act import MISSING_MASK_CELL_VALUE
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package
from lerobot.utils.utils import init_logging

if TYPE_CHECKING or _transformers_available:
    from transformers import Sam3Model, Sam3Processor
else:
    Sam3Model = None
    Sam3Processor = None

OBJECT_DISTANCE_LABEL = "object_distance"
STAGE_LABEL = "stage"
OBJECT_MASK_LABEL = "object_mask"
ALL_LABELS = (OBJECT_DISTANCE_LABEL, STAGE_LABEL, OBJECT_MASK_LABEL)
STAGE_THRESHOLD_FIELD_NAMES = (
    "gripper_closed_threshold",
    "grasp_distance_meters",
    "pull_joint_displacement_threshold",
)
OBJECT_DISTANCE_FEATURE_KEY = "observation.object_distance"
STAGE_FEATURE_KEY = "observation.stage"
OBJECT_MASK_FEATURE_KEY_PREFIX = "observation.object_mask"
MASK_OVERLAY_COLOR = np.array([255, 0, 0], dtype=np.float32)
MASK_OVERLAY_OPACITY = 0.5
APPROACH_STAGE = 0
GRASP_STAGE = 1
PULL_STAGE = 2


@dataclass
class StageThresholds:
    gripper_closed_threshold: float
    grasp_distance_meters: float
    pull_joint_displacement_threshold: float
    gripper_still_frames: int = 0
    gripper_still_speed_threshold: float | None = None


@dataclass
class StageChannelIndices:
    gripper_command: int
    gripper_state: int | None
    joint_positions: list[int]


@dataclass
class LabelObjectTaskConfig:
    repo_id: str
    new_repo_id: str
    root: str | None = None
    new_root: str | None = None
    labels_to_write: list[str] = field(default_factory=lambda: list(ALL_LABELS))
    gripper_closed_threshold: float | None = None
    grasp_distance_meters: float | None = None
    pull_joint_displacement_threshold: float | None = None
    rgb_image_key: str = f"{OBS_IMAGES}.realsense"
    depth_image_key: str = f"{OBS_IMAGES}.realsense_depth"
    object_text_prompt: str = "car door handle"
    sam3_model_name_or_path: str = "facebook/sam3"
    # Clockwise, a multiple of 90. The image is turned before SAM 3 and the mask is turned back.
    sam_input_rotation_degrees: int = 0
    mask_score_threshold: float = 0.2
    mask_downsample_factor: int = 16
    distance_median_filter_frames: int = 5
    mark_missing_labels: bool = True
    gripper_action_channel_name: str = "gripper.position.x"
    gripper_state_channel_name: str | None = None
    gripper_still_frames: int = 0
    gripper_still_speed_threshold: float | None = None
    debug_output_directory: str | None = None
    label_cache_directory: str | None = None
    debug_frames_per_episode: int = 6
    sam_batch_size: int = 8
    device: str = "cuda"

    @property
    def writes_object_distance(self) -> bool:
        return OBJECT_DISTANCE_LABEL in self.labels_to_write

    @property
    def writes_stage(self) -> bool:
        return STAGE_LABEL in self.labels_to_write

    @property
    def writes_object_mask(self) -> bool:
        return OBJECT_MASK_LABEL in self.labels_to_write

    @property
    def needs_object_distance(self) -> bool:
        return self.writes_object_distance or self.writes_stage

    @property
    def requires_gripper_state(self) -> bool:
        return self.writes_stage and self.gripper_still_frames > 0

    @property
    def resolved_gripper_state_channel_name(self) -> str:
        return self.gripper_state_channel_name or self.gripper_action_channel_name

    @property
    def stage_thresholds(self) -> StageThresholds:
        missing_names = [name for name in STAGE_THRESHOLD_FIELD_NAMES if getattr(self, name) is None]
        if missing_names:
            raise ValueError(f"The stage label requires {missing_names}.")
        return StageThresholds(
            gripper_closed_threshold=self.gripper_closed_threshold,
            grasp_distance_meters=self.grasp_distance_meters,
            pull_joint_displacement_threshold=self.pull_joint_displacement_threshold,
            gripper_still_frames=self.gripper_still_frames,
            gripper_still_speed_threshold=self.gripper_still_speed_threshold,
        )

    @property
    def object_mask_feature_key(self) -> str:
        camera_name = self.rgb_image_key.removeprefix(f"{OBS_IMAGES}.")
        return f"{OBJECT_MASK_FEATURE_KEY_PREFIX}.{camera_name}"


@dataclass
class Sam3Segmenter:
    model: "Sam3Model"
    processor: "Sam3Processor"
    device: str


@dataclass
class ObjectMeasurement:
    rgb_image: np.ndarray
    mask: np.ndarray
    distance: float | None
    object_found: bool


@dataclass
class EpisodeSegmentation:
    raw_distances: np.ndarray
    object_masks: np.ndarray
    object_found: np.ndarray
    distances_measured: bool


@dataclass
class StageInputs:
    gripper_commands: np.ndarray
    gripper_positions: np.ndarray | None
    joint_positions: np.ndarray


class SegmentationCache:
    SETTINGS_FILE_NAME = "config.json"

    def __init__(self, directory: Path, config: LabelObjectTaskConfig):
        self.directory = directory
        self.needs_object_distance = config.needs_object_distance
        self.settings = {
            "repo_id": config.repo_id,
            "rgb_image_key": config.rgb_image_key,
            "depth_image_key": config.depth_image_key,
            "object_text_prompt": config.object_text_prompt,
            "sam3_model_name_or_path": config.sam3_model_name_or_path,
            "sam_input_rotation_degrees": config.sam_input_rotation_degrees,
            "mask_score_threshold": config.mask_score_threshold,
            "mask_downsample_factor": config.mask_downsample_factor,
        }

    def write_settings(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        settings_path = self.directory / self.SETTINGS_FILE_NAME
        temporary_path = settings_path.with_name(settings_path.name + ".tmp")
        temporary_path.write_text(json.dumps(self.settings, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_path, settings_path)

    def load(self, episode_index: int, frame_count: int) -> EpisodeSegmentation | None:
        episode_path = self.episode_path(episode_index)
        if not episode_path.exists():
            return None
        try:
            with np.load(episode_path, allow_pickle=False) as cached:
                if str(cached["settings_json"]) != self.settings_json():
                    logging.info("Episode %d: the cache has other segmentation settings", episode_index)
                    return None
                segmentation = EpisodeSegmentation(
                    raw_distances=cached["raw_distances"],
                    object_masks=cached["object_masks"],
                    object_found=cached["object_found"],
                    distances_measured=bool(cached["distances_measured"]),
                )
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
            logging.warning(
                "Episode %d: ignoring the unreadable cache %s (%s)", episode_index, episode_path, error
            )
            return None
        if len(segmentation.raw_distances) != frame_count:
            logging.info("Episode %d: the cache has another frame count", episode_index)
            return None
        if self.needs_object_distance and not segmentation.distances_measured:
            logging.info("Episode %d: the cache has no object distances", episode_index)
            return None
        return segmentation

    def save(self, episode_index: int, segmentation: EpisodeSegmentation) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        episode_path = self.episode_path(episode_index)
        temporary_path = episode_path.with_name(episode_path.name + ".tmp")
        with open(temporary_path, "wb") as temporary_file:
            np.savez(
                temporary_file,
                settings_json=np.array(self.settings_json()),
                raw_distances=segmentation.raw_distances,
                object_masks=segmentation.object_masks,
                object_found=segmentation.object_found,
                distances_measured=np.array(segmentation.distances_measured),
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, episode_path)

    def episode_path(self, episode_index: int) -> Path:
        return self.directory / f"episode_{episode_index:03d}.npz"

    def settings_json(self) -> str:
        return json.dumps(self.settings, sort_keys=True)


@parser.wrap()
def label_dataset(config: LabelObjectTaskConfig) -> None:
    validate_config(config)
    dataset = LeRobotDataset(config.repo_id, root=config.root, depth_output_unit="m")
    validate_dataset_features(dataset, config)
    logging.info(
        "Labelling %s for %d episodes (%d frames) of %s",
        config.labels_to_write,
        dataset.meta.total_episodes,
        dataset.meta.total_frames,
        config.repo_id,
    )
    segmentation_cache = SegmentationCache(resolve_label_cache_directory(dataset, config), config)
    segmentation_cache.write_settings()
    logging.info("Segmentation cache: %s", segmentation_cache.directory)

    # SAM 3 loads on the first episode that is not cached, so a fully cached rerun never touches the GPU.
    @functools.cache
    def get_segmenter() -> Sam3Segmenter:
        segmenter = load_sam3_segmenter(config.sam3_model_name_or_path, config.device)
        logging.info("Loaded SAM 3 from %s on %s", config.sam3_model_name_or_path, config.device)
        return segmenter

    with (
        logging_redirect_tqdm(),
        tqdm(total=dataset.meta.total_frames, desc="Labelling frames", unit="frame") as progress_bar,
    ):
        label_arrays_per_episode = [
            label_episode(dataset, episode_index, get_segmenter, config, progress_bar, segmentation_cache)
            for episode_index in range(dataset.meta.total_episodes)
        ]
    write_label_features(dataset, label_arrays_per_episode, config)


def resolve_label_cache_directory(dataset: LeRobotDataset, config: LabelObjectTaskConfig) -> Path:
    if config.label_cache_directory is not None:
        return Path(config.label_cache_directory)
    return Path(dataset.root).parent / "label_cache"


def validate_config(config: LabelObjectTaskConfig) -> None:
    unknown_labels = [label for label in config.labels_to_write if label not in ALL_LABELS]
    if (
        not config.labels_to_write
        or unknown_labels
        or len(set(config.labels_to_write)) != len(config.labels_to_write)
    ):
        raise ValueError(
            f"`labels_to_write` must be a non-empty list of distinct labels from {list(ALL_LABELS)}. "
            f"Got {config.labels_to_write}."
        )
    if config.writes_stage:
        missing_names = [name for name in STAGE_THRESHOLD_FIELD_NAMES if getattr(config, name) is None]
        if missing_names:
            raise ValueError(f"Writing the stage label requires {missing_names}.")
    if config.gripper_still_frames < 0:
        raise ValueError(f"`gripper_still_frames` must not be negative. Got {config.gripper_still_frames}.")
    if config.requires_gripper_state and (
        config.gripper_still_speed_threshold is None or config.gripper_still_speed_threshold <= 0
    ):
        raise ValueError(
            "`gripper_still_speed_threshold` must be positive when `gripper_still_frames` is set. "
            f"Got {config.gripper_still_speed_threshold}."
        )
    if config.distance_median_filter_frames < 1 or config.distance_median_filter_frames % 2 == 0:
        raise ValueError(
            f"`distance_median_filter_frames` must be a positive odd number. "
            f"Got {config.distance_median_filter_frames}."
        )
    if config.mask_downsample_factor < 1:
        raise ValueError(f"`mask_downsample_factor` must be positive. Got {config.mask_downsample_factor}.")
    if config.sam_batch_size < 1:
        raise ValueError(f"`sam_batch_size` must be positive. Got {config.sam_batch_size}.")
    if not 0.0 < config.mask_score_threshold < 1.0:
        raise ValueError(f"`mask_score_threshold` must be in (0, 1). Got {config.mask_score_threshold}.")
    if config.sam_input_rotation_degrees % 90 != 0:
        raise ValueError(
            f"`sam_input_rotation_degrees` must be a multiple of 90. Got {config.sam_input_rotation_degrees}."
        )
    if config.debug_output_directory is not None and config.debug_frames_per_episode < 1:
        raise ValueError(
            f"`debug_frames_per_episode` must be positive. Got {config.debug_frames_per_episode}."
        )


def validate_dataset_features(dataset: LeRobotDataset, config: LabelObjectTaskConfig) -> None:
    features = dataset.meta.features
    if config.rgb_image_key not in features:
        raise ValueError(
            f"`{config.rgb_image_key}` is not a dataset feature. Cameras: {dataset.meta.camera_keys}."
        )
    if config.needs_object_distance:
        validate_depth_feature(dataset, config)
    if config.writes_stage:
        action_names = feature_channel_names(features[ACTION])
        if config.gripper_action_channel_name not in action_names:
            raise ValueError(
                f"`gripper_action_channel_name` {config.gripper_action_channel_name!r} is not an action "
                f"channel. Channels: {action_names}."
            )
    if config.requires_gripper_state:
        state_names = feature_channel_names(features[OBS_STATE])
        if config.resolved_gripper_state_channel_name not in state_names:
            raise ValueError(
                f"`gripper_state_channel_name` {config.resolved_gripper_state_channel_name!r} is not a state "
                f"channel. Channels: {state_names}."
            )


def validate_depth_feature(dataset: LeRobotDataset, config: LabelObjectTaskConfig) -> None:
    features = dataset.meta.features
    if config.depth_image_key not in features:
        raise ValueError(
            f"`{config.depth_image_key}` is not a dataset feature. Cameras: {dataset.meta.camera_keys}."
        )
    if config.depth_image_key not in dataset.meta.depth_keys:
        raise ValueError(
            f"`{config.depth_image_key}` is not a depth feature. Depth: {dataset.meta.depth_keys}."
        )
    rgb_size = get_image_height_and_width(features[config.rgb_image_key])
    depth_size = get_image_height_and_width(features[config.depth_image_key])
    if rgb_size != depth_size:
        raise ValueError(
            f"The depth image {depth_size} must be aligned to the RGB image {rgb_size}: "
            "same height and width."
        )


def get_image_height_and_width(feature: dict) -> tuple[int, int]:
    shape = tuple(feature["shape"])
    names = feature.get("names") or []
    # Same rule as `dataset_to_policy_features`: height, width, channel when the last name is the channel,
    # channel, height, width otherwise.
    is_channel_last = len(names) == 3 and names[2] in ("channel", "channels")
    return (shape[0], shape[1]) if is_channel_last else (shape[1], shape[2])


def load_sam3_segmenter(model_name_or_path: str, device: str) -> Sam3Segmenter:
    require_package("transformers", extra="transformers-dep")
    model = Sam3Model.from_pretrained(model_name_or_path).to(device).eval()
    processor = Sam3Processor.from_pretrained(model_name_or_path)
    return Sam3Segmenter(model=model, processor=processor, device=device)


def label_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    get_segmenter: Callable[[], Sam3Segmenter],
    config: LabelObjectTaskConfig,
    progress_bar: tqdm,
    segmentation_cache: SegmentationCache,
) -> dict[str, np.ndarray]:
    episode = dataset.meta.episodes[episode_index]
    frame_indices = list(range(episode["dataset_from_index"], episode["dataset_to_index"]))

    segmentation = segmentation_cache.load(episode_index, len(frame_indices))
    is_cached = segmentation is not None
    if segmentation is None:
        segmentation = segment_episode(
            dataset, episode_index, frame_indices, get_segmenter(), config, progress_bar
        )
        segmentation_cache.save(episode_index, segmentation)
    else:
        progress_bar.update(len(frame_indices))

    progress_bar.set_postfix(episode=f"{episode_index + 1}/{dataset.meta.total_episodes}")
    logging.info(
        "Episode %d/%d: %d frames, %d without an object detection%s",
        episode_index + 1,
        dataset.meta.total_episodes,
        len(frame_indices),
        int((~segmentation.object_found).sum()),
        " (cached)" if is_cached else "",
    )
    label_arrays: dict[str, np.ndarray] = {}
    if config.writes_object_mask:
        label_arrays[config.object_mask_feature_key] = mark_missing_object_masks(segmentation, config)
    if config.needs_object_distance:
        stage_inputs = read_stage_inputs(dataset, frame_indices, config) if config.writes_stage else None
        label_arrays.update(
            label_object_distance_and_stage(episode_index, segmentation.raw_distances, stage_inputs, config)
        )
    return label_arrays


def segment_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    frame_indices: list[int],
    segmenter: Sam3Segmenter,
    config: LabelObjectTaskConfig,
    progress_bar: tqdm,
) -> EpisodeSegmentation:
    debug_frame_positions = find_debug_frame_positions(len(frame_indices), config)
    raw_distances = np.full(len(frame_indices), np.nan)
    object_found = np.zeros(len(frame_indices), dtype=bool)
    object_masks = []
    for batch_start in range(0, len(frame_indices), config.sam_batch_size):
        batch_frame_indices = frame_indices[batch_start : batch_start + config.sam_batch_size]
        frames = [dataset[frame_index] for frame_index in batch_frame_indices]
        measurements = measure_objects_in_frames(segmenter, frames, config)
        for offset, measurement in enumerate(measurements):
            frame_position = batch_start + offset
            if measurement.distance is not None:
                raw_distances[frame_position] = measurement.distance
            object_found[frame_position] = measurement.object_found
            object_masks.append(downsample_mask(measurement.mask, config.mask_downsample_factor))
            if frame_position in debug_frame_positions:
                overlay_path = Path(config.debug_output_directory) / (
                    f"episode_{episode_index:03d}_frame_{frame_position:04d}.png"
                )
                write_mask_overlay(measurement.rgb_image, measurement.mask, overlay_path)
            progress_bar.update(1)
    return EpisodeSegmentation(
        raw_distances=raw_distances,
        object_masks=np.stack(object_masks),
        object_found=object_found,
        distances_measured=config.needs_object_distance,
    )


def mark_missing_object_masks(segmentation: EpisodeSegmentation, config: LabelObjectTaskConfig) -> np.ndarray:
    object_masks = segmentation.object_masks.copy()
    if config.mark_missing_labels:
        object_masks[~segmentation.object_found] = MISSING_MASK_CELL_VALUE
    return object_masks


def read_stage_inputs(
    dataset: LeRobotDataset, frame_indices: list[int], config: LabelObjectTaskConfig
) -> StageInputs:
    stage_channel_indices = find_stage_channel_indices(dataset, config)
    actions = read_column(dataset, ACTION, frame_indices)
    states = read_column(dataset, OBS_STATE, frame_indices)
    # The gripper is compared with its thresholds in float64, as when it was read frame by frame.
    return StageInputs(
        gripper_commands=actions[:, stage_channel_indices.gripper_command].astype(np.float64),
        gripper_positions=(
            None
            if stage_channel_indices.gripper_state is None
            else states[:, stage_channel_indices.gripper_state].astype(np.float64)
        ),
        joint_positions=states[:, stage_channel_indices.joint_positions],
    )


def read_column(dataset: LeRobotDataset, column_name: str, frame_indices: list[int]) -> np.ndarray:
    return torch.stack(dataset.select_columns(column_name)[frame_indices][column_name]).numpy()


def find_stage_channel_indices(dataset: LeRobotDataset, config: LabelObjectTaskConfig) -> StageChannelIndices:
    state_channel_names = feature_channel_names(dataset.meta.features[OBS_STATE])
    return StageChannelIndices(
        gripper_command=feature_channel_names(dataset.meta.features[ACTION]).index(
            config.gripper_action_channel_name
        ),
        gripper_state=(
            state_channel_names.index(config.resolved_gripper_state_channel_name)
            if config.requires_gripper_state
            else None
        ),
        joint_positions=find_position_channel_indices(state_channel_names),
    )


def label_object_distance_and_stage(
    episode_index: int,
    raw_distances: np.ndarray,
    stage_inputs: StageInputs | None,
    config: LabelObjectTaskConfig,
) -> dict[str, np.ndarray]:
    missing_frame_count = int(np.isnan(raw_distances).sum())
    if missing_frame_count == len(raw_distances):
        raise ValueError(
            f"Episode {episode_index}: SAM 3 found no {config.object_text_prompt!r} with valid depth "
            "in any frame. Check `object_text_prompt`, `mask_score_threshold` and the depth alignment."
        )
    object_distance = median_filter_distances(
        fill_missing_distances(raw_distances), config.distance_median_filter_frames
    )

    label_arrays: dict[str, np.ndarray] = {}
    if config.writes_object_distance:
        written_distance = object_distance
        if config.mark_missing_labels:
            written_distance = np.where(np.isnan(raw_distances), np.nan, object_distance)
        label_arrays[OBJECT_DISTANCE_FEATURE_KEY] = written_distance.astype(np.float32)
    stage = None
    if config.writes_stage:
        stage = assign_stage_labels(
            stage_inputs.gripper_commands,
            object_distance,
            stage_inputs.joint_positions,
            config.stage_thresholds,
            stage_inputs.gripper_positions,
        )
        label_arrays[STAGE_FEATURE_KEY] = stage
    logging.info(
        "Episode %d: %d frames, %d without a valid object distance, grasp from frame %s, pull from frame %s",
        episode_index,
        len(raw_distances),
        missing_frame_count,
        first_frame_of_stage(stage, GRASP_STAGE) if stage is not None else "not labelled",
        first_frame_of_stage(stage, PULL_STAGE) if stage is not None else "not labelled",
    )
    if config.debug_output_directory is not None:
        label_plot_path = Path(config.debug_output_directory) / f"episode_{episode_index:03d}_labels.png"
        write_label_plot(object_distance, stage, label_plot_path)
    return label_arrays


def measure_objects_in_frames(
    segmenter: Sam3Segmenter, frames: list[dict[str, torch.Tensor]], config: LabelObjectTaskConfig
) -> list[ObjectMeasurement]:
    rgb_images = [to_uint8_rgb_image(frame[config.rgb_image_key]) for frame in frames]
    segmenter_images = [
        rotate_clockwise(rgb_image, config.sam_input_rotation_degrees) for rgb_image in rgb_images
    ]
    masks = [
        None if mask is None else rotate_clockwise(mask, -config.sam_input_rotation_degrees)
        for mask in segment_object_masks(
            segmenter, segmenter_images, config.object_text_prompt, config.mask_score_threshold
        )
    ]
    measurements = []
    for frame, rgb_image, mask in zip(frames, rgb_images, masks, strict=True):
        if mask is None:
            empty_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
            measurements.append(ObjectMeasurement(rgb_image, empty_mask, None, False))
            continue
        if not config.needs_object_distance:
            measurements.append(ObjectMeasurement(rgb_image, mask, None, True))
            continue
        depth_image = frame[config.depth_image_key].squeeze(0).numpy()
        measurements.append(
            ObjectMeasurement(rgb_image, mask, median_depth_under_mask(depth_image, mask), True)
        )
    return measurements


def find_position_channel_indices(state_channel_names: list[str]) -> list[int]:
    position_channel_indices = [
        channel_index for channel_index, name in enumerate(state_channel_names) if "position" in name
    ]
    return position_channel_indices or list(range(len(state_channel_names)))


def find_debug_frame_positions(frame_count: int, config: LabelObjectTaskConfig) -> set[int]:
    if config.debug_output_directory is None:
        return set()
    sampled_positions = np.linspace(0, frame_count - 1, config.debug_frames_per_episode)
    return {int(position) for position in sampled_positions}


def to_uint8_rgb_image(image: torch.Tensor) -> np.ndarray:
    return (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def rotate_clockwise(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 90 != 0:
        raise ValueError(f"The rotation must be a multiple of 90 degrees. Got {degrees}.")
    # `np.rot90` turns counter-clockwise, so a clockwise turn is a negative number of quarter turns.
    return np.ascontiguousarray(np.rot90(image, k=-(degrees // 90)))


def segment_object_masks(
    segmenter: Sam3Segmenter,
    rgb_images: list[np.ndarray],
    object_text_prompt: str,
    mask_score_threshold: float,
) -> list[np.ndarray | None]:
    if not rgb_images:
        return []
    inputs = segmenter.processor(
        images=[Image.fromarray(rgb_image) for rgb_image in rgb_images],
        text=[object_text_prompt] * len(rgb_images),
        return_tensors="pt",
    ).to(segmenter.device)
    with torch.no_grad():
        outputs = segmenter.model(**inputs)
    original_sizes = inputs["original_sizes"]
    if isinstance(original_sizes, torch.Tensor):
        original_sizes = original_sizes.tolist()
    instances_per_image = segmenter.processor.post_process_instance_segmentation(
        outputs,
        threshold=mask_score_threshold,
        mask_threshold=0.5,
        target_sizes=original_sizes,
    )
    masks = []
    for instances in instances_per_image:
        if len(instances["scores"]) == 0:
            masks.append(None)
            continue
        best_instance_index = int(instances["scores"].argmax())
        masks.append(instances["masks"][best_instance_index].cpu().numpy().astype(bool))
    return masks


def segment_object_mask(
    segmenter: Sam3Segmenter, rgb_image: np.ndarray, object_text_prompt: str, mask_score_threshold: float
) -> np.ndarray | None:
    return segment_object_masks(segmenter, [rgb_image], object_text_prompt, mask_score_threshold)[0]


def median_depth_under_mask(depth_image_meters: np.ndarray, mask: np.ndarray) -> float | None:
    valid_depths = depth_image_meters[mask & np.isfinite(depth_image_meters) & (depth_image_meters > 0)]
    if valid_depths.size == 0:
        return None
    return float(np.median(valid_depths))


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    height, width = mask.shape
    downsampled_height = math.ceil(height / factor)
    downsampled_width = math.ceil(width / factor)
    padded_mask = np.zeros((downsampled_height * factor, downsampled_width * factor), dtype=np.float32)
    padded_mask[:height, :width] = mask
    blocks = padded_mask.reshape(downsampled_height, factor, downsampled_width, factor)
    return (blocks.mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


def write_mask_overlay(rgb_image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    overlay = rgb_image.astype(np.float32)
    overlay[mask] = (1 - MASK_OVERLAY_OPACITY) * overlay[mask] + MASK_OVERLAY_OPACITY * MASK_OVERLAY_COLOR
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.astype(np.uint8)).save(output_path)


def fill_missing_distances(distances: np.ndarray) -> np.ndarray:
    # Forward fill missing (NaN) distances; a leading gap takes the first valid distance.
    valid_positions = np.flatnonzero(~np.isnan(distances))
    if valid_positions.size == 0:
        raise ValueError("At least one valid distance is required to fill the missing ones.")
    last_valid_position = np.maximum.accumulate(np.where(np.isnan(distances), -1, np.arange(len(distances))))
    last_valid_position = np.where(last_valid_position < 0, valid_positions[0], last_valid_position)
    return distances[last_valid_position]


def median_filter_distances(distances: np.ndarray, window_frames: int) -> np.ndarray:
    half_window = window_frames // 2
    padded_distances = np.pad(distances, (half_window, half_window), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded_distances, window_frames)
    return np.median(windows, axis=1)


def assign_stage_labels(
    gripper_commands: np.ndarray,
    object_distances: np.ndarray,
    joint_positions: np.ndarray,
    thresholds: StageThresholds,
    gripper_positions: np.ndarray | None = None,
) -> np.ndarray:
    # Stages latch at their first crossing, so a label never goes backwards within an episode.
    stage = np.full(len(gripper_commands), APPROACH_STAGE, dtype=np.int64)
    grasp_onset = find_grasp_onset(gripper_commands, gripper_positions, object_distances, thresholds)
    if grasp_onset is None:
        return stage

    stage[grasp_onset:] = GRASP_STAGE
    displacement_from_grasp = np.linalg.norm(joint_positions - joint_positions[grasp_onset], axis=1)
    has_pulled = (displacement_from_grasp > thresholds.pull_joint_displacement_threshold) & (
        np.arange(len(stage)) > grasp_onset
    )
    if has_pulled.any():
        stage[int(np.argmax(has_pulled)) :] = PULL_STAGE
    return stage


def find_grasp_onset(
    gripper_commands: np.ndarray,
    gripper_positions: np.ndarray | None,
    object_distances: np.ndarray,
    thresholds: StageThresholds,
) -> int | None:
    is_near_object = object_distances < thresholds.grasp_distance_meters
    if thresholds.gripper_still_frames == 0:
        is_grasping = (gripper_commands < thresholds.gripper_closed_threshold) & is_near_object
        return int(np.argmax(is_grasping)) if is_grasping.any() else None

    # The grasp starts when the measured gripper is closed and has stopped moving, so the frames
    # while it is still closing stay in the approach. The label starts at the first still frame.
    still_frames = thresholds.gripper_still_frames
    is_closed = gripper_positions < thresholds.gripper_closed_threshold
    speed = np.abs(np.diff(gripper_positions, prepend=gripper_positions[0]))
    is_slow = speed < thresholds.gripper_still_speed_threshold
    slow_frames_in_window = np.convolve(is_slow.astype(int), np.ones(still_frames, dtype=int))[: len(is_slow)]
    is_grasping = is_closed & (slow_frames_in_window >= still_frames) & is_near_object
    return int(np.argmax(is_grasping)) - still_frames + 1 if is_grasping.any() else None


def first_frame_of_stage(stage: np.ndarray, stage_label: int) -> int | None:
    frames_in_stage = np.flatnonzero(stage == stage_label)
    return int(frames_in_stage[0]) if frames_in_stage.size else None


def write_label_plot(object_distance: np.ndarray, stage: np.ndarray | None, output_path: Path) -> None:
    require_package("matplotlib", extra="matplotlib-dep")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_count = 1 if stage is None else 2
    figure, axes = plt.subplots(axes_count, 1, sharex=True, figsize=(10, 2.5 * axes_count))
    axes = np.atleast_1d(axes)
    axes[0].plot(object_distance)
    axes[0].set_ylabel("object distance (m)")
    if stage is not None:
        axes[1].step(np.arange(len(stage)), stage, where="post")
        axes[1].set_yticks([APPROACH_STAGE, GRASP_STAGE, PULL_STAGE], ["approach", "grasp", "pull"])
    axes[-1].set_xlabel("frame")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100)
    plt.close(figure)


def write_label_features(
    dataset: LeRobotDataset,
    label_arrays_per_episode: list[dict[str, np.ndarray]],
    config: LabelObjectTaskConfig,
) -> None:
    label_values = {
        feature_key: np.concatenate([label_arrays[feature_key] for label_arrays in label_arrays_per_episode])
        for feature_key in label_arrays_per_episode[0]
    }
    feature_infos = {
        feature_key: {"dtype": str(values.dtype), "shape": list(values.shape[1:]) or [1], "names": None}
        for feature_key, values in label_values.items()
    }

    logging.info(
        "Writing %s to a copy of the dataset at %s",
        sorted(label_values),
        config.new_root or config.new_repo_id,
    )
    labelled_dataset = add_features(
        dataset,
        {feature_key: (label_values[feature_key], info) for feature_key, info in feature_infos.items()},
        output_dir=config.new_root,
        repo_id=config.new_repo_id,
    )
    # `recompute_stats` cannot index the 2-D mask column of the written parquet, so the stats of
    # the new features are computed from the labels in memory and merged into `stats.json`.
    episode_stats = [
        compute_episode_stats(valid_arrays, feature_infos)
        for valid_arrays in (
            drop_missing_label_values(label_arrays, config) for label_arrays in label_arrays_per_episode
        )
        if valid_arrays
    ]
    label_stats = aggregate_stats(episode_stats) if episode_stats else {}
    # `add_features` copies the stats that `LeRobotDataset` rescaled to metres in memory, while the copy keeps
    # the recorded depth unit, so the depth stats are read again from the source on disk.
    source_stats = load_stats(dataset.root) or {}
    depth_stats = {key: source_stats[key] for key in dataset.meta.depth_keys if key in source_stats}
    labelled_dataset.meta.stats = {**(load_stats(labelled_dataset.root) or {}), **depth_stats, **label_stats}
    write_stats(labelled_dataset.meta.stats, labelled_dataset.root)
    logging.info("Wrote %s to %s", sorted(label_values), labelled_dataset.root)


def drop_missing_label_values(
    label_arrays: dict[str, np.ndarray], config: LabelObjectTaskConfig
) -> dict[str, np.ndarray]:
    valid_arrays = {}
    for feature_key, values in label_arrays.items():
        if feature_key == OBJECT_DISTANCE_FEATURE_KEY:
            is_valid = ~np.isnan(values)
        elif feature_key == STAGE_FEATURE_KEY:
            is_valid = values >= 0
        else:
            is_valid = ~(values == MISSING_MASK_CELL_VALUE).reshape(len(values), -1).any(axis=1)
        if is_valid.any():
            valid_arrays[feature_key] = values[is_valid]
    return valid_arrays


def main() -> None:
    init_logging()
    label_dataset()


if __name__ == "__main__":
    main()
