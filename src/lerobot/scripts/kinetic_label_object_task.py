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
leaves out of its losses, unless `--mark_missing_labels false` fills them instead.

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

import logging
import math
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
    object_text_prompt: str = "door handle"
    sam3_model_name_or_path: str = "facebook/sam3"
    mask_score_threshold: float = 0.4
    mask_downsample_factor: int = 16
    distance_median_filter_frames: int = 5
    mark_missing_labels: bool = True
    gripper_action_channel_name: str = "gripper.position.x"
    debug_output_directory: str | None = None
    debug_frames_per_episode: int = 6
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
    def stage_thresholds(self) -> StageThresholds:
        missing_names = [name for name in STAGE_THRESHOLD_FIELD_NAMES if getattr(self, name) is None]
        if missing_names:
            raise ValueError(f"The stage label requires {missing_names}.")
        return StageThresholds(
            gripper_closed_threshold=self.gripper_closed_threshold,
            grasp_distance_meters=self.grasp_distance_meters,
            pull_joint_displacement_threshold=self.pull_joint_displacement_threshold,
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
    segmenter = load_sam3_segmenter(config.sam3_model_name_or_path, config.device)
    logging.info("Loaded SAM 3 from %s on %s", config.sam3_model_name_or_path, config.device)

    with (
        logging_redirect_tqdm(),
        tqdm(total=dataset.meta.total_frames, desc="Labelling frames", unit="frame") as progress_bar,
    ):
        label_arrays_per_episode = [
            label_episode(dataset, episode_index, segmenter, config, progress_bar)
            for episode_index in range(dataset.meta.total_episodes)
        ]
    write_label_features(dataset, label_arrays_per_episode, config)


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
    if config.distance_median_filter_frames < 1 or config.distance_median_filter_frames % 2 == 0:
        raise ValueError(
            f"`distance_median_filter_frames` must be a positive odd number. "
            f"Got {config.distance_median_filter_frames}."
        )
    if config.mask_downsample_factor < 1:
        raise ValueError(f"`mask_downsample_factor` must be positive. Got {config.mask_downsample_factor}.")
    if not 0.0 < config.mask_score_threshold < 1.0:
        raise ValueError(f"`mask_score_threshold` must be in (0, 1). Got {config.mask_score_threshold}.")
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
    rgb_shape = tuple(features[config.rgb_image_key]["shape"][:2])
    depth_shape = tuple(features[config.depth_image_key]["shape"][:2])
    if rgb_shape != depth_shape:
        raise ValueError(
            f"The depth image {depth_shape} must be aligned to the RGB image {rgb_shape}: "
            "same height and width."
        )


def load_sam3_segmenter(model_name_or_path: str, device: str) -> Sam3Segmenter:
    require_package("transformers", extra="transformers-dep")
    model = Sam3Model.from_pretrained(model_name_or_path).to(device).eval()
    processor = Sam3Processor.from_pretrained(model_name_or_path)
    return Sam3Segmenter(model=model, processor=processor, device=device)


def label_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    segmenter: Sam3Segmenter,
    config: LabelObjectTaskConfig,
    progress_bar: tqdm,
) -> dict[str, np.ndarray]:
    episode = dataset.meta.episodes[episode_index]
    frame_indices = range(episode["dataset_from_index"], episode["dataset_to_index"])
    stage_channel_indices = find_stage_channel_indices(dataset, config) if config.writes_stage else None
    debug_frame_positions = find_debug_frame_positions(len(frame_indices), config)

    raw_distances = np.full(len(frame_indices), np.nan)
    undetected_frame_count = 0
    object_masks = []
    gripper_commands = []
    joint_positions = []
    for frame_position, frame_index in enumerate(frame_indices):
        frame = dataset[frame_index]
        measurement = measure_object_in_frame(segmenter, frame, config)
        if measurement.distance is not None:
            raw_distances[frame_position] = measurement.distance
        undetected_frame_count += not measurement.object_found

        if config.writes_object_mask:
            object_mask = downsample_mask(measurement.mask, config.mask_downsample_factor)
            if config.mark_missing_labels and not measurement.object_found:
                object_mask = np.full_like(object_mask, MISSING_MASK_CELL_VALUE)
            object_masks.append(object_mask)
        if stage_channel_indices is not None:
            gripper_channel_index, position_channel_indices = stage_channel_indices
            gripper_commands.append(float(frame[ACTION][gripper_channel_index]))
            joint_positions.append(frame[OBS_STATE][position_channel_indices].numpy())
        if frame_position in debug_frame_positions:
            overlay_path = Path(config.debug_output_directory) / (
                f"episode_{episode_index:03d}_frame_{frame_position:04d}.png"
            )
            write_mask_overlay(measurement.rgb_image, measurement.mask, overlay_path)
        progress_bar.update(1)

    progress_bar.set_postfix(episode=f"{episode_index + 1}/{dataset.meta.total_episodes}")
    logging.info(
        "Episode %d/%d: %d frames, %d without an object detection",
        episode_index + 1,
        dataset.meta.total_episodes,
        len(frame_indices),
        undetected_frame_count,
    )
    label_arrays: dict[str, np.ndarray] = {}
    if config.writes_object_mask:
        label_arrays[config.object_mask_feature_key] = np.stack(object_masks)
    if config.needs_object_distance:
        label_arrays.update(
            label_object_distance_and_stage(
                episode_index, raw_distances, gripper_commands, joint_positions, config
            )
        )
    return label_arrays


def find_stage_channel_indices(
    dataset: LeRobotDataset, config: LabelObjectTaskConfig
) -> tuple[int, list[int]]:
    gripper_channel_index = feature_channel_names(dataset.meta.features[ACTION]).index(
        config.gripper_action_channel_name
    )
    position_channel_indices = find_position_channel_indices(
        feature_channel_names(dataset.meta.features[OBS_STATE])
    )
    return gripper_channel_index, position_channel_indices


def label_object_distance_and_stage(
    episode_index: int,
    raw_distances: np.ndarray,
    gripper_commands: list[float],
    joint_positions: list[np.ndarray],
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
            np.array(gripper_commands), object_distance, np.stack(joint_positions), config.stage_thresholds
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


def measure_object_in_frame(
    segmenter: Sam3Segmenter, frame: dict[str, torch.Tensor], config: LabelObjectTaskConfig
) -> ObjectMeasurement:
    rgb_image = to_uint8_rgb_image(frame[config.rgb_image_key])
    mask = segment_object_mask(segmenter, rgb_image, config.object_text_prompt, config.mask_score_threshold)
    if mask is None:
        return ObjectMeasurement(rgb_image, np.zeros(rgb_image.shape[:2], dtype=bool), None, False)
    if not config.needs_object_distance:
        return ObjectMeasurement(rgb_image, mask, None, True)
    depth_image = frame[config.depth_image_key].squeeze(0).numpy()
    return ObjectMeasurement(rgb_image, mask, median_depth_under_mask(depth_image, mask), True)


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


def segment_object_mask(
    segmenter: Sam3Segmenter, rgb_image: np.ndarray, object_text_prompt: str, mask_score_threshold: float
) -> np.ndarray | None:
    inputs = segmenter.processor(
        images=Image.fromarray(rgb_image), text=object_text_prompt, return_tensors="pt"
    ).to(segmenter.device)
    with torch.no_grad():
        outputs = segmenter.model(**inputs)
    instances = segmenter.processor.post_process_instance_segmentation(
        outputs,
        threshold=mask_score_threshold,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    if len(instances["scores"]) == 0:
        return None
    best_instance_index = int(instances["scores"].argmax())
    return instances["masks"][best_instance_index].cpu().numpy().astype(bool)


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
) -> np.ndarray:
    # Stages latch at their first crossing, so a label never goes backwards within an episode.
    is_grasping = (gripper_commands < thresholds.gripper_closed_threshold) & (
        object_distances < thresholds.grasp_distance_meters
    )
    stage = np.full(len(gripper_commands), APPROACH_STAGE, dtype=np.int64)
    if not is_grasping.any():
        return stage

    grasp_onset = int(np.argmax(is_grasping))
    stage[grasp_onset:] = GRASP_STAGE
    displacement_from_grasp = np.linalg.norm(joint_positions - joint_positions[grasp_onset], axis=1)
    has_pulled = (displacement_from_grasp > thresholds.pull_joint_displacement_threshold) & (
        np.arange(len(stage)) > grasp_onset
    )
    if has_pulled.any():
        stage[int(np.argmax(has_pulled)) :] = PULL_STAGE
    return stage


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
    labelled_dataset.meta.stats = {**(load_stats(labelled_dataset.root) or {}), **label_stats}
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
