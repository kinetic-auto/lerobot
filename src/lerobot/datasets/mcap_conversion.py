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
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .dataset_validation import CONVERSION_MANIFEST_FILE_NAME
from .joint_layout import packed_feature_names
from .lerobot_dataset import LeRobotDataset
from .mcap_depth import (
    CameraParameters,
    align_depth_to_rgb,
    decode_compressed_depth_mm,
    load_camera_parameters_from_intrinsics_yaml,
    load_depth_to_rgb_transform,
    validate_depth_image_dims,
)
from .mcap_episodes import episode_identifier, find_episode_paths, decode_topic_messages
from .mcap_topics import GRIPPER_FEEDBACK_NAME, ROBOT_STATE_NAME, load_mcap_topics
from .mcap_validation import (
    JSON_REPORT_FILE_NAME,
    REPORT_DIRECTORY_NAME,
    SEVERITY_RANK,
    Severity,
    load_episode_severities,
)

STATE_FIELDS = ("position", "effort")
IMAGE_CHANNEL_NAMES = ["height", "width", "channels"]


@dataclass
class McapConversionConfig:
    session_directory: str
    output_root: str
    repo_id: str
    d405_color_intrinsics_path: str | None = None
    d405_depth_intrinsics_path: str | None = None
    align_depth: bool = True
    task: str = "open the door"
    fps: int = 10
    action_lookahead_frames: int = 3
    quality_report_path: str | None = None
    include_flagged: str = "warning"
    episode_allow_list_path: str | None = None
    robot_type: str = "ur20"
    joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    )
    gripper_joint_name: str = "gripper"
    camera_feature_names: list[str] = field(
        default_factory=lambda: ["pc1", "realsense", "realsense_depth"]
    )
    img_sizes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "pc1": [640, 360],
            "realsense": [640, 360],
            "realsense_depth": [640, 360],
        }
    )
    no_return_depth_mm: int = 10000


@dataclass
class TimedMessages:
    timestamps: np.ndarray
    payloads: list[Any]


@dataclass
class DepthAlignment:
    depth_camera: CameraParameters
    color_camera: CameraParameters
    depth_frame_wrt_rgb_frame: np.ndarray


@dataclass
class EpisodeStreams:
    joint_timestamps: np.ndarray
    joint_positions: np.ndarray
    joint_efforts: np.ndarray
    gripper_timestamps: np.ndarray
    gripper_positions: np.ndarray
    gripper_efforts: np.ndarray
    camera_frames: dict[str, TimedMessages]
    depth_alignment: DepthAlignment | None


def convert_session(config: McapConversionConfig) -> None:
    # Validate the mcap conversion config
    validate_config(config)

    selected_paths, skipped_reasons = filter_episodes_by_quality(config)
    depth_alignment = load_depth_alignment(config) if config.align_depth else None
    state_names = packed_feature_names([*config.joint_names, config.gripper_joint_name], STATE_FIELDS)

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=config.fps,
        features=build_features(config, state_names),
        root=config.output_root,
        robot_type=config.robot_type,
        use_videos=True,
    )

    converted_episodes = []
    try:
        for mcap_path in selected_paths:
            # Get the episode ID
            episode_id = episode_identifier(mcap_path)
            logging.info("Converting %s", episode_id)
            try:
                frame_count = convert_episode(dataset, mcap_path, depth_alignment, config)
            except (ValueError, cv2.error) as error:
                skipped_reasons[episode_id] = f"conversion failed: {error}"
                logging.error("Skipped %s: %s", episode_id, skipped_reasons[episode_id])
                continue
            
            if frame_count is None:
                skipped_reasons[episode_id] = "shorter than the action lookahead after time alignment"
                logging.warning("Skipped %s: %s", episode_id, skipped_reasons[episode_id])
                continue

            # Add the episode to the manifest
            converted_episodes.append(
                {
                    "episode_id": episode_id,
                    "episode_index": len(converted_episodes),
                    "frame_count": frame_count,
                }
            )
            logging.info("Converted %s (%d frames)", episode_id, frame_count)
    finally:
        dataset.finalize()

    # Write the manifest
    write_manifest(Path(config.output_root), converted_episodes, skipped_reasons)

    if not converted_episodes:
        raise ValueError(f"No episode could be converted. Skipped: {skipped_reasons}")


def validate_config(config: McapConversionConfig) -> None:
    if config.fps <= 0:
        raise ValueError(f"fps must be positive, got {config.fps}.")
    if config.action_lookahead_frames < 1:
        raise ValueError(f"action_lookahead_frames must be at least 1, got {config.action_lookahead_frames}.")
    if config.include_flagged not in {severity.value for severity in Severity}:
        raise ValueError(
            f"include_flagged must be one of {[severity.value for severity in Severity]}, "
            f"got {config.include_flagged!r}."
        )
    for feature_name in config.camera_feature_names:
        image_size = config.img_sizes.get(feature_name)
        if image_size is None or len(image_size) != 2:
            raise ValueError(f"img_sizes[{feature_name!r}] must be [width, height], got {image_size}.")
    max_depth_mm = np.iinfo(np.uint16).max
    if not 0 < config.no_return_depth_mm <= max_depth_mm:
        raise ValueError(
            f"no_return_depth_mm must be in (0, {max_depth_mm}], got {config.no_return_depth_mm}."
        )
    if not config.joint_names:
        raise ValueError("joint_names must not be empty.")
    if config.align_depth and not (
        config.d405_color_intrinsics_path and config.d405_depth_intrinsics_path
    ):
        raise ValueError(
            "align_depth requires d405_color_intrinsics_path and d405_depth_intrinsics_path."
        )


def filter_episodes_by_quality(config: McapConversionConfig) -> tuple[list[Path], dict[str, str]]:
    severities = load_episode_severities(resolve_quality_report_path(config))
    allow_list = load_allow_list(config.episode_allow_list_path)
    maximum_rank = SEVERITY_RANK[Severity(config.include_flagged)]

    selected_paths: list[Path] = []
    skipped_reasons: dict[str, str] = {}
    for mcap_path in find_episode_paths(Path(config.session_directory)):
        episode_id = episode_identifier(mcap_path)
        if allow_list is not None and episode_id not in allow_list:
            skipped_reasons[episode_id] = "not in the allow list"
        elif episode_id not in severities:
            skipped_reasons[episode_id] = "missing from the quality report"
        elif SEVERITY_RANK[severities[episode_id]] > maximum_rank:
            skipped_reasons[episode_id] = f"quality report severity is {severities[episode_id].value}"
        else:
            selected_paths.append(mcap_path)
    if not selected_paths:
        raise ValueError(f"No episodes left to convert. Skipped: {skipped_reasons}")
    return selected_paths, skipped_reasons


def resolve_quality_report_path(config: McapConversionConfig) -> Path:
    if config.quality_report_path is not None:
        report_path = Path(config.quality_report_path)
    else:
        report_path = Path(config.session_directory) / REPORT_DIRECTORY_NAME / JSON_REPORT_FILE_NAME
    if not report_path.exists():
        raise FileNotFoundError(
            f"Quality report {report_path} not found. Run lerobot-mcap-valid on the session first."
        )
    return report_path


def load_allow_list(allow_list_path: str | None) -> set[str] | None:
    if allow_list_path is None:
        return None
    lines = Path(allow_list_path).read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def load_depth_alignment(config: McapConversionConfig) -> DepthAlignment:
    if config.d405_depth_intrinsics_path is None or config.d405_color_intrinsics_path is None:
        raise ValueError(
            "align_depth requires d405_color_intrinsics_path and d405_depth_intrinsics_path."
        )
    depth_intrinsics_path = Path(config.d405_depth_intrinsics_path).expanduser()
    color_intrinsics_path = Path(config.d405_color_intrinsics_path).expanduser()
    return DepthAlignment(
        depth_camera=load_camera_parameters_from_intrinsics_yaml(depth_intrinsics_path),
        color_camera=load_camera_parameters_from_intrinsics_yaml(color_intrinsics_path),
        depth_frame_wrt_rgb_frame=load_depth_to_rgb_transform(depth_intrinsics_path, color_intrinsics_path),
    )


def build_features(config: McapConversionConfig, state_names: list[str]) -> dict[str, dict]:
    features: dict[str, dict] = {
        OBS_STATE: {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        ACTION: {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
    }
    for feature_name in config.camera_feature_names:
        width, height = config.img_sizes[feature_name]
        is_depth = feature_name.endswith("_depth")
        feature = {
            "dtype": "video",
            "shape": (height, width, 1 if is_depth else 3),
            "names": IMAGE_CHANNEL_NAMES,
        }
        if is_depth:
            feature["info"] = {"is_depth_map": True}
        features[f"{OBS_IMAGES}.{feature_name}"] = feature
    return features


def convert_episode(
    dataset: LeRobotDataset,
    mcap_path: Path,
    depth_alignment: DepthAlignment | None,
    config: McapConversionConfig,
) -> int | None:
    streams = read_episode_streams(mcap_path, depth_alignment, config)
    time_grid = build_time_grid(streams, config)
    if time_grid is None:
        return None

    # State observations
    state = build_state_matrix(streams, time_grid)
    if not np.isfinite(state).all():
        raise ValueError(f"{mcap_path} has non-finite joint or gripper values.")

    # Camera observations
    frame_count = len(time_grid) - config.action_lookahead_frames
    camera_observations = (
        decode_camera_observations(streams, time_grid[:frame_count], config)
        if streams.camera_frames
        else None
    )

    # Add the frames to the dataset
    added_frame_count = 0
    try:
        for frame_index in range(frame_count):
            frame = {
                OBS_STATE: state[frame_index],
                ACTION: state[frame_index + config.action_lookahead_frames],
                "task": config.task,
            }
            if camera_observations is not None:
                frame.update(next(camera_observations))
            dataset.add_frame(frame)
            added_frame_count += 1
    except (ValueError, cv2.error):
        if added_frame_count > 0:
            dataset.clear_episode_buffer()
        raise
    dataset.save_episode()
    return frame_count


def read_episode_streams(
    mcap_path: Path, depth_alignment: DepthAlignment | None, config: McapConversionConfig
) -> EpisodeStreams:
    topics = load_mcap_topics()
    state_topic = topics.topic_named(ROBOT_STATE_NAME)
    gripper_topic = topics.topic_named(GRIPPER_FEEDBACK_NAME)
    camera_topics = {
        feature_name: topics.topic_named(feature_name) for feature_name in config.camera_feature_names
    }
    depth_topics = set(topics.depth_topics)
    joint_rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    gripper_rows: list[tuple[float, float, float]] = []
    image_timestamps: dict[str, list[float]] = {topic: [] for topic in camera_topics.values()}
    image_payloads: dict[str, list[Any]] = {topic: [] for topic in camera_topics.values()}
    joint_indices: list[int] | None = None

    for topic, timestamp, message in decode_topic_messages(mcap_path, topics.required_topics()):
        if topic == state_topic:
            if joint_indices is None:
                joint_indices = find_joint_indices(list(message.name), config.joint_names)
            positions = np.asarray(message.position, dtype=np.float64)[joint_indices]
            efforts = np.asarray(message.effort, dtype=np.float64)[joint_indices]
            joint_rows.append((timestamp, positions, efforts))
        elif topic == gripper_topic:
            # Torque is unused. Sim bags store NaN, so the effort channel is zero for every source.
            gripper_rows.append((timestamp, message.opening_m, 0.0))
        elif topic in image_timestamps:
            image_timestamps[topic].append(timestamp)
            if topic in depth_topics:
                image_payloads[topic].append((message.format, bytes(message.data)))
            else:
                image_payloads[topic].append(bytes(message.data))

    require_messages(mcap_path, state_topic, len(joint_rows))
    require_messages(mcap_path, gripper_topic, len(gripper_rows))
    for topic, timestamps in image_timestamps.items():
        require_messages(mcap_path, topic, len(timestamps))

    return EpisodeStreams(
        joint_timestamps=np.array([joint_row[0] for joint_row in joint_rows]),
        joint_positions=np.stack([joint_row[1] for joint_row in joint_rows]),
        joint_efforts=np.stack([joint_row[2] for joint_row in joint_rows]),
        gripper_timestamps=np.array([gripper_row[0] for gripper_row in gripper_rows]),
        gripper_positions=np.array([gripper_row[1] for gripper_row in gripper_rows]),
        gripper_efforts=np.array([gripper_row[2] for gripper_row in gripper_rows]),
        camera_frames={
            feature_name: TimedMessages(np.array(image_timestamps[topic]), image_payloads[topic])
            for feature_name, topic in camera_topics.items()
        },
        depth_alignment=depth_alignment,
    )


def find_joint_indices(message_joint_names: list[str], wanted_joint_names: Sequence[str]) -> list[int]:
    missing_joint_names = [name for name in wanted_joint_names if name not in message_joint_names]
    if missing_joint_names:
        raise ValueError(
            f"Joints {missing_joint_names} are not in the recorded joint states {message_joint_names}."
        )
    return [message_joint_names.index(name) for name in wanted_joint_names]


def require_messages(mcap_path: Path, topic: str, message_count: int) -> None:
    if message_count == 0:
        raise ValueError(f"{mcap_path} has no messages on {topic}.")


def build_time_grid(streams: EpisodeStreams, config: McapConversionConfig) -> np.ndarray | None:
    all_timestamps = [
        streams.joint_timestamps,
        streams.gripper_timestamps,
        *[frames.timestamps for frames in streams.camera_frames.values()],
    ]
    start_time = max(timestamps[0] for timestamps in all_timestamps)
    end_time = min(timestamps[-1] for timestamps in all_timestamps)
    grid_length = int(np.floor((end_time - start_time) * config.fps)) + 1
    if grid_length <= config.action_lookahead_frames:
        return None
    return start_time + np.arange(grid_length) / config.fps


def latest_message_indices(timestamps: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(timestamps, time_grid, side="right") - 1, 0, None)


def build_state_matrix(streams: EpisodeStreams, time_grid: np.ndarray) -> np.ndarray:
    joint_rows = latest_message_indices(streams.joint_timestamps, time_grid)
    gripper_rows = latest_message_indices(streams.gripper_timestamps, time_grid)
    positions = np.concatenate(
        [streams.joint_positions[joint_rows], streams.gripper_positions[gripper_rows, None]], axis=1
    )
    efforts = np.concatenate(
        [streams.joint_efforts[joint_rows], streams.gripper_efforts[gripper_rows, None]], axis=1
    )
    # Block layout matches packed_feature_names: every position first, then every effort
    return np.concatenate([positions, efforts], axis=1).astype(np.float32)


def decode_camera_observations(
    streams: EpisodeStreams, time_grid: np.ndarray, config: McapConversionConfig
) -> Iterator[dict[str, np.ndarray]]:
    if not streams.camera_frames:
        return
    decoders = {}
    for feature_name, frames in streams.camera_frames.items():
        output_size = tuple(config.img_sizes[feature_name])
        if feature_name.endswith("_depth"):
            decoder = depth_frame_decoder(
                streams.depth_alignment, output_size, config.no_return_depth_mm
            )
        else:
            decoder = LatestMessageDecoder(lambda payload, size=output_size: decode_rgb_image(payload, size))
        decoders[f"{OBS_IMAGES}.{feature_name}"] = (frames, decoder)
    message_indices = {
        feature_key: latest_message_indices(frames.timestamps, time_grid)
        for feature_key, (frames, _) in decoders.items()
    }
    for frame_index in range(len(time_grid)):
        yield {
            feature_key: decoder(frames.payloads, message_indices[feature_key][frame_index])
            for feature_key, (frames, decoder) in decoders.items()
        }


class LatestMessageDecoder:
    def __init__(self, decode: Callable[[Any], np.ndarray]) -> None:
        self._decode = decode
        self._decoded_index: int | None = None
        self._decoded_image: np.ndarray | None = None

    def __call__(self, payloads: list[Any], message_index: int) -> np.ndarray:
        if message_index != self._decoded_index:
            self._decoded_image = self._decode(payloads[message_index])
            self._decoded_index = message_index
        return self._decoded_image


def depth_frame_decoder(
    depth_alignment: DepthAlignment | None,
    output_size: tuple[int, int],
    no_return_depth_mm: int,
) -> LatestMessageDecoder:
    if depth_alignment is None:
        return LatestMessageDecoder(
            lambda payload: decode_depth_image(payload, output_size, no_return_depth_mm)
        )
    return LatestMessageDecoder(
        lambda payload: decode_aligned_depth_image(
            payload, depth_alignment, output_size, no_return_depth_mm
        )
    )


def decode_rgb_image(jpeg_bytes: bytes, output_size: tuple[int, int]) -> np.ndarray:
    bgr_image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("A recorded camera frame did not decode as a JPEG image.")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb_image, output_size, interpolation=cv2.INTER_AREA)


def decode_depth_image(
    depth_payload: tuple[str, bytes],
    output_size: tuple[int, int],
    no_return_depth_mm: int,
) -> np.ndarray:
    depth_millimeters = decode_depth_millimeters(depth_payload)
    return store_depth_image(depth_millimeters, output_size, no_return_depth_mm)


def decode_aligned_depth_image(
    depth_payload: tuple[str, bytes],
    alignment: DepthAlignment,
    output_size: tuple[int, int],
    no_return_depth_mm: int,
) -> np.ndarray:
    depth_millimeters = decode_depth_millimeters(depth_payload)
    validate_depth_image_dims(depth_millimeters, alignment.depth_camera)
    aligned_millimeters = align_depth_to_rgb(
        depth_millimeters, alignment.depth_camera, alignment.color_camera, alignment.depth_frame_wrt_rgb_frame
    )
    return store_depth_image(aligned_millimeters, output_size, no_return_depth_mm)


def decode_depth_millimeters(depth_payload: tuple[str, bytes]) -> np.ndarray:
    depth_format, compressed_bytes = depth_payload
    return decode_compressed_depth_mm(depth_format, compressed_bytes)


def store_depth_image(
    depth_millimeters: np.ndarray, output_size: tuple[int, int], no_return_depth_mm: int
) -> np.ndarray:
    # Nearest-neighbor keeps a no-return zero from blending into a real range
    resized_millimeters = cv2.resize(depth_millimeters, output_size, interpolation=cv2.INTER_NEAREST)
    # The dataset's depth video decodes into a floor above zero, so a no-return zero would read back as
    # a very near surface. Store it at the far end of the range instead.
    resized_millimeters[resized_millimeters == 0] = no_return_depth_mm
    return resized_millimeters[..., None]


def write_manifest(
    output_root: Path, converted_episodes: list[dict[str, Any]], skipped_reasons: dict[str, str]
) -> None:
    manifest = {"converted": converted_episodes, "skipped": skipped_reasons}
    (output_root / CONVERSION_MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=2))
