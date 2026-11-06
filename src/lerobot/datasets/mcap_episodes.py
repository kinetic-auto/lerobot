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

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.import_utils import require_package

require_package("mcap", extra="mcap-dep")
require_package("mcap-ros2-support", extra="mcap-dep", import_name="mcap_ros2")

from mcap.reader import make_reader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402

NANOSECONDS_PER_SECOND = 1e-9


def find_episode_paths(session_directory: Path) -> list[Path]:
    # Find all .mcap files in the session directory
    episode_paths = sorted(session_directory.rglob("*.mcap"))

    # Group the episode paths by their identifier
    paths_by_identifier: dict[str, list[Path]] = {}
    for episode_path in episode_paths:
        paths_by_identifier.setdefault(episode_identifier(episode_path), []).append(episode_path)

    # Check if there are any duplicate episode IDs
    duplicated = {identifier: paths for identifier, paths in paths_by_identifier.items() if len(paths) > 1}
    if duplicated:
        raise ValueError(
            f"Several .mcap files share one episode id (split bags are not supported): {duplicated}"
        )

    return episode_paths


def episode_identifier(mcap_path: Path) -> str:
    # Sessions store one episode per folder as <episode>/bag/bag_0.mcap
    if mcap_path.parent.name == "bag":
        return mcap_path.parent.parent.name

    # Return the episode ID as the stem of the mcap path
    return mcap_path.stem


def read_message_timestamps(mcap_path: Path, topics: list[str]) -> dict[str, np.ndarray]:
    # Read the message timestamps for each topic
    timestamps_by_topic: dict[str, list[float]] = {topic: [] for topic in topics}
    with open(mcap_path, "rb") as mcap_file:
        reader = make_reader(mcap_file)
        for _, channel, message in reader.iter_messages(topics=topics):
            timestamps_by_topic[channel.topic].append(message.log_time * NANOSECONDS_PER_SECOND)

    return {topic: np.sort(np.array(timestamps)) for topic, timestamps in timestamps_by_topic.items()}


def decode_topic_messages(mcap_path: Path, topics: list[str]) -> Iterator[tuple[str, float, Any]]:
    # Decode each message on the requested topics
    with open(mcap_path, "rb") as mcap_file:
        reader = make_reader(mcap_file, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded_message in reader.iter_decoded_messages(topics=topics):
            yield channel.topic, message.log_time * NANOSECONDS_PER_SECOND, decoded_message
