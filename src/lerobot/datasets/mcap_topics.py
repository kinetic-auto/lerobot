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

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_MCAP_TOPICS_PATH = Path(__file__).parent / "config" / "mcap_conversion.yaml"
ROBOT_STATE_NAME = "robot_state"
GRIPPER_FEEDBACK_NAME = "gripper_feedback"


@dataclass(frozen=True)
class McapTopics:
    topics: tuple[str, ...]
    topics_mapping: dict[str, str]

    def required_topics(self) -> list[str]:
        return list(self.topics)

    def topic_named(self, name: str) -> str:
        matches = [topic for topic, mapped_name in self.topics_mapping.items() if mapped_name == name]
        if len(matches) != 1:
            raise ValueError(f"topics_mapping must contain {name!r} exactly once, found {matches}.")
        return matches[0]

    @property
    def depth_topics(self) -> tuple[str, ...]:
        return tuple(topic for topic, name in self.topics_mapping.items() if name.endswith("_depth"))


def load_mcap_topics(path: str | Path | None = None) -> McapTopics:
    config_path = Path(path) if path is not None else DEFAULT_MCAP_TOPICS_PATH
    with config_path.open() as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must be a mapping.")

    topics = raw.get("topics")
    topics_mapping = raw.get("topics_mapping")
    if (
        not isinstance(topics, list)
        or not topics
        or any(not isinstance(topic, str) or not topic for topic in topics)
    ):
        raise ValueError(f"{config_path} topics must be a non-empty list of topic strings.")
    if len(topics) != len(set(topics)):
        raise ValueError(f"{config_path} lists the same topic more than once.")
    if not isinstance(topics_mapping, dict) or not topics_mapping:
        raise ValueError(f"{config_path} topics_mapping must be a non-empty mapping.")
    if set(topics_mapping) != set(topics):
        raise ValueError(f"{config_path} topics_mapping keys must match topics.")
    if any(not isinstance(name, str) or not name for name in topics_mapping.values()):
        raise ValueError(f"{config_path} topics_mapping values must be non-empty strings.")
    if len(set(topics_mapping.values())) != len(topics_mapping):
        raise ValueError(f"{config_path} topics_mapping names must be unique.")
    return McapTopics(topics=tuple(topics), topics_mapping=dict(topics_mapping))
