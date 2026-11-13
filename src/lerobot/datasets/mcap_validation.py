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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from mcap.exceptions import McapError
import numpy as np

from .mcap_depth import (
    CameraParameters,
    decode_compressed_depth_mm,
    load_camera_parameters_from_intrinsics_yaml,
    validate_depth_image_dims,
)
from .mcap_episodes import (
    episode_identifier,
    find_episode_paths,
    decode_topic_messages,
    read_message_timestamps,
)
from .mcap_topics import load_mcap_topics

REPORT_DIRECTORY_NAME = "mcap_valid_reports"
JSON_REPORT_FILE_NAME = "report.json"
MARKDOWN_REPORT_FILE_NAME = "report.md"


class Severity(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    CRITICAL = "critical"


SEVERITY_RANK = {Severity.PASS: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


@dataclass
class McapValidationConfig:
    session_directory: str
    report_directory: str | None = None
    required_topics: list[str] = field(default_factory=lambda: load_mcap_topics().required_topics())
    depth_topics: list[str] = field(default_factory=lambda: list(load_mcap_topics().depth_topics))
    d405_depth_intrinsics_path: str | None = None
    stream_gap_factor: float = 5.0
    stream_min_gap_seconds: float = 0.5
    fps_tolerance: float = 0.30
    fail_on_critical: bool = False


@dataclass
class Issue:
    severity: Severity
    message: str


@dataclass
class TopicReport:
    topic: str
    message_count: int
    mean_rate_hz: float
    max_gap_seconds: float
    issues: list[Issue] = field(default_factory=list)


@dataclass
class EpisodeReport:
    episode_id: str
    mcap_path: str
    duration_seconds: float
    topics: list[TopicReport]
    episode_issues: list[Issue] = field(default_factory=list)

    @property
    def all_issues(self) -> list[Issue]:
        topic_issues = [issue for topic_report in self.topics for issue in topic_report.issues]
        return self.episode_issues + topic_issues

    @property
    def severity(self) -> Severity:
        return max(
            (issue.severity for issue in self.all_issues),
            key=SEVERITY_RANK.get,
            default=Severity.PASS,
        )


def validate_session(config: McapValidationConfig) -> list[EpisodeReport]:
    # Find the .mcap files in the session directory
    mcap_paths = find_episode_paths(Path(config.session_directory))
    if not mcap_paths:
        raise FileNotFoundError(f"No .mcap episode files found under {config.session_directory}.")

    # Load the depth camera parameters if provided
    depth_camera = None
    if config.d405_depth_intrinsics_path is not None:
        depth_camera = load_depth_camera(config)

    # Generate episode reports
    episode_reports = [validate_episode(mcap_path, depth_camera, config) for mcap_path in mcap_paths]

    # Flag episodes with inconsistent topic hz
    flag_topic_hz_degradation(episode_reports, config.fps_tolerance)

    return episode_reports


def load_depth_camera(config: McapValidationConfig) -> CameraParameters | None:
    return load_camera_parameters_from_intrinsics_yaml(Path(config.d405_depth_intrinsics_path).expanduser())


def validate_episode(
    mcap_path: Path, depth_camera: CameraParameters | None, config: McapValidationConfig
) -> EpisodeReport:
    try:
        timestamps_by_topic = read_message_timestamps(mcap_path, config.required_topics)
    except (OSError, McapError) as error:
        return EpisodeReport(
            episode_id=episode_identifier(mcap_path),
            mcap_path=str(mcap_path),
            duration_seconds=0.0,
            topics=[],
            episode_issues=[Issue(Severity.CRITICAL, f"unreadable mcap: {error}")],
        )

    # Calculate the episode start and end times
    window_start, window_end = get_episode_duration(timestamps_by_topic)

    # Generate topic reports
    topic_reports = [
        build_topic_report(topic, timestamps_by_topic[topic], window_start, window_end, config)
        for topic in config.required_topics
    ]

    # Check if the depth decodes
    episode_issues = validate_depth_topics(mcap_path, depth_camera, config.depth_topics)

    return EpisodeReport(
        episode_id=episode_identifier(mcap_path),
        mcap_path=str(mcap_path),
        duration_seconds=window_end - window_start,
        topics=topic_reports,
        episode_issues=episode_issues,
    )


def get_episode_duration(timestamps_by_topic: dict[str, np.ndarray]) -> tuple[float, float]:
    recorded_timestamps = [timestamps for timestamps in timestamps_by_topic.values() if timestamps.size > 0]
    if not recorded_timestamps:
        return 0.0, 0.0

    # Get the minimum and maximum timestamps
    min_timestamp = min(float(timestamps[0]) for timestamps in recorded_timestamps)
    max_timestamp = max(float(timestamps[-1]) for timestamps in recorded_timestamps)

    return min_timestamp, max_timestamp


def build_topic_report(
    topic: str,
    timestamps: np.ndarray,
    window_start: float,
    window_end: float,
    config: McapValidationConfig,
) -> TopicReport:
    if timestamps.size == 0:
        return TopicReport(
            topic=topic,
            message_count=0,
            mean_rate_hz=0.0,
            max_gap_seconds=0.0,
            issues=[Issue(Severity.CRITICAL, "no messages recorded")],
        )

    # Check if a topic has any gaps
    intervals = np.diff(timestamps)
    max_gap_seconds = float(intervals.max()) if intervals.size else 0.0
    median_interval = float(np.median(intervals)) if intervals.size else 0.0
    gap_threshold = max(config.stream_gap_factor * median_interval, config.stream_min_gap_seconds)

    issues = []
    if max_gap_seconds > gap_threshold:
        issues.append(Issue(Severity.CRITICAL, f"gap of {max_gap_seconds:.2f}s inside the stream"))
    edge_gaps = {"leading": timestamps[0] - window_start, "trailing": window_end - timestamps[-1]}
    for edge_name, edge_gap in edge_gaps.items():
        if edge_gap > gap_threshold:
            issues.append(Issue(Severity.WARNING, f"{edge_name} gap of {edge_gap:.2f}s versus the episode"))

    # Calculate the mean topic hz
    recorded_seconds = float(timestamps[-1] - timestamps[0])
    mean_rate_hz = (timestamps.size - 1) / recorded_seconds if recorded_seconds > 0 else 0.0

    return TopicReport(
        topic=topic,
        message_count=int(timestamps.size),
        mean_rate_hz=mean_rate_hz,
        max_gap_seconds=max_gap_seconds,
        issues=issues,
    )


def validate_depth_topics(
    mcap_path: Path, depth_camera: CameraParameters | None, depth_topics: list[str]
) -> list[Issue]:

    if not depth_topics:
        return []

    for _, _, depth_message in decode_topic_messages(mcap_path, depth_topics):
        try:
            depth_mm = decode_compressed_depth_mm(depth_message.format, bytes(depth_message.data))
            if depth_camera is not None:
                validate_depth_image_dims(depth_mm, depth_camera)
        except ValueError as error:
            return [Issue(Severity.CRITICAL, f"depth frame is unusable: {error}")]

        return []

    return []


def flag_topic_hz_degradation(episode_reports: list[EpisodeReport], fps_tolerance: float) -> None:
    reference = next((report for report in episode_reports if report.topics), None)
    if reference is None:
        return
    topics = [topic_report.topic for topic_report in reference.topics]

    # Check the hz for each topic across all episodes
    for topic_index, topic in enumerate(topics):
        rates = [
            report.topics[topic_index].mean_rate_hz
            for report in episode_reports
            if topic_index < len(report.topics) and report.topics[topic_index].message_count >= 2
        ]
        if not rates:
            continue

        # Calculate the median and minimum rates
        median_rate_hz = float(np.median(rates))
        minimum_rate_hz = median_rate_hz * (1 - fps_tolerance)

        # Flag episodes with rates below the minimum
        for report in episode_reports:
            if topic_index >= len(report.topics):
                continue
            topic_report = report.topics[topic_index]
            if topic_report.message_count >= 2 and topic_report.mean_rate_hz < minimum_rate_hz:
                message = (
                    f"rate {topic_report.mean_rate_hz:.1f}Hz is below the batch median {median_rate_hz:.1f}Hz"
                )
                topic_report.issues.append(Issue(Severity.WARNING, message))


def write_reports(episode_reports: list[EpisodeReport], report_directory: Path) -> Path:
    report_directory.mkdir(parents=True, exist_ok=True)
    json_path = report_directory / JSON_REPORT_FILE_NAME
    json_path.write_text(json.dumps([episode_report_to_dict(report) for report in episode_reports], indent=2))
    (report_directory / MARKDOWN_REPORT_FILE_NAME).write_text(format_markdown_report(episode_reports))
    return json_path


def episode_report_to_dict(report: EpisodeReport) -> dict:
    return {
        "episode_id": report.episode_id,
        "mcap_path": report.mcap_path,
        "severity": report.severity.value,
        "duration_seconds": report.duration_seconds,
        "episode_issues": [issue_to_dict(issue) for issue in report.episode_issues],
        "topics": [
            {
                "topic": topic_report.topic,
                "message_count": topic_report.message_count,
                "mean_rate_hz": topic_report.mean_rate_hz,
                "max_gap_seconds": topic_report.max_gap_seconds,
                "issues": [issue_to_dict(issue) for issue in topic_report.issues],
            }
            for topic_report in report.topics
        ],
    }


def issue_to_dict(issue: Issue) -> dict:
    return {"severity": issue.severity.value, "message": issue.message}


def format_markdown_report(episode_reports: list[EpisodeReport]) -> str:
    counts = {
        severity: sum(report.severity == severity for report in episode_reports) for severity in Severity
    }
    lines = [
        "# MCAP validation report",
        "",
        f"Episodes: {len(episode_reports)} (critical {counts[Severity.CRITICAL]}, "
        f"warning {counts[Severity.WARNING]}, pass {counts[Severity.PASS]})",
        "",
        "| Episode | Severity | Duration (s) | Issues |",
        "|---|---|---|---|",
    ]
    for report in episode_reports:
        issue_text = "; ".join(f"{issue.severity.value}: {issue.message}" for issue in report.all_issues)
        lines.append(
            f"| {report.episode_id} | {report.severity.value} | "
            f"{report.duration_seconds:.1f} | {issue_text} |"
        )
    return "\n".join(lines) + "\n"


def load_episode_severities(report_json_path: Path) -> dict[str, Severity]:
    entries = json.loads(report_json_path.read_text())
    return {entry["episode_id"]: Severity(entry["severity"]) for entry in entries}
