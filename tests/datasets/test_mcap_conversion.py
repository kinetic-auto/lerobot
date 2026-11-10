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
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("mcap", reason="mcap is required (install lerobot[mcap-dep])")
pytest.importorskip("mcap_ros2", reason="mcap-ros2-support is required (install lerobot[mcap-dep])")

from lerobot.datasets.mcap_conversion import (
    DepthAlignment,
    EpisodeStreams,
    McapConversionConfig,
    TimedMessages,
    build_features,
    build_state_matrix,
    build_time_grid,
    decode_aligned_depth_image,
    decode_depth_image,
    build_depth_encoder_config,
    depth_frame_decoder,
    find_joint_indices,
    latest_message_indices,
    load_allow_list,
    resolve_quality_report_path,
    filter_episodes_by_quality,
    validate_config,
)
from lerobot.datasets.mcap_depth import (
    COMPRESSED_DEPTH_HEADER_BYTES,
    CameraParameters,
    align_depth_to_rgb,
    build_transform_matrix,
    decode_compressed_depth_mm,
    load_depth_to_rgb_transform,
    load_optical_frame_wrt_camera_link,
    validate_depth_image_dims,
)
from lerobot.datasets.mcap_episodes import find_episode_paths
from lerobot.datasets.mcap_validation import (
    EpisodeReport,
    McapValidationConfig,
    Severity,
    TopicReport,
    build_topic_report,
    flag_topic_hz_degradation,
    load_episode_severities,
    write_reports,
)

IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


def make_camera() -> CameraParameters:
    return CameraParameters(
        camera_matrix=np.array([[100.0, 0.0, 16.0], [0.0, 100.0, 12.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        width=32,
        height=24,
    )


def make_conversion_config(tmp_path: Path, **overrides) -> McapConversionConfig:
    values = {
        "session_directory": str(tmp_path / "session"),
        "output_root": str(tmp_path / "dataset"),
        "repo_id": "local/test",
        "d405_color_intrinsics_path": "unused.yaml",
        "d405_depth_intrinsics_path": "unused.yaml",
    }
    values.update(overrides)
    return McapConversionConfig(**values)


def make_streams(joint_start: float, camera_start: float, end: float) -> EpisodeStreams:
    joint_timestamps = np.arange(joint_start, end, 0.01)
    gripper_timestamps = np.arange(joint_start, end, 0.05)
    camera_timestamps = np.arange(camera_start, end, 0.1)
    camera_messages = TimedMessages(camera_timestamps, [None] * len(camera_timestamps))
    camera = make_camera()
    return EpisodeStreams(
        joint_timestamps=joint_timestamps,
        joint_positions=np.stack([joint_timestamps, joint_timestamps + 100], axis=1),
        joint_efforts=np.stack([-joint_timestamps, -joint_timestamps - 100], axis=1),
        gripper_timestamps=gripper_timestamps,
        gripper_positions=gripper_timestamps + 1000,
        gripper_efforts=-gripper_timestamps - 1000,
        camera_frames={name: camera_messages for name in ("pc1", "realsense", "realsense_depth")},
        depth_alignment=DepthAlignment(camera, camera, np.eye(4)),
    )


def depth_payload(depth_millimeters: np.ndarray) -> tuple[str, bytes]:
    _, encoded_png = cv2.imencode(".encoded_png", depth_millimeters)
    return "16UC1; compressedDepth", bytes(COMPRESSED_DEPTH_HEADER_BYTES) + encoded_png.tobytes()


def test_decode_compressed_depth_roundtrip():
    depth_millimeters = (np.arange(12, dtype=np.uint16).reshape(3, 4) + 1) * 100
    _, encoded_png = cv2.imencode(".encoded_png", depth_millimeters)
    payload = bytes(COMPRESSED_DEPTH_HEADER_BYTES) + encoded_png.tobytes()
    decoded = decode_compressed_depth_mm("16UC1; compressedDepth", payload)
    np.testing.assert_array_equal(decoded, depth_millimeters)


def test_decode_compressed_depth_rejects_inverse_depth_format():
    with pytest.raises(ValueError, match="Unsupported compressedDepth format"):
        decode_compressed_depth_mm("32FC1; compressedDepth", b"")


def test_build_transform_matrix_normalizes_the_quaternion():
    non_unit_quarter_turn = (0.0, 0.0, 2 * np.sin(np.pi / 4), 2 * np.cos(np.pi / 4))
    transform = build_transform_matrix((0.0, 0.0, 0.0), non_unit_quarter_turn)
    np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-9)


def test_decode_aligned_depth_image_stores_no_return_at_the_far_end():
    camera = make_camera()
    depth_millimeters = np.full((24, 32), 500, dtype=np.uint16)
    depth_millimeters[:, :4] = 0
    _, encoded_png = cv2.imencode(".encoded_png", depth_millimeters)
    payload = ("16UC1; compressedDepth", bytes(COMPRESSED_DEPTH_HEADER_BYTES) + encoded_png.tobytes())
    alignment = DepthAlignment(camera, camera, np.eye(4))
    depth_image = decode_aligned_depth_image(payload, alignment, (32, 24), no_return_depth_mm=10000)
    assert depth_image.shape == (24, 32, 1)
    assert depth_image.dtype == np.uint16
    assert (depth_image[:, :4, 0] == 10000).all()
    assert (depth_image[:, 4:, 0] == 500).all()


def test_decode_depth_image_zeroes_depth_outside_the_configured_range():
    depth_millimeters = np.full((24, 32), 500, dtype=np.uint16)
    depth_millimeters[:, :4] = 0
    depth_millimeters[:, -4:] = 4000
    payload = depth_payload(depth_millimeters)
    depth_image = decode_depth_image(
        payload, (32, 24), no_return_depth_mm=10000, valid_depth_range_meters=(0.06, 2.0)
    )
    assert depth_image.dtype == np.uint16
    assert (depth_image[:, :4, 0] == 0).all()
    assert (depth_image[:, 4:-4, 0] == 500).all()
    assert (depth_image[:, -4:, 0] == 0).all()


def test_valid_depth_range_sets_the_encoder_floor_to_zero(tmp_path):
    depth_encoder_config = build_depth_encoder_config(
        make_conversion_config(tmp_path, valid_depth_range_meters=(0.06, 2.0))
    )

    assert depth_encoder_config is not None
    assert depth_encoder_config.depth_min == 0.0


def test_default_depth_encoder_is_kept_without_a_valid_depth_range(tmp_path):
    assert build_depth_encoder_config(make_conversion_config(tmp_path)) is None


def test_decode_depth_image_keeps_pixels_an_alignment_would_shift():
    depth_millimeters = np.full((24, 32), 500, dtype=np.uint16)
    depth_millimeters[:, :4] = 0
    payload = depth_payload(depth_millimeters)
    depth_image = decode_depth_image(payload, (32, 24), no_return_depth_mm=10000)
    assert (depth_image[:, :4, 0] == 10000).all()
    assert (depth_image[:, 4:, 0] == 500).all()
    # 10 mm along x at 500 mm depth and a 100 px focal length moves every pixel 2 columns right
    alignment = DepthAlignment(
        make_camera(), make_camera(), build_transform_matrix((0.010, 0.0, 0.0), IDENTITY_ROTATION)
    )
    aligned_image = decode_aligned_depth_image(payload, alignment, (32, 24), no_return_depth_mm=10000)
    assert (aligned_image[:, :2, 0] == 10000).all()
    assert not np.array_equal(depth_image, aligned_image)


def test_depth_frame_decoder_aligns_only_when_a_teleop_alignment_is_loaded():
    depth_millimeters = np.full((24, 32), 500, dtype=np.uint16)
    depth_millimeters[:, :4] = 0
    payloads = [depth_payload(depth_millimeters)]
    unaligned = depth_frame_decoder(None, (32, 24), 10000)(payloads, 0)
    alignment = DepthAlignment(
        make_camera(), make_camera(), build_transform_matrix((0.010, 0.0, 0.0), IDENTITY_ROTATION)
    )
    aligned = depth_frame_decoder(alignment, (32, 24), 10000)(payloads, 0)
    assert (unaligned[:, :4, 0] == 10000).all()
    assert (aligned[:, :2, 0] == 10000).all()


def test_decode_aligned_depth_image_rejects_another_stream_profile():
    depth_millimeters = np.full((48, 64), 500, dtype=np.uint16)
    _, encoded_png = cv2.imencode(".encoded_png", depth_millimeters)
    payload = ("16UC1; compressedDepth", bytes(COMPRESSED_DEPTH_HEADER_BYTES) + encoded_png.tobytes())
    alignment = DepthAlignment(make_camera(), make_camera(), np.eye(4))
    with pytest.raises(ValueError, match="stream profile changed"):
        decode_aligned_depth_image(payload, alignment, (32, 24), no_return_depth_mm=10000)


def test_link_from_optical_transform_names_the_file_without_the_extrinsics(tmp_path):
    intrinsics_path = tmp_path / "no_extrinsics.yaml"
    intrinsics_path.write_text("img_size: [32, 24]\n")
    with pytest.raises(ValueError, match="no_extrinsics.yaml has no camera_link_to_optical"):
        load_optical_frame_wrt_camera_link(intrinsics_path)


def test_find_episode_paths_rejects_split_bags(tmp_path):
    (tmp_path / "ep" / "bag").mkdir(parents=True)
    (tmp_path / "ep" / "bag" / "bag_0.mcap").touch()
    (tmp_path / "ep" / "bag" / "bag_1.mcap").touch()
    with pytest.raises(ValueError, match="share one episode id"):
        find_episode_paths(tmp_path)


def test_build_transform_matrix_rotates_about_z():
    quarter_turn = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    transform = build_transform_matrix((1.0, 2.0, 3.0), quarter_turn)
    np.testing.assert_allclose(transform @ np.array([1.0, 0.0, 0.0, 1.0]), [1.0, 3.0, 3.0, 1.0], atol=1e-9)


def write_intrinsics_yaml(
    path: Path, roll: float, pitch: float, yaw: float, x_translation: float = 0.0
) -> Path:
    path.write_text(
        f"camera_link_to_optical: {{roll: {roll}, pitch: {pitch}, yaw: {yaw}, x_T: {x_translation}, y_T: 0.0, z_T: 0.0}}\n"
    )
    return path


def test_depth_to_rgb_transform_is_identity_for_equal_optical_frames(tmp_path):
    depth_path = write_intrinsics_yaml(tmp_path / "depth.yaml", -90.0, 0.0, -90.0)
    color_path = write_intrinsics_yaml(tmp_path / "color.yaml", -90.0, 0.0, -90.0)
    np.testing.assert_allclose(load_depth_to_rgb_transform(depth_path, color_path), np.eye(4), atol=1e-12)


def test_depth_to_rgb_transform_maps_the_optical_axis_to_camera_link_forward(tmp_path):
    # The nominal optical frame looks along camera_link x, with image x pointing to link -y
    nominal_path = write_intrinsics_yaml(tmp_path / "nominal.yaml", -90.0, 0.0, -90.0)
    link_from_optical = load_optical_frame_wrt_camera_link(nominal_path)
    np.testing.assert_allclose(link_from_optical @ [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(link_from_optical @ [1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0], atol=1e-12)


def test_depth_to_rgb_transform_carries_the_small_factory_offsets(tmp_path):
    depth_path = write_intrinsics_yaml(tmp_path / "depth.yaml", -90.0, 0.0, -90.0)
    color_path = write_intrinsics_yaml(
        tmp_path / "color.yaml", -90.053671879, 0.002085174, -89.741140048, -1e-5
    )
    depth_frame_wrt_rgb_frame = load_depth_to_rgb_transform(depth_path, color_path)
    rotation_angle_degrees = np.degrees(np.arccos((np.trace(depth_frame_wrt_rgb_frame[:3, :3]) - 1) / 2))
    assert 0.2 < rotation_angle_degrees < 0.3
    # The optical frames sit about 10 micrometres apart, expressed in the color frame
    assert np.linalg.norm(depth_frame_wrt_rgb_frame[:3, 3]) == pytest.approx(1e-5, rel=0.01)


def test_validate_depth_image_dims_rejects_another_stream_profile():
    camera = make_camera()
    validate_depth_image_dims(np.zeros((24, 32), dtype=np.uint16), camera)
    with pytest.raises(ValueError, match="stream profile changed"):
        validate_depth_image_dims(np.zeros((48, 64), dtype=np.uint16), camera)


def test_align_depth_to_rgb_with_identity_keeps_depth():
    camera = make_camera()
    depth_millimeters = np.full((24, 32), 500, dtype=np.uint16)
    depth_millimeters[:, :4] = 0
    aligned = align_depth_to_rgb(depth_millimeters, camera, camera, np.eye(4))
    np.testing.assert_array_equal(aligned, depth_millimeters)


def test_align_depth_to_rgb_shifts_by_the_baseline():
    camera = make_camera()
    depth_millimeters = np.full((24, 32), 500, dtype=np.uint16)
    # 10 mm along x at 500 mm depth and a 100 px focal length moves every pixel 2 columns right
    depth_frame_wrt_rgb_frame = build_transform_matrix((0.010, 0.0, 0.0), IDENTITY_ROTATION)
    aligned = align_depth_to_rgb(depth_millimeters, camera, camera, depth_frame_wrt_rgb_frame)
    assert (aligned[:, 2:] == 500).all()
    assert (aligned[:, :2] == 0).all()


def test_find_joint_indices_reorders_by_name():
    assert find_joint_indices(["b", "c", "a"], ["a", "b", "c"]) == [2, 0, 1]


def test_find_joint_indices_names_the_missing_joint():
    with pytest.raises(ValueError, match="'z'"):
        find_joint_indices(["a", "b"], ["a", "z"])


def test_latest_message_indices_holds_the_previous_message():
    timestamps = np.array([0.0, 1.0, 2.0])
    time_grid = np.array([0.5, 1.0, 5.0, -1.0])
    np.testing.assert_array_equal(latest_message_indices(timestamps, time_grid), [0, 1, 2, 0])


def test_build_time_grid_starts_when_every_stream_has_data(tmp_path):
    config = make_conversion_config(tmp_path, fps=10, action_lookahead_frames=3)
    time_grid = build_time_grid(make_streams(joint_start=0.0, camera_start=1.0, end=5.0), config)
    assert time_grid[0] == pytest.approx(1.0)
    np.testing.assert_allclose(np.diff(time_grid), 0.1)
    assert time_grid[-1] <= 5.0


def test_build_time_grid_is_none_when_shorter_than_the_lookahead(tmp_path):
    config = make_conversion_config(tmp_path, fps=10, action_lookahead_frames=3)
    assert build_time_grid(make_streams(joint_start=0.0, camera_start=0.0, end=0.25), config) is None


def test_build_state_matrix_packs_positions_then_efforts():
    streams = make_streams(joint_start=0.0, camera_start=0.0, end=2.0)
    state = build_state_matrix(streams, np.array([1.0]))
    # joint 0, joint 1, gripper opening, then the same order for effort
    np.testing.assert_allclose(state[0], [1.0, 101.0, 1001.0, -1.0, -101.0, -1001.0], atol=0.06)


def test_build_features_flags_depth_and_sizes_cameras(tmp_path):
    config = make_conversion_config(
        tmp_path,
        img_sizes={"pc1": [64, 36], "realsense": [32, 18], "realsense_depth": [32, 18]},
    )
    features = build_features(config, ["a.position", "a.effort"])
    assert features["observation.images.pc1"]["shape"] == (36, 64, 3)
    assert features["observation.images.realsense"]["shape"] == (18, 32, 3)
    assert features["observation.images.realsense_depth"]["shape"] == (18, 32, 1)
    assert features["observation.images.realsense_depth"]["info"]["is_depth_map"] is True
    assert features["action"]["shape"] == (2,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fps": 0},
        {"action_lookahead_frames": 0},
        {"include_flagged": "bad"},
        {"img_sizes": {"pc1": [640], "realsense": [640, 360], "realsense_depth": [640, 360]}},
        {"no_return_depth_mm": 0},
        {"no_return_depth_mm": 70000},
        {"valid_depth_range_meters": (-0.1, 2.0)},
        {"valid_depth_range_meters": (2.0, 0.06)},
    ],
)
def test_validate_config_rejects_bad_values(tmp_path, overrides):
    with pytest.raises(ValueError):
        validate_config(make_conversion_config(tmp_path, **overrides))


def test_validate_config_requires_intrinsics_when_aligning_depth(tmp_path):
    aligning = make_conversion_config(
        tmp_path, d405_color_intrinsics_path=None, d405_depth_intrinsics_path=None
    )
    with pytest.raises(ValueError, match="align_depth requires"):
        validate_config(aligning)
    validate_config(
        make_conversion_config(
            tmp_path,
            align_depth=False,
            d405_color_intrinsics_path=None,
            d405_depth_intrinsics_path=None,
        )
    )


def write_session(tmp_path: Path, severities: dict[str, str]) -> None:
    session = tmp_path / "session"
    for episode_id in severities:
        (session / episode_id / "bag").mkdir(parents=True)
        (session / episode_id / "bag" / "bag_0.mcap").touch()
    report_directory = session / "mcap_valid_reports"
    report_directory.mkdir()
    entries = [{"episode_id": episode, "severity": severity} for episode, severity in severities.items()]
    (report_directory / "report.json").write_text(json.dumps(entries))


def test_filter_episodes_by_quality_skips_critical_by_default(tmp_path):
    write_session(tmp_path, {"ep_a": "pass", "ep_b": "warning", "ep_c": "critical"})
    selected_paths, skipped_reasons = filter_episodes_by_quality(make_conversion_config(tmp_path))
    assert [path.parent.parent.name for path in selected_paths] == ["ep_a", "ep_b"]
    assert "critical" in skipped_reasons["ep_c"]


def test_filter_episodes_by_quality_pass_only_skips_warnings(tmp_path):
    write_session(tmp_path, {"ep_a": "pass", "ep_b": "warning"})
    selected_paths, _ = filter_episodes_by_quality(make_conversion_config(tmp_path, include_flagged="pass"))
    assert [path.parent.parent.name for path in selected_paths] == ["ep_a"]


def test_filter_episodes_by_quality_honors_the_allow_list(tmp_path):
    write_session(tmp_path, {"ep_a": "pass", "ep_b": "pass"})
    allow_list_path = tmp_path / "allow.txt"
    allow_list_path.write_text("# good episodes\nep_b\n\n")
    config = make_conversion_config(tmp_path, episode_allow_list_path=str(allow_list_path))
    selected_paths, skipped_reasons = filter_episodes_by_quality(config)
    assert [path.parent.parent.name for path in selected_paths] == ["ep_b"]
    assert skipped_reasons == {"ep_a": "not in the allow list"}


def test_load_allow_list_ignores_comments_and_blank_lines(tmp_path):
    allow_list_path = tmp_path / "allow.txt"
    allow_list_path.write_text("# comment\nep_a\n\n  ep_b  \n")
    assert load_allow_list(str(allow_list_path)) == {"ep_a", "ep_b"}
    assert load_allow_list(None) is None


def test_conversion_requires_the_quality_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="lerobot-mcap-valid"):
        resolve_quality_report_path(make_conversion_config(tmp_path))


def test_empty_topic_is_critical():
    report = build_topic_report("/t", np.array([]), 0.0, 1.0, McapValidationConfig(session_directory="s"))
    assert report.message_count == 0
    assert report.issues[0].severity == Severity.CRITICAL


def test_gap_inside_a_stream_is_critical():
    timestamps = np.array([0.0, 0.1, 0.2, 2.0, 2.1])
    report = build_topic_report("/t", timestamps, 0.0, 2.1, McapValidationConfig(session_directory="s"))
    assert [issue.severity for issue in report.issues] == [Severity.CRITICAL]


def test_late_start_is_only_a_warning():
    timestamps = np.array([1.0, 1.1, 1.2])
    report = build_topic_report("/t", timestamps, 0.0, 1.2, McapValidationConfig(session_directory="s"))
    assert [issue.severity for issue in report.issues] == [Severity.WARNING]


def test_slow_episode_is_flagged_against_the_batch_median():
    def make_report(episode_id: str, rate_hz: float) -> EpisodeReport:
        topic_report = TopicReport("/t", message_count=50, mean_rate_hz=rate_hz, max_gap_seconds=0.1)
        return EpisodeReport(episode_id, "path", 5.0, [topic_report])

    reports = [make_report("a", 10.0), make_report("b", 10.0), make_report("c", 5.0)]
    flag_topic_hz_degradation(reports, fps_tolerance=0.15)
    assert [report.severity for report in reports] == [Severity.PASS, Severity.PASS, Severity.WARNING]


def test_report_roundtrip_keeps_each_episode_severity(tmp_path):
    clean = EpisodeReport("clean", "p", 1.0, [TopicReport("/t", 10, 10.0, 0.1)])
    empty = build_topic_report("/t", np.array([]), 0.0, 0.0, McapValidationConfig(session_directory="s"))
    broken = EpisodeReport("broken", "p", 1.0, [empty])
    json_path = write_reports([clean, broken], tmp_path / "reports")
    assert load_episode_severities(json_path) == {"clean": Severity.PASS, "broken": Severity.CRITICAL}
    assert (tmp_path / "reports" / "report.md").exists()
