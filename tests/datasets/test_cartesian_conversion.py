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

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.cartesian_conversion import (
    JointSpaceLayout,
    JointToCartesianConversionConfig,
    _contact_mask,
    _contact_segments,
    _force_magnitude_histogram,
    _free_motion_mask,
    _wrench_baseline,
    convert_joint_frame,
    convert_joints_to_cartesian,
    propose_joint_map,
)
from lerobot.datasets.joint_layout import packed_feature_names
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.cartesian_quantities import (
    CartesianFrameSample,
    FramePoseRot6D,
    FrameTwist,
    FrameWrench,
    GripperChannels,
)
from lerobot.utils.constants import ACTION, OBS_STATE

PLANAR_URDF = """<?xml version="1.0"?>
<robot name="planar">
  <link name="base_link"/>
  <joint name="follower_r_joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <link name="link1"/>
  <joint name="follower_r_joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <link name="link2"/>
  <joint name="tcp_joint" type="fixed">
    <parent link="link2"/>
    <child link="tcp"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <link name="tcp"/>
  <joint name="follower_r_finger_joint1" type="prismatic">
    <parent link="link2"/>
    <child link="finger_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.05" effort="1" velocity="0.1"/>
  </joint>
  <link name="finger_link"/>
</robot>
"""

JOINT_MAP = {
    "right_joint1": "follower_r_joint1",
    "right_joint2": "follower_r_joint2",
}
DATASET_JOINTS = ["right_joint1", "right_joint2", "right_finger_joint1"]
POSE_NAMES = FramePoseRot6D.feature_names()
WRENCH_NAMES = FrameWrench.feature_names()
EXPECTED_NAMES = POSE_NAMES + ["gripper.position.x"] + WRENCH_NAMES
POSE_KEEP_EFFORT_NAMES = (
    POSE_NAMES + ["gripper.position.x"] + packed_feature_names(DATASET_JOINTS, ["effort"])
)


def test_propose_joint_map_matches_suffix_and_rejects_unmatched():
    mapping = propose_joint_map(
        ["right_finger_joint1", "right_joint1", "right_joint2"],
        ["follower_r_joint1", "follower_r_joint2"],
    )
    assert mapping == JOINT_MAP
    with pytest.raises(ValueError, match="follower_r_joint9"):
        propose_joint_map(["right_joint1"], ["follower_r_joint1", "follower_r_joint9"])


def test_pose_round_trip_and_sample_feature_names():
    assert POSE_NAMES == [
        "ee.position.x",
        "ee.position.y",
        "ee.position.z",
        "ee.orientation.rot6d.0",
        "ee.orientation.rot6d.1",
        "ee.orientation.rot6d.2",
        "ee.orientation.rot6d.3",
        "ee.orientation.rot6d.4",
        "ee.orientation.rot6d.5",
    ]
    assert FrameTwist.feature_names() == [
        "ee.linear.x",
        "ee.linear.y",
        "ee.linear.z",
        "ee.angular.x",
        "ee.angular.y",
        "ee.angular.z",
    ]
    assert WRENCH_NAMES == [
        "ee.force.x",
        "ee.force.y",
        "ee.force.z",
        "ee.torque.x",
        "ee.torque.y",
        "ee.torque.z",
    ]
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose = FramePoseRot6D(translation=np.array([1.0, 2.0, 3.0]), rotation=rotation)
    restored = FramePoseRot6D.from_vector(pose.to_vector())
    np.testing.assert_allclose(restored.translation, pose.translation)
    np.testing.assert_allclose(restored.rotation, pose.rotation)

    sample = CartesianFrameSample(
        quantities=(
            pose,
            GripperChannels(("gripper.position.x",), np.array([0.1]), "position"),
            FrameWrench(force=np.array([1.0, 0.0, 0.0]), torque=np.zeros(3)),
        ),
        fields=("position", "effort"),
    )
    assert sample.feature_names() == EXPECTED_NAMES
    assert sample.to_vector().shape == (16,)


def _write_urdf(directory: Path) -> Path:
    path = directory / "robot.urdf"
    path.write_text(PLANAR_URDF)
    return path


def _layout() -> JointSpaceLayout:
    names = packed_feature_names(DATASET_JOINTS, ["position", "effort"])
    layout = JointSpaceLayout.parse_feature({"names": names, "dtype": "float32", "shape": (len(names),)})
    layout.split_arm_and_gripper(["right_joint1", "right_joint2"])
    return layout


@pytest.fixture
def planar_kinematics(tmp_path):
    pytest.importorskip("placo")
    from lerobot.model.kinematics import RobotKinematics

    return RobotKinematics(
        str(_write_urdf(tmp_path)),
        target_frame_name="tcp",
        joint_names=["follower_r_joint1", "follower_r_joint2"],
    )


def test_convert_joint_frame_matches_analytic_fk_and_wrench(planar_kinematics):
    q_rad = np.array([0.4, 0.9])
    jacobian = planar_kinematics.frame_jacobian(np.rad2deg(q_rad))
    wrench = jacobian @ np.array([0.3, -0.2])
    torque = jacobian.T @ wrench
    sample = convert_joint_frame(
        planar_kinematics,
        np.rad2deg(q_rad),
        None,
        torque,
        _layout(),
        gripper_by_field={"position": np.array([0.02]), "effort": np.array([0.0])},
        subtract_gravity=False,
    )
    pose = next(quantity for quantity in sample.quantities if isinstance(quantity, FramePoseRot6D))
    solved = next(quantity for quantity in sample.quantities if isinstance(quantity, FrameWrench))
    expected_x = np.cos(q_rad[0]) + np.cos(q_rad.sum())
    expected_y = np.sin(q_rad[0]) + np.sin(q_rad.sum())
    np.testing.assert_allclose(pose.translation, [expected_x, expected_y, 0.0], atol=1e-6)
    np.testing.assert_allclose(solved.to_vector(), wrench, atol=1e-6)
    np.testing.assert_allclose(jacobian.T @ solved.to_vector(), torque, atol=1e-6)


def test_convert_joint_frame_keeps_joint_effort(planar_kinematics):
    layout = _layout()
    layout.convert_fields = ["position"]
    q_rad = np.array([0.4, 0.9])
    sample = convert_joint_frame(
        planar_kinematics,
        np.rad2deg(q_rad),
        None,
        None,
        layout,
        gripper_by_field={"position": np.array([0.02])},
        kept_by_field={"effort": np.array([0.3, -0.2, 0.0])},
        subtract_gravity=False,
    )
    assert sample.feature_names() == POSE_KEEP_EFFORT_NAMES
    np.testing.assert_allclose(sample.to_vector()[-3:], [0.3, -0.2, 0.0])


def _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory, features=None):
    names = packed_feature_names(DATASET_JOINTS, ["position", "effort"])
    if features is None:
        features = {
            OBS_STATE: {"dtype": "float32", "shape": (len(names),), "names": names},
            ACTION: {"dtype": "float32", "shape": (len(names),), "names": names},
        }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "source", features=features, use_videos=False)
    for episode in range(2):
        for frame in range(12):
            q1 = 0.4 + 0.002 * (episode * 12 + frame)
            values = dict(zip(names, np.array([q1, 0.9, 0.02, 0.0, 0.0, 0.0], dtype=np.float32), strict=True))
            row = {"task": "reach"}
            for key, feature in features.items():
                row[key] = np.array([values[name] for name in feature["names"]], dtype=np.float32)
            dataset.add_frame(row)
        dataset.save_episode()
    dataset.finalize()
    return dataset


def test_convert_joints_to_cartesian(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)
    source_names = list(dataset.meta.features[OBS_STATE]["names"])
    config = JointToCartesianConversionConfig(
        urdf=str(_write_urdf(tmp_path)),
        target_frame="tcp",
        joint_map=JOINT_MAP,
        interactive=False,
        plot_episodes=0,
    )
    converted, _report = convert_joints_to_cartesian(
        dataset,
        config,
        output_dir=tmp_path / "cartesian",
        repo_id="local/cartesian",
    )

    assert dataset.meta.features[OBS_STATE]["names"] == source_names
    for key in (OBS_STATE, ACTION):
        assert converted.meta.features[key]["names"] == EXPECTED_NAMES
    assert (converted.root / "meta" / "kinematics.urdf").is_file()
    provenance_path = converted.root / "meta" / "cartesian_conversion.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["target_frame"] == "tcp"
    assert provenance["joint_units"] == "rad"
    assert provenance["orientation"] == "rotation_6d_columns"
    assert provenance["feature_keys"] == [OBS_STATE, ACTION]
    assert provenance["convert_fields"] is None
    assert provenance["channel_names"][OBS_STATE] == EXPECTED_NAMES
    stats = json.loads((converted.root / "meta" / "stats.json").read_text())
    assert len(stats[OBS_STATE]["mean"]) == len(EXPECTED_NAMES)
    assert len(stats[ACTION]["mean"]) == len(EXPECTED_NAMES)
    episodes = pd.read_parquet(converted.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert len(episodes["stats/observation.state/mean"].iloc[0]) == len(EXPECTED_NAMES)
    assert "fk_residual" not in provenance["evaluation"]
    ik_round_trip = provenance["evaluation"]["ik_round_trip"]
    for key in (OBS_STATE, ACTION):
        per_joint = ik_round_trip[key]["per_joint_abs_error_rad"]
        assert per_joint
        for summary in per_joint.values():
            assert set(summary) >= {"mean", "median", "min", "max"}
            assert summary["max"] < 0.01
    wrench_stats = provenance["evaluation"]["wrench_stats"]
    for key in (OBS_STATE, ACTION):
        stats = wrench_stats[key]
        regimes = stats["regimes"]
        assert regimes["free"]["frames"] + regimes["contact"]["frames"] == 24
        assert stats["baseline"]["frames"] > 0
        assert sum(item["fraction"] for item in stats["force_histogram"]) == pytest.approx(1.0)
    reloaded = LeRobotDataset(converted.repo_id, root=converted.root)
    assert len(reloaded) == 24
    assert reloaded.meta.features[OBS_STATE]["names"] == EXPECTED_NAMES


def _conversion_config(tmp_path, **overrides):
    config = {
        "urdf": str(_write_urdf(tmp_path)),
        "target_frame": "tcp",
        "joint_map": JOINT_MAP,
        "interactive": False,
        "plot_episodes": 0,
    }
    config.update(overrides)
    return JointToCartesianConversionConfig(**config)


def test_convert_position_keeps_joint_effort(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)
    converted, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path, convert_fields=["position"]),
        output_dir=tmp_path / "cartesian-pose",
        repo_id="local/cartesian-pose",
    )
    for key in (OBS_STATE, ACTION):
        assert converted.meta.features[key]["names"] == POSE_KEEP_EFFORT_NAMES
    provenance = json.loads((converted.root / "meta" / "cartesian_conversion.json").read_text())
    assert provenance["convert_fields"] == ["position"]
    for key in (OBS_STATE, ACTION):
        per_joint = provenance["evaluation"]["ik_round_trip"][key]["per_joint_abs_error_rad"]
        assert max(summary["max"] for summary in per_joint.values()) < 0.01
    assert provenance["evaluation"]["wrench_stats"] == {}


def test_convert_action_only_leaves_state(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)
    source_state = list(dataset.meta.features[OBS_STATE]["names"])
    converted, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path, feature_keys=[ACTION]),
        output_dir=tmp_path / "cartesian-action",
        repo_id="local/cartesian-action",
    )
    assert converted.meta.features[OBS_STATE]["names"] == source_state
    assert converted.meta.features[ACTION]["names"] == EXPECTED_NAMES
    provenance = json.loads((converted.root / "meta" / "cartesian_conversion.json").read_text())
    assert provenance["feature_keys"] == [ACTION]
    assert OBS_STATE not in provenance["channel_names"]


def test_convert_action_without_state(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    names = packed_feature_names(DATASET_JOINTS, ["position", "effort"])
    dataset = _record_planar_dataset(
        tmp_path,
        empty_lerobot_dataset_factory,
        features={ACTION: {"dtype": "float32", "shape": (len(names),), "names": names}},
    )
    converted, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path),
        output_dir=tmp_path / "cartesian-action-only",
        repo_id="local/cartesian-action-only",
    )
    assert OBS_STATE not in converted.meta.features
    assert converted.meta.features[ACTION]["names"] == EXPECTED_NAMES


def test_convert_state_effort_uses_action_positions(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    effort_names = packed_feature_names(DATASET_JOINTS, ["effort"])
    action_names = packed_feature_names(DATASET_JOINTS, ["position", "effort"])
    dataset = _record_planar_dataset(
        tmp_path,
        empty_lerobot_dataset_factory,
        features={
            OBS_STATE: {"dtype": "float32", "shape": (len(effort_names),), "names": effort_names},
            ACTION: {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        },
    )
    converted, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path),
        output_dir=tmp_path / "cartesian-effort-state",
        repo_id="local/cartesian-effort-state",
    )
    assert converted.meta.features[OBS_STATE]["names"] == WRENCH_NAMES
    assert converted.meta.features[ACTION]["names"] == EXPECTED_NAMES


def test_interactive_false_without_frame_raises(tmp_path, empty_lerobot_dataset_factory):
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)
    config = JointToCartesianConversionConfig(
        urdf=str(_write_urdf(tmp_path)),
        interactive=False,
        evaluate=False,
        plot_episodes=0,
    )
    with pytest.raises(ValueError, match="target_frame"):
        convert_joints_to_cartesian(
            dataset,
            config,
            output_dir=tmp_path / "cartesian",
            repo_id="local/cartesian",
        )


def _contact_config(**overrides) -> JointToCartesianConversionConfig:
    values = {
        "interactive": False,
        "plot_episodes": 0,
        "evaluate": False,
    }
    values.update(overrides)
    return JointToCartesianConversionConfig(**values)


def test_gripper_open_side_rejects_invalid():
    with pytest.raises(ValueError, match="gripper_open_side"):
        JointToCartesianConversionConfig(gripper_open_side="left")


def test_contact_mask_removes_short_runs_and_stays_in_episode():
    force = np.zeros(18)
    force[3:6] = 5.0
    force[8] = 5.0
    force[9] = 5.0
    episode = np.repeat([0, 1], 9)
    mask = _contact_mask(force, 1.0, 3, episode)
    expected = np.zeros(18, dtype=bool)
    expected[3:6] = True
    np.testing.assert_array_equal(mask, expected)


def test_contact_mask_fills_short_free_gaps():
    force = np.array([5.0, 5.0, 5.0, 0.0, 5.0, 5.0, 5.0, 5.0])
    episode = np.zeros(8, dtype=np.int64)
    mask = _contact_mask(force, 1.0, 2, episode)
    np.testing.assert_array_equal(mask, np.ones(8, dtype=bool))


def test_free_motion_mask_picks_still_and_open_frames():
    config = _contact_config(
        free_motion_speed_threshold_m_s=0.02,
        contact_min_frames=2,
        gripper_open_side="max",
        gripper_open_tolerance=0.1,
    )
    translation = np.zeros((10, 3))
    translation[:, 0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0]
    gripper = np.ones(10)
    episode = np.zeros(10, dtype=np.int64)
    mask, source = _free_motion_mask(translation, gripper, episode, 10.0, config)
    assert source == "still_and_open"
    expected = np.array([True, True, True, True, True, True, False, False, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_free_motion_mask_falls_back_to_episode_start():
    config = _contact_config(
        free_motion_speed_threshold_m_s=0.02,
        contact_min_frames=2,
        gripper_open_side="max",
        gripper_open_tolerance=0.1,
    )
    translation = np.zeros((10, 3))
    translation[:, 0] = np.arange(10, dtype=np.float64)
    gripper = np.zeros(10)
    episode = np.zeros(10, dtype=np.int64)
    mask, source = _free_motion_mask(translation, gripper, episode, 10.0, config)
    assert source == "episode_start"
    np.testing.assert_array_equal(mask, np.array([True] * 6 + [False] * 4))


def test_wrench_baseline_percentiles_on_known_array():
    force = np.zeros((20, 3))
    force[:, 0] = 4.0
    torque = np.zeros((20, 3))
    torque[:, 0] = 2.0
    mask = np.ones(20, dtype=bool)
    baseline = _wrench_baseline(force, torque, mask)
    assert baseline["frames"] == 20
    np.testing.assert_allclose(baseline["mean_force_vector_n"], [4.0, 0.0, 0.0])
    assert baseline["force_median_n"] == pytest.approx(4.0)
    assert baseline["force_p95_n"] == pytest.approx(4.0)
    assert baseline["torque_median_nm"] == pytest.approx(2.0)
    assert baseline["torque_p95_nm"] == pytest.approx(2.0)


def test_contact_segments_counts_runs_and_seconds():
    contact = np.array([False, False, True, True, True, False, True, True] + [False] * 8)
    episode = np.repeat([0, 1], 8)
    stats = _contact_segments(contact, episode, 10.0)
    assert stats["total"] == 2
    assert stats["with_contact"] == 1
    assert stats["segments_total"] == 2
    assert stats["segment_seconds"]["mean"] == pytest.approx(0.25)
    assert stats["segments_per_episode"]["mean"] == pytest.approx(1.0)


def test_force_magnitude_histogram_edges_and_fractions():
    magnitude = np.array([0.0, 4.9, 5.0, 12.0, 25.0, 40.0, 80.0])
    items = _force_magnitude_histogram(magnitude)
    assert [item["range_n"] for item in items] == ["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", ">50"]
    assert sum(item["frames"] for item in items) == 7
    assert sum(item["fraction"] for item in items) == pytest.approx(1.0)
    by_range = {item["range_n"]: item["frames"] for item in items}
    assert by_range["0-5"] == 2
    assert by_range["5-10"] == 1
    assert by_range[">50"] == 1
