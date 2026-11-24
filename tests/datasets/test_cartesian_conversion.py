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
from dataclasses import asdict
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
    transform_wrench_to_target_frame,
)
from lerobot.datasets.joint_layout import (
    SENSOR_WRENCH_CHANNEL_NAMES,
    SENSOR_WRENCH_FEATURE_KEY,
    packed_feature_names,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.cartesian_quantities import (
    CartesianFrameSample,
    FramePoseRot6D,
    FrameTwist,
    FrameWrench,
    GripperChannels,
)
from lerobot.model.urdf_inspection import UR_CALIBRATION_JOINT_MAPPING
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.model.test_urdf_inspection import write_kinematics_calibration, write_ur_chain_urdf

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
    manifest = json.loads((converted.root / "cartesian_conversion_manifest.json").read_text())
    assert manifest["repo_id"] == dataset.repo_id
    assert manifest["root"] == str(dataset.root)
    assert manifest["new_repo_id"] == converted.repo_id
    assert manifest["new_root"] == str(converted.root)
    assert manifest["operation"] == json.loads(json.dumps(asdict(config)))
    assert manifest["converted"] == [
        {"episode_index": 0, "frame_count": 12},
        {"episode_index": 1, "frame_count": 12},
    ]
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


def test_conversion_manifest_records_the_joint_map_used_when_the_flag_was_unset(
    tmp_path, empty_lerobot_dataset_factory
):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)

    converted, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path, joint_map=None),
        output_dir=tmp_path / "cartesian",
        repo_id="local/cartesian",
    )

    manifest = json.loads((converted.root / "cartesian_conversion_manifest.json").read_text())
    assert manifest["operation"]["joint_map"] == JOINT_MAP
    assert manifest["operation"]["target_frame"] == "tcp"


def test_conversion_manifest_carries_episode_ids_from_the_source_mcap_manifest(
    tmp_path, empty_lerobot_dataset_factory
):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)
    source_manifest = {
        "converted": [
            {"episode_id": "episode-a", "episode_index": 0, "frame_count": 12},
            {"episode_id": "episode-b", "episode_index": 1, "frame_count": 12},
        ]
    }
    (dataset.root / "mcap_conversion_manifest.json").write_text(json.dumps(source_manifest))

    converted, _report = convert_joints_to_cartesian(
        dataset, _conversion_config(tmp_path), output_dir=tmp_path / "cartesian", repo_id="local/cartesian"
    )

    manifest = json.loads((converted.root / "cartesian_conversion_manifest.json").read_text())
    assert manifest["converted"] == [
        {"episode_id": "episode-a", "episode_index": 0, "frame_count": 12},
        {"episode_id": "episode-b", "episode_index": 1, "frame_count": 12},
    ]


@pytest.mark.parametrize(
    "source_manifest_text",
    [
        None,
        "not json",
        json.dumps([1, 2]),
        json.dumps({"converted": [{"episode_id": "episode-a", "episode_index": 0, "frame_count": 12}]}),
        json.dumps(
            {
                "converted": [
                    {"episode_id": "episode-a", "episode_index": 1, "frame_count": 12},
                    {"episode_id": "episode-b", "episode_index": 0, "frame_count": 12},
                ]
            }
        ),
    ],
    ids=[
        "no_manifest",
        "malformed_json",
        "not_an_object",
        "episode_count_mismatch",
        "episode_index_mismatch",
    ],
)
def test_conversion_manifest_leaves_out_episode_ids_without_a_matching_source_manifest(
    tmp_path, empty_lerobot_dataset_factory, source_manifest_text
):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)
    if source_manifest_text is not None:
        (dataset.root / "mcap_conversion_manifest.json").write_text(source_manifest_text)

    converted, _report = convert_joints_to_cartesian(
        dataset, _conversion_config(tmp_path), output_dir=tmp_path / "cartesian", repo_id="local/cartesian"
    )

    manifest = json.loads((converted.root / "cartesian_conversion_manifest.json").read_text())
    assert manifest["converted"] == [
        {"episode_index": 0, "frame_count": 12},
        {"episode_index": 1, "frame_count": 12},
    ]


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


def test_convert_position_only_action_gives_pose_and_gripper(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    state_names = packed_feature_names(DATASET_JOINTS, ["position", "effort"])
    action_names = packed_feature_names(DATASET_JOINTS, ["position"])
    dataset = _record_planar_dataset(
        tmp_path,
        empty_lerobot_dataset_factory,
        features={
            OBS_STATE: {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
            ACTION: {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        },
    )
    converted, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path),
        output_dir=tmp_path / "cartesian-position-action",
        repo_id="local/cartesian-position-action",
    )
    assert converted.meta.features[OBS_STATE]["names"] == EXPECTED_NAMES
    assert converted.meta.features[ACTION]["names"] == POSE_NAMES + ["gripper.position.x"]


def _record_planar_dataset_with_effort(tmp_path, empty_lerobot_dataset_factory):
    names = packed_feature_names(DATASET_JOINTS, ["position", "effort"])
    features = {OBS_STATE: {"dtype": "float32", "shape": (len(names),), "names": names}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "source", features=features, use_videos=False)
    for frame in range(12):
        joint_positions = [0.4 + 0.05 * frame, 0.9, 0.02]
        joint_efforts = [0.3, -0.2, 0.0]
        dataset.add_frame(
            {"task": "reach", OBS_STATE: np.array(joint_positions + joint_efforts, dtype=np.float32)}
        )
    dataset.save_episode()
    dataset.finalize()
    return dataset


def _read_state_rows(dataset: LeRobotDataset) -> np.ndarray:
    return np.stack([np.asarray(row) for row in dataset.hf_dataset[OBS_STATE]])


def test_default_wrench_is_expressed_in_the_end_effector_frame(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset_with_effort(tmp_path, empty_lerobot_dataset_factory)
    end_effector_frame, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path, subtract_gravity=False, evaluate=False),
        output_dir=tmp_path / "cartesian-local",
        repo_id="local/cartesian-local",
    )
    base_axes, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(
            tmp_path, subtract_gravity=False, evaluate=False, jacobian_frame_reference="local_world_aligned"
        ),
        output_dir=tmp_path / "cartesian-aligned",
        repo_id="local/cartesian-aligned",
    )

    provenance = json.loads((end_effector_frame.root / "meta" / "cartesian_conversion.json").read_text())
    assert provenance["jacobian_frame_reference"] == "local"
    end_effector_rows = _read_state_rows(
        LeRobotDataset("local/cartesian-local", root=end_effector_frame.root)
    )
    base_axes_rows = _read_state_rows(LeRobotDataset("local/cartesian-aligned", root=base_axes.root))
    wrench_start = len(POSE_NAMES) + 1
    for end_effector_row, base_axes_row in zip(end_effector_rows, base_axes_rows, strict=True):
        rotation = FramePoseRot6D.from_vector(base_axes_row[: len(POSE_NAMES)]).rotation
        base_axes_force = base_axes_row[wrench_start : wrench_start + 3]
        base_axes_torque = base_axes_row[wrench_start + 3 : wrench_start + 6]
        np.testing.assert_allclose(end_effector_row[:wrench_start], base_axes_row[:wrench_start], atol=1e-6)
        np.testing.assert_allclose(
            end_effector_row[wrench_start : wrench_start + 3], rotation.T @ base_axes_force, atol=1e-5
        )
        np.testing.assert_allclose(
            end_effector_row[wrench_start + 3 : wrench_start + 6], rotation.T @ base_axes_torque, atol=1e-5
        )
    assert np.abs(base_axes_rows[:, wrench_start:]).max() > 0.01


SENSOR_FRAME = "link2"
SENSOR_WRENCH = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])


def test_transform_wrench_to_target_frame_adds_the_torque_of_the_frame_offset():
    sensor_frame_in_world = np.eye(4)
    target_frame_in_world = np.eye(4)
    target_frame_in_world[:3, 3] = [1.0, 0.0, 0.0]
    wrench = transform_wrench_to_target_frame(
        SENSOR_WRENCH, sensor_frame_in_world, target_frame_in_world, "local"
    )
    np.testing.assert_allclose(wrench.force, SENSOR_WRENCH[:3])
    np.testing.assert_allclose(
        wrench.torque, SENSOR_WRENCH[3:] + np.cross([-1.0, 0.0, 0.0], SENSOR_WRENCH[:3])
    )


def test_transform_wrench_to_target_frame_returns_the_wrench_for_the_same_frame():
    frame_in_world = np.eye(4)
    frame_in_world[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    frame_in_world[:3, 3] = [0.3, -0.2, 0.5]
    wrench = transform_wrench_to_target_frame(SENSOR_WRENCH, frame_in_world, frame_in_world, "local")
    np.testing.assert_allclose(wrench.to_vector(), SENSOR_WRENCH, atol=1e-12)


@pytest.mark.parametrize("jacobian_frame_reference", ["local", "world", "local_world_aligned"])
def test_transformed_sensor_wrench_does_the_same_work_as_the_sensor_wrench(
    planar_kinematics, tmp_path, jacobian_frame_reference
):
    from lerobot.model.kinematics import RobotKinematics

    sensor_kinematics = RobotKinematics(
        str(_write_urdf(tmp_path)),
        target_frame_name=SENSOR_FRAME,
        joint_names=["follower_r_joint1", "follower_r_joint2"],
    )
    joint_positions_degrees = np.rad2deg([0.4, 0.9])
    joint_velocities = np.array([0.3, -0.7])
    sensor_twist = sensor_kinematics.frame_jacobian(joint_positions_degrees, "local") @ joint_velocities
    target_twist = (
        planar_kinematics.frame_jacobian(joint_positions_degrees, jacobian_frame_reference) @ joint_velocities
    )

    sample = convert_joint_frame(
        planar_kinematics,
        joint_positions_degrees,
        None,
        np.zeros(2),
        _layout(),
        gripper_by_field={"position": np.array([0.02])},
        jacobian_frame_reference=jacobian_frame_reference,
        sensor_wrench=SENSOR_WRENCH,
        wrench_sensor_frame=SENSOR_FRAME,
    )
    target_wrench = next(quantity for quantity in sample.quantities if isinstance(quantity, FrameWrench))

    assert target_wrench.to_vector() @ target_twist == pytest.approx(SENSOR_WRENCH @ sensor_twist)


def _record_planar_dataset_with_sensor_wrench(tmp_path, empty_lerobot_dataset_factory, action_fields):
    state_names = packed_feature_names(DATASET_JOINTS, ["position", "effort"])
    action_names = packed_feature_names(DATASET_JOINTS, action_fields)
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        ACTION: {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        SENSOR_WRENCH_FEATURE_KEY: {"dtype": "float32", "shape": (6,), "names": SENSOR_WRENCH_CHANNEL_NAMES},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "source", features=features, use_videos=False)
    for frame in range(12):
        state = np.array([0.4 + 0.05 * frame, 0.9, 0.02, 0.3, -0.2, 0.0], dtype=np.float32)
        dataset.add_frame(
            {
                "task": "reach",
                OBS_STATE: state,
                ACTION: state[: len(action_names)],
                SENSOR_WRENCH_FEATURE_KEY: (SENSOR_WRENCH * (1 + frame)).astype(np.float32),
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return dataset


def test_ft_sensor_wrench_source_writes_the_sensor_wrench_at_the_target_frame(
    tmp_path, empty_lerobot_dataset_factory
):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset_with_sensor_wrench(tmp_path, empty_lerobot_dataset_factory, ["position"])
    converted, _report = convert_joints_to_cartesian(
        dataset,
        _conversion_config(tmp_path, wrench_source="ft_sensor_wrench", wrench_sensor_frame=SENSOR_FRAME),
        output_dir=tmp_path / "cartesian",
        repo_id="local/cartesian",
    )

    reloaded = LeRobotDataset("local/cartesian", root=converted.root)
    assert SENSOR_WRENCH_FEATURE_KEY not in reloaded.meta.features
    assert reloaded.meta.features[OBS_STATE]["names"] == EXPECTED_NAMES
    wrench_start = len(POSE_NAMES) + 1
    for frame, state_row in enumerate(_read_state_rows(reloaded)):
        sensor_wrench = SENSOR_WRENCH * (1 + frame)
        # The tcp is 1 m along x from link2 with the same orientation.
        expected_torque = sensor_wrench[3:] + np.cross([-1.0, 0.0, 0.0], sensor_wrench[:3])
        np.testing.assert_allclose(state_row[wrench_start : wrench_start + 3], sensor_wrench[:3], rtol=1e-5)
        np.testing.assert_allclose(state_row[wrench_start + 3 :], expected_torque, rtol=1e-5)
    provenance = json.loads((converted.root / "meta" / "cartesian_conversion.json").read_text())
    assert provenance["wrench_source"] == "ft_sensor_wrench"
    assert provenance["wrench_sensor_frame"] == SENSOR_FRAME


def test_ft_sensor_wrench_source_raises_without_the_sensor_wrench_feature(
    tmp_path, empty_lerobot_dataset_factory
):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset_with_effort(tmp_path, empty_lerobot_dataset_factory)
    with pytest.raises(ValueError, match="--wrench_topic"):
        convert_joints_to_cartesian(
            dataset,
            _conversion_config(tmp_path, wrench_source="ft_sensor_wrench", wrench_sensor_frame=SENSOR_FRAME),
            output_dir=tmp_path / "cartesian",
            repo_id="local/cartesian",
        )


def test_ft_sensor_wrench_source_raises_on_an_action_with_effort(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset_with_sensor_wrench(
        tmp_path, empty_lerobot_dataset_factory, ["position", "effort"]
    )
    with pytest.raises(ValueError, match="action"):
        convert_joints_to_cartesian(
            dataset,
            _conversion_config(tmp_path, wrench_source="ft_sensor_wrench", wrench_sensor_frame=SENSOR_FRAME),
            output_dir=tmp_path / "cartesian",
            repo_id="local/cartesian",
        )


def test_ft_sensor_wrench_source_raises_on_a_sensor_frame_missing_from_the_urdf(
    tmp_path, empty_lerobot_dataset_factory
):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset_with_sensor_wrench(tmp_path, empty_lerobot_dataset_factory, ["position"])
    with pytest.raises(ValueError, match="tool0"):
        convert_joints_to_cartesian(
            dataset,
            _conversion_config(tmp_path, wrench_source="ft_sensor_wrench"),
            output_dir=tmp_path / "cartesian",
            repo_id="local/cartesian",
        )


UR_DATASET_JOINTS = [*UR_CALIBRATION_JOINT_MAPPING.values(), "gripper"]
UR_JOINT_MAP = {joint_name: f"robot_0_{joint_name}" for joint_name in UR_CALIBRATION_JOINT_MAPPING.values()}


def _record_ur_dataset(tmp_path, empty_lerobot_dataset_factory):
    names = packed_feature_names(UR_DATASET_JOINTS, ["position"])
    features = {OBS_STATE: {"dtype": "float32", "shape": (len(names),), "names": names}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "ur-source", features=features, use_videos=False)
    for frame in range(6):
        joint_positions = [0.1 * frame, 0.2, -0.3, 0.4, 0.5, -0.6, 0.02]
        dataset.add_frame({"task": "reach", OBS_STATE: np.array(joint_positions, dtype=np.float32)})
    dataset.save_episode()
    dataset.finalize()
    return dataset


def _ur_conversion_config(tmp_path, **overrides):
    config = {
        "urdf": str(write_ur_chain_urdf(tmp_path / "ur.urdf")),
        "target_frame": "link_5",
        "joint_map": UR_JOINT_MAP,
        "interactive": False,
        "plot_episodes": 0,
        "evaluate": False,
    }
    config.update(overrides)
    return JointToCartesianConversionConfig(**config)


def test_kinematics_calibration_sets_the_joint_origins_used_for_the_pose(
    tmp_path, empty_lerobot_dataset_factory
):
    pytest.importorskip("placo")
    dataset = _record_ur_dataset(tmp_path, empty_lerobot_dataset_factory)
    calibration_path = write_kinematics_calibration(tmp_path / "robot_0_kinematics.yaml")
    calibrated, _report = convert_joints_to_cartesian(
        dataset,
        _ur_conversion_config(tmp_path, kinematics_calibration_path=str(calibration_path)),
        output_dir=tmp_path / "cartesian-calibrated",
        repo_id="local/cartesian-calibrated",
    )
    uncalibrated, _report = convert_joints_to_cartesian(
        dataset,
        _ur_conversion_config(tmp_path),
        output_dir=tmp_path / "cartesian-uncalibrated",
        repo_id="local/cartesian-uncalibrated",
    )

    stored_urdf = (calibrated.root / "meta" / "kinematics.urdf").read_text()
    assert '<origin xyz="0.0 -0.2 0.3" rpy="0.01 -0.02 0.0"' in stored_urdf
    provenance = json.loads((calibrated.root / "meta" / "cartesian_conversion.json").read_text())
    assert provenance["kinematics_calibration"]["name"] == "robot_0_kinematics.yaml"
    assert provenance["kinematics_calibration"]["hash"] == "calib_123"
    assert len(provenance["kinematics_calibration"]["sha256"]) == 64
    calibrated_rows = _read_state_rows(LeRobotDataset("local/cartesian-calibrated", root=calibrated.root))
    uncalibrated_rows = _read_state_rows(
        LeRobotDataset("local/cartesian-uncalibrated", root=uncalibrated.root)
    )
    assert not np.allclose(calibrated_rows[:, :3], uncalibrated_rows[:, :3])


def test_kinematics_calibration_raises_when_a_calibrated_joint_is_not_in_the_joint_map(
    tmp_path, empty_lerobot_dataset_factory
):
    pytest.importorskip("placo")
    dataset = _record_planar_dataset(tmp_path, empty_lerobot_dataset_factory)
    calibration_path = write_kinematics_calibration(tmp_path / "robot_0_kinematics.yaml")
    with pytest.raises(ValueError, match="shoulder_pan_joint"):
        convert_joints_to_cartesian(
            dataset,
            _conversion_config(tmp_path, kinematics_calibration_path=str(calibration_path)),
            output_dir=tmp_path / "cartesian-planar-calibrated",
            repo_id="local/cartesian-planar-calibrated",
        )


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
