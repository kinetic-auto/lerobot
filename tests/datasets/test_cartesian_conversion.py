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
    JointToCartesianConversionConfig,
    JointSpaceLayout,
    convert_joint_frame,
    convert_joints_to_cartesian,
    propose_joint_map,
)
from lerobot.datasets.joint_layout import packed_feature_names
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.cartesian_quantities import (
    CartesianFrameSample,
    FramePoseRot6D,
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
EXPECTED_NAMES = POSE_NAMES + ["right_finger_joint1.position"] + WRENCH_NAMES + ["right_finger_joint1.effort"]
POSE_KEEP_EFFORT_NAMES = (
    POSE_NAMES + ["right_finger_joint1.position"] + packed_feature_names(DATASET_JOINTS, ["effort"])
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
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose = FramePoseRot6D(translation=np.array([1.0, 2.0, 3.0]), rotation=rotation)
    restored = FramePoseRot6D.from_vector(pose.to_vector())
    np.testing.assert_allclose(restored.translation, pose.translation)
    np.testing.assert_allclose(restored.rotation, pose.rotation)

    sample = CartesianFrameSample(
        quantities=(
            pose,
            GripperChannels(("right_finger_joint1",), np.array([0.1]), "position"),
            FrameWrench(force=np.array([1.0, 0.0, 0.0]), torque=np.zeros(3)),
            GripperChannels(("right_finger_joint1",), np.array([0.2]), "effort"),
        ),
        fields=("position", "effort"),
    )
    assert sample.feature_names() == EXPECTED_NAMES
    assert sample.to_vector().shape == (17,)


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
        evaluation_stride=4,
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
    rmse = provenance["evaluation"]["ik_round_trip"]["per_joint_rmse_rad"]
    assert max(rmse.values()) < 0.01
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
        "evaluation_stride": 4,
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
    assert provenance["evaluation"]["ik_round_trip"]["per_joint_rmse_rad"]
    assert provenance["evaluation"]["wrench_zero_check"] == {}


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
    assert converted.meta.features[OBS_STATE]["names"] == WRENCH_NAMES + ["right_finger_joint1.effort"]
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
