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

import xml.etree.ElementTree as ET  # nosec B405 test fixture, not network XML

import pytest

from lerobot.model.urdf_inspection import (
    UR_CALIBRATION_JOINT_MAPPING,
    apply_arm_kinematics_calibration,
    movable_joint_chain_to_frame,
    parse_urdf_joints,
    read_arm_kinematics_calibration,
    write_kinematics_only_urdf,
)

PLANAR_URDF = """<?xml version="1.0"?>
<robot name="planar">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <link name="link1">
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <link name="link2"/>
  <joint name="tcp_joint" type="fixed">
    <parent link="link2"/>
    <child link="tcp"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <link name="tcp"/>
  <joint name="finger" type="prismatic">
    <parent link="link2"/>
    <child link="finger_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
  <link name="finger_link"/>
</robot>
"""


def test_parse_order_chain_and_kinematics_only_copy(tmp_path):
    source = tmp_path / "planar.urdf"
    source.write_text(PLANAR_URDF)
    joints = parse_urdf_joints(source)
    assert [joint.name for joint in joints] == ["joint1", "joint2", "tcp_joint", "finger"]

    chain = movable_joint_chain_to_frame(joints, "tcp")
    assert [joint.name for joint in chain] == ["joint1", "joint2"]

    chain_from_joint = movable_joint_chain_to_frame(joints, "tcp_joint")
    assert [joint.name for joint in chain_from_joint] == ["joint1", "joint2"]
    assert chain_from_joint[0].child_link == "link1"

    kinematics_only_urdf = write_kinematics_only_urdf(source, tmp_path / "kinematics.urdf")
    text = kinematics_only_urdf.read_text()
    assert "<visual" not in text
    assert "<collision" not in text
    assert [joint.name for joint in parse_urdf_joints(kinematics_only_urdf)] == [
        "joint1",
        "joint2",
        "tcp_joint",
        "finger",
    ]


def test_parse_skips_nested_ros2_control_joints(tmp_path):
    source = tmp_path / "nested.urdf"
    source.write_text(
        """<?xml version="1.0"?>
<robot name="nested">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
  </joint>
  <link name="link1"/>
  <ros2_control name="hw" type="system">
    <joint name="joint1">
      <command_interface name="position"/>
    </joint>
  </ros2_control>
</robot>
"""
    )
    joints = parse_urdf_joints(source)
    assert [joint.name for joint in joints] == ["joint1"]
    assert joints[0].parent_link == "base_link"
    assert joints[0].child_link == "link1"


def write_ur_chain_urdf(path, joint_prefix="robot_0_"):
    joint_elements = []
    parent_link = "base_link"
    for joint_index, joint_name in enumerate(UR_CALIBRATION_JOINT_MAPPING.values()):
        child_link = f"link_{joint_index}"
        joint_elements.append(
            f"""  <joint name="{joint_prefix}{joint_name}" type="revolute">
    <parent link="{parent_link}"/>
    <child link="{child_link}"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.28" upper="6.28" effort="10" velocity="1"/>
  </joint>
  <link name="{child_link}"/>"""
        )
        parent_link = child_link
    path.write_text(
        '<?xml version="1.0"?>\n<robot name="ur">\n  <link name="base_link"/>\n'
        + "\n".join(joint_elements)
        + "\n</robot>\n"
    )
    return path


def write_kinematics_calibration(path, link_names=None):
    link_names = list(UR_CALIBRATION_JOINT_MAPPING) if link_names is None else link_names
    lines = ["kinematics:"]
    for link_index, link_name in enumerate(link_names):
        lines.append(f"  {link_name}:")
        for key, value in zip(
            ("x", "y", "z", "roll", "pitch", "yaw"),
            (0.1 * link_index, -0.2, 0.3, 0.01, -0.02, 0.03 * link_index),
            strict=True,
        ):
            lines.append(f"    {key}: {value}")
    lines.append("  hash: calib_123")
    path.write_text("\n".join(lines) + "\n")
    return path


def calibration_joint_mapping(joint_prefix="robot_0_"):
    return {
        link_name: f"{joint_prefix}{joint_name}"
        for link_name, joint_name in UR_CALIBRATION_JOINT_MAPPING.items()
    }


def test_arm_kinematics_calibration_updates_each_calibrated_joint_origin(tmp_path):
    urdf_path = write_ur_chain_urdf(tmp_path / "ur.urdf")
    calibration_path = write_kinematics_calibration(tmp_path / "robot_0_kinematics.yaml")

    apply_arm_kinematics_calibration(urdf_path, calibration_path, calibration_joint_mapping())

    calibration = read_arm_kinematics_calibration(calibration_path)
    assert calibration["hash"] == "calib_123"
    origin_by_joint = {
        element.attrib["name"]: element.find("origin").attrib
        for element in ET.parse(urdf_path).getroot().iter("joint")  # nosec B314
    }
    for link_name, joint_name in calibration_joint_mapping().items():
        expected_position = [calibration[link_name][key] for key in ("x", "y", "z")]
        expected_orientation = [calibration[link_name][key] for key in ("roll", "pitch", "yaw")]
        assert [float(value) for value in origin_by_joint[joint_name]["xyz"].split()] == expected_position
        assert [float(value) for value in origin_by_joint[joint_name]["rpy"].split()] == expected_orientation


def test_arm_kinematics_calibration_raises_on_a_missing_link(tmp_path):
    urdf_path = write_ur_chain_urdf(tmp_path / "ur.urdf")
    calibration_path = write_kinematics_calibration(
        tmp_path / "robot_0_kinematics.yaml", link_names=["shoulder", "upper_arm"]
    )
    with pytest.raises(ValueError, match="forearm"):
        apply_arm_kinematics_calibration(urdf_path, calibration_path, calibration_joint_mapping())


def test_arm_kinematics_calibration_raises_on_a_missing_joint(tmp_path):
    urdf_path = write_ur_chain_urdf(tmp_path / "ur.urdf", joint_prefix="")
    calibration_path = write_kinematics_calibration(tmp_path / "robot_0_kinematics.yaml")
    with pytest.raises(ValueError, match="robot_0_shoulder_pan_joint"):
        apply_arm_kinematics_calibration(urdf_path, calibration_path, calibration_joint_mapping())


def test_arm_kinematics_calibration_raises_on_a_link_missing_a_key(tmp_path):
    urdf_path = write_ur_chain_urdf(tmp_path / "ur.urdf")
    calibration_path = write_kinematics_calibration(tmp_path / "robot_0_kinematics.yaml")
    calibration_path.write_text(calibration_path.read_text().replace("    yaw: 0.0\n", "", 1))
    with pytest.raises(ValueError, match="shoulder"):
        apply_arm_kinematics_calibration(urdf_path, calibration_path, calibration_joint_mapping())


def test_arm_kinematics_calibration_raises_without_a_kinematics_mapping(tmp_path):
    calibration_path = tmp_path / "robot_0_kinematics.yaml"
    calibration_path.write_text("shoulder: 1\n")
    with pytest.raises(ValueError, match="kinematics"):
        read_arm_kinematics_calibration(calibration_path)
