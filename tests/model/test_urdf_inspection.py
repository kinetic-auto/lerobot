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

from lerobot.model.urdf_inspection import (
    movable_joint_chain_to_frame,
    parse_urdf_joints,
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
