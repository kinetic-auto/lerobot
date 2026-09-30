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

import numpy as np
import pytest

pytest.importorskip("placo")

from lerobot.model.kinematics import RobotKinematics

PLANAR_URDF = """<?xml version="1.0"?>
<robot name="planar">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <link name="link1"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" effort="10" velocity="1"/>
  </joint>
  <link name="link2"/>
  <joint name="tcp_joint" type="fixed">
    <parent link="link2"/>
    <child link="tcp"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <link name="tcp"/>
</robot>
"""


def _analytic_jacobian(q1: float, q2: float, length_1: float = 1.0, length_2: float = 1.0) -> np.ndarray:
    jacobian = np.zeros((6, 2))
    jacobian[0, 0] = -length_1 * np.sin(q1) - length_2 * np.sin(q1 + q2)
    jacobian[0, 1] = -length_2 * np.sin(q1 + q2)
    jacobian[1, 0] = length_1 * np.cos(q1) + length_2 * np.cos(q1 + q2)
    jacobian[1, 1] = length_2 * np.cos(q1 + q2)
    jacobian[5, 0] = 1.0
    jacobian[5, 1] = 1.0
    return jacobian


def test_frame_jacobian_matches_planar_arm(tmp_path):
    urdf_path = tmp_path / "planar.urdf"
    urdf_path.write_text(PLANAR_URDF)
    kinematics = RobotKinematics(str(urdf_path), target_frame_name="tcp", joint_names=["joint1", "joint2"])
    q_rad = np.array([0.4, -0.7])
    jacobian = kinematics.frame_jacobian(np.rad2deg(q_rad), use_deg=True)
    assert jacobian.shape == (6, 2)
    np.testing.assert_allclose(jacobian, _analytic_jacobian(*q_rad), atol=1e-6)
    np.testing.assert_allclose(kinematics.frame_jacobian(q_rad, use_deg=False), jacobian, atol=1e-6)
    np.testing.assert_allclose(kinematics.joint_limits("joint2"), [-2.0, 2.0])
