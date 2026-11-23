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

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 local robot description, not network XML
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

MOVABLE_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic"})
_MESH_ELEMENTS = frozenset({"visual", "collision"})
UR_CALIBRATION_JOINT_MAPPING = {
    "shoulder": "shoulder_pan_joint",
    "upper_arm": "shoulder_lift_joint",
    "forearm": "elbow_joint",
    "wrist_1": "wrist_1_joint",
    "wrist_2": "wrist_2_joint",
    "wrist_3": "wrist_3_joint",
}
_CALIBRATION_POSITION_KEYS = ("x", "y", "z")
_CALIBRATION_ORIENTATION_KEYS = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str


def parse_urdf_joints(urdf_path: str | Path) -> list[UrdfJoint]:
    """Read joints from a URDF in document order.

    Args:
        urdf_path (str | Path): URDF path.

    Returns:
        list[UrdfJoint]: Joints.
    """
    root = ET.parse(urdf_path).getroot()  # nosec B314
    joints: list[UrdfJoint] = []
    # Only robot-level joints. Nested <joint> tags (ros2_control, gazebo) are not URDF joints.
    for element in root:
        if _local_name(element.tag) != "joint":
            continue
        parent_link = _link_attribute(element, "parent")
        child_link = _link_attribute(element, "child")
        name = element.attrib["name"]
        if not parent_link or not child_link:
            raise ValueError(f"Joint {name!r} is missing a parent or child link.")
        joints.append(
            UrdfJoint(
                name=name,
                joint_type=element.attrib["type"],
                parent_link=parent_link,
                child_link=child_link,
            )
        )
    return joints


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _link_attribute(joint: ET.Element, role: str) -> str:
    for child in joint:
        if _local_name(child.tag) == role:
            return child.attrib.get("link", "")
    return ""


def movable_joints(joints: list[UrdfJoint]) -> list[UrdfJoint]:
    """Keep revolute, continuous, and prismatic joints.

    Args:
        joints (list[UrdfJoint]): Joints.

    Returns:
        list[UrdfJoint]: Movable joints.
    """
    return [joint for joint in joints if joint.joint_type in MOVABLE_JOINT_TYPES]


def movable_joint_chain_to_frame(joints: list[UrdfJoint], frame_name: str) -> list[UrdfJoint]:
    """Return movable joints that move a frame, root first.

    Args:
        joints (list[UrdfJoint]): Joints.
        frame_name (str): Link or joint name.

    Returns:
        list[UrdfJoint]: Movable joints on the path.
    """
    start_link = _frame_link(joints, frame_name)
    child_to_joint = {joint.child_link: joint for joint in joints}
    path: list[UrdfJoint] = []
    seen: set[str] = set()
    link = start_link
    while link in child_to_joint:
        joint = child_to_joint[link]
        if joint.name in seen:
            raise ValueError(f"URDF joint graph has a cycle at {joint.name!r}.")
        seen.add(joint.name)
        path.append(joint)
        link = joint.parent_link
    path.reverse()
    return [joint for joint in path if joint.joint_type in MOVABLE_JOINT_TYPES]


def _frame_link(joints: list[UrdfJoint], frame_name: str) -> str:
    links = {joint.parent_link for joint in joints} | {joint.child_link for joint in joints}
    if frame_name in links:
        return frame_name
    for joint in joints:
        if joint.name == frame_name:
            return joint.child_link
    raise ValueError(f"Frame {frame_name!r} is not a link or joint in the URDF.")


def frame_candidates(joints: list[UrdfJoint], chain: list[UrdfJoint]) -> list[str]:
    """List link and joint names along a chain, leaf first.

    Args:
        joints (list[UrdfJoint]): All joints.
        chain (list[UrdfJoint]): Movable joints, root first.

    Returns:
        list[str]: Frame names.
    """
    if not chain:
        return []
    root_first: list[str] = [chain[0].parent_link]
    for joint in chain:
        root_first.append(joint.name)
        root_first.append(joint.child_link)
    children: dict[str, list[UrdfJoint]] = {}
    for joint in joints:
        children.setdefault(joint.parent_link, []).append(joint)
    chain_names = {joint.name for joint in chain}

    def append_fixed(link: str) -> None:
        for joint in children.get(link, []):
            if joint.name in chain_names or joint.joint_type != "fixed":
                continue
            root_first.append(joint.name)
            root_first.append(joint.child_link)
            append_fixed(joint.child_link)

    append_fixed(chain[-1].child_link)
    leaf_first: list[str] = []
    seen: set[str] = set()
    for name in reversed(root_first):
        if name and name not in seen:
            seen.add(name)
            leaf_first.append(name)
    return leaf_first


def write_kinematics_only_urdf(urdf_path: str | Path, output_path: str | Path) -> Path:
    """Copy a URDF with visual and collision elements removed.

    Args:
        urdf_path (str | Path): Source URDF.
        output_path (str | Path): Destination path.

    Returns:
        Path: Written URDF.
    """
    tree = ET.parse(urdf_path)  # nosec B314
    root = tree.getroot()
    _strip_namespaces(root)
    for parent in root.iter():
        for child in list(parent):
            if _local_name(child.tag) in _MESH_ELEMENTS:
                parent.remove(child)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, xml_declaration=True, encoding="unicode")
    return destination


def _strip_namespaces(element: ET.Element) -> None:
    element.tag = _local_name(element.tag)
    element.attrib = {_local_name(key): value for key, value in element.attrib.items()}
    for child in list(element):
        _strip_namespaces(child)


def apply_arm_kinematics_calibration(
    urdf_path: str | Path,
    kinematics_calibration_path: str | Path,
    calibration_joint_mapping: dict[str, str],
) -> Path:
    calibration = read_arm_kinematics_calibration(kinematics_calibration_path)
    missing_links = [link_name for link_name in calibration_joint_mapping if link_name not in calibration]
    if missing_links:
        raise ValueError(f"Kinematics calibration {kinematics_calibration_path} lacks links {missing_links}.")
    incomplete_links = [
        link_name
        for link_name in calibration_joint_mapping
        if not isinstance(calibration[link_name], dict)
        or any(
            key not in calibration[link_name]
            for key in (*_CALIBRATION_POSITION_KEYS, *_CALIBRATION_ORIENTATION_KEYS)
        )
    ]
    if incomplete_links:
        raise ValueError(
            f"Kinematics calibration {kinematics_calibration_path} links {incomplete_links} "
            f"need every one of {[*_CALIBRATION_POSITION_KEYS, *_CALIBRATION_ORIENTATION_KEYS]}."
        )

    tree = ET.parse(urdf_path)  # nosec B314
    joint_by_name = {
        element.attrib["name"]: element for element in tree.getroot() if _local_name(element.tag) == "joint"
    }
    missing_joints = [name for name in calibration_joint_mapping.values() if name not in joint_by_name]
    if missing_joints:
        raise ValueError(f"URDF {urdf_path} lacks the calibrated joints {missing_joints}.")

    for link_name, joint_name in calibration_joint_mapping.items():
        _update_joint_origin(joint_by_name[joint_name], calibration[link_name])
    tree.write(urdf_path, xml_declaration=True, encoding="unicode")
    return Path(urdf_path)


def read_arm_kinematics_calibration(kinematics_calibration_path: str | Path) -> dict[str, Any]:
    with open(kinematics_calibration_path) as calibration_file:
        calibration_data = yaml.safe_load(calibration_file)
    if not isinstance(calibration_data, dict) or not isinstance(calibration_data.get("kinematics"), dict):
        raise ValueError(f"Kinematics calibration {kinematics_calibration_path} has no 'kinematics' mapping.")
    return calibration_data["kinematics"]


def _update_joint_origin(joint: ET.Element, link_calibration: dict[str, float]) -> None:
    origin = next((child for child in joint if _local_name(child.tag) == "origin"), None)
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    origin.attrib["xyz"] = " ".join(repr(float(link_calibration[key])) for key in _CALIBRATION_POSITION_KEYS)
    origin.attrib["rpy"] = " ".join(repr(float(link_calibration[key])) for key in _CALIBRATION_ORIENTATION_KEYS)
