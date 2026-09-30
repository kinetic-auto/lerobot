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

import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.model.cartesian_quantities import (
    CartesianFrameSample,
    FramePoseRot6D,
    FrameTwist,
    FrameWrench,
    GripperChannels,
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.model.urdf_inspection import (
    UrdfJoint,
    frame_candidates,
    movable_joint_chain_to_frame,
    parse_urdf_joints,
    write_kinematics_only_urdf,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import flatten_dict

from .compute_stats import compute_episode_stats
from .dataset_tools import modify_features, recompute_stats
from .io_utils import write_episodes
from .joint_layout import VALID_JOINT_FIELDS, feature_channel_names, io_block_layout
from .lerobot_dataset import LeRobotDataset
from .utils import DATA_DIR, DEFAULT_EPISODES_PATH

_EE_NAME_HINTS = ("tcp", "hand", "ee", "wrist")
_JOINT_ANGLE_UNITS = ("rad", "deg")
_JACOBIAN_FRAME_REFERENCES = ("local", "world", "local_world_aligned")
_MAX_STILL_JOINT_DELTA_RAD = 1e-2
_SINGULAR_JACOBIAN_COND_THRESHOLD = 1e3
_MAX_IK_JOINT_ERROR_RAD = float(np.deg2rad(1.0))

# Default IK uses a tiny orientation weight and 8 iterations, which leaves a joint
# residual above the round-trip tolerance even when the pose came from those joints.
_IK_POSITION_WEIGHT = 1.0
_IK_ORIENTATION_WEIGHT = 1.0
_IK_MAX_ITERS = 30

_CONVERTIBLE_FEATURE_KEYS = (OBS_STATE, ACTION)
_JOINT_FEATURE_KEYS = {OBS_STATE: "observation.joint_state", ACTION: "action.joints"}


@dataclass
class JointToCartesianConversionConfig:
    # Path to the robot URDF. Visual and collision meshes are stripped before load.
    urdf: str = ""
    # URDF link or joint to express the end effector in. Prompted when unset.
    target_frame: str | None = None
    # Dataset joint name to URDF joint name. Prompted when unset.
    joint_map: dict[str, str] | None = None
    # rad or deg. Default is radians.
    joint_units: str = "rad"
    # placo Jacobian frame: local, world, or local_world_aligned.
    jacobian_frame_reference: str = "local_world_aligned"
    # Subtract static gravity torques before solving the end-effector wrench.
    subtract_gravity: bool = True
    # Prompt for the frame and joint map when they are not set.
    interactive: bool = True
    # Write the IK round-trip and trajectory report.
    evaluate: bool = True
    # Frames between IK round-trip samples.
    evaluation_stride: int = 10
    # Episodes of the new dataset to plot. Zero skips plotting.
    plot_episodes: int = 3
    # Also store the original joint vectors as observation.joint_state and action.joints.
    keep_joint_features: bool = False
    # observation.state and/or action. Default: every one of those keys present on the dataset.
    feature_keys: list[str] | None = None
    # position, velocity, and/or effort. Default: every packed field on that feature.
    convert_fields: list[str] | None = None


@dataclass
class JointSpaceLayout:
    joint_names: list[str]
    fields: list[str]
    index: dict[tuple[str, str], int]
    arm_joints: list[str] = field(default_factory=list)
    gripper_joints: list[str] = field(default_factory=list)
    convert_fields: list[str] | None = None

    def resolve_convert_fields(self) -> list[str]:
        existing_fields = [field_name for field_name in self.fields if field_name in VALID_JOINT_FIELDS]
        if self.convert_fields is None:
            return existing_fields
        conversion_requested_fields = set(self.convert_fields)

        return [field_name for field_name in existing_fields if field_name in conversion_requested_fields]

    def has_fields_to_convert(self) -> bool:
        return bool(self.resolve_convert_fields())

    @classmethod
    def parse_feature(cls, feature: Mapping[str, Any]) -> JointSpaceLayout:
        names = feature_channel_names(feature)

        # Extract the joint names and fields from the feature names.
        try:
            joint_names, fields = io_block_layout(names)
        except ValueError as exc:
            raise ValueError(f"Feature names are not packed '<joint>.<field>' blocks: {names}") from exc

        # Create a mapping of joint names and fields to their positions in the feature.
        index = {
            (name.rpartition(".")[0], name.rpartition(".")[2]): position
            for position, name in enumerate(names)
        }

        return cls(joint_names=list(joint_names), fields=list(fields), index=index)

    def split_arm_and_gripper(self, arm_joints: Sequence[str]) -> None:
        missing = [name for name in arm_joints if name not in self.joint_names]
        if missing:
            raise ValueError(f"Joint map names {missing} are not in the feature.")
        arm = list(arm_joints)
        self.arm_joints = arm
        arm_set = set(arm)
        self.gripper_joints = [name for name in self.joint_names if name not in arm_set]


@dataclass(frozen=True)
class JointToCartesianConversionReport:
    ik_round_trip: dict[str, Any]
    trajectory: dict[str, Any]
    wrench_zero_check: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ik_round_trip": self.ik_round_trip,
            "trajectory": self.trajectory,
            "wrench_zero_check": self.wrench_zero_check,
        }


def convert_joints_to_cartesian(
    dataset: LeRobotDataset,
    config: JointToCartesianConversionConfig,
    output_dir: str | Path,
    repo_id: str,
) -> tuple[LeRobotDataset, JointToCartesianConversionReport]:
    """Write a dataset with Cartesian frame quantities.

    Args:
        dataset (LeRobotDataset): Source dataset.
        config (JointToCartesianConversionConfig): Conversion settings.
        output_dir (str | Path): Output directory.
        repo_id (str): Dataset id.

    Returns:
        tuple[LeRobotDataset, JointToCartesianConversionReport]: Converted dataset and report.
    """
    # Validate the configuration.
    _validate_config(config, dataset, output_dir)

    # Resolve features and packed layouts.
    feature_layouts: dict[str, JointSpaceLayout] = {}
    for key in _resolve_feature_keys(config, dataset):
        feature_layout = JointSpaceLayout.parse_feature(dataset.meta.features[key])

        feature_layout.convert_fields = config.convert_fields

        if feature_layout.resolve_convert_fields():
            feature_layouts[key] = feature_layout

    if not feature_layouts:
        raise ValueError("No features have requested fields to convert.")

    keys_to_convert = list(feature_layouts)

    # Resolve the URDF chain, joint map, and arm/gripper split.
    dataset_joint_names = _union_joint_names(list(feature_layouts.values()))
    urdf_joints = parse_urdf_joints(config.urdf)
    kinematic_chain_to_target_frame, joint_map, target_frame = _resolve_frame_and_map(
        urdf_joints, dataset_joint_names, config
    )
    arm_joints = _order_arm_dataset_joints(
        joint_map, [joint.name for joint in kinematic_chain_to_target_frame], dataset_joint_names
    )

    # Split the arm and gripper joints for each feature layout.
    for feature_layout in feature_layouts.values():
        feature_layout.split_arm_and_gripper(arm_joints)

    # Load a sibling feature when conversion needs joint positions this feature lacks.
    load_keys = list(keys_to_convert)
    position_layouts: dict[str, JointSpaceLayout] = {}
    if any(
        layout.has_fields_to_convert() and "position" not in layout.fields
        for layout in feature_layouts.values()
    ):
        for key in _CONVERTIBLE_FEATURE_KEYS:
            if key in feature_layouts or key not in dataset.meta.features:
                continue
            extra = JointSpaceLayout.parse_feature(dataset.meta.features[key])
            if "position" not in extra.fields:
                continue
            extra.split_arm_and_gripper(arm_joints)
            position_layouts[key] = extra
            load_keys.append(key)
            break

    # Read packed columns.
    columns = _read_parquet_columns(dataset, [*load_keys, "episode_index"])
    matrices = {key: columns[key] for key in load_keys}
    episode_index = np.asarray(columns["episode_index"], dtype=np.int64).reshape(-1)
    if len(episode_index) == 0:
        raise ValueError("Dataset has no frames to convert.")
    for key, matrix in matrices.items():
        if len(matrix) != len(episode_index):
            raise ValueError(f"{key} and episode_index have different lengths.")

    units = config.joint_units
    logging.info("Using joint units: %s", units)

    # Convert the features and write the new dataset.
    with tempfile.TemporaryDirectory() as tmp:
        # Build kinematics from a mesh-stripped URDF.
        kinematics_urdf = write_kinematics_only_urdf(config.urdf, Path(tmp) / "kinematics.urdf")
        kinematics = RobotKinematics(
            urdf_path=str(kinematics_urdf),
            target_frame_name=target_frame,
            joint_names=[joint.name for joint in kinematic_chain_to_target_frame],
        )

        # Convert selected features.
        converted_vectors: dict[str, np.ndarray] = {}
        converted_names: dict[str, list[str]] = {}
        for key in keys_to_convert:
            feature_layout = feature_layouts[key]
            q_key, q_layout = _resolve_q_feature(key, feature_layout, feature_layouts, position_layouts)
            uses_sibling_positions = q_key is not None and q_key != key
            converted_vectors[key], converted_names[key] = _convert_joint_matrix(
                matrices[key],
                feature_layout,
                kinematics,
                units,
                config,
                kinematic_chain_to_target_frame[0].parent_link,
                q_matrix=matrices[q_key] if uses_sibling_positions and q_key is not None else None,
                q_layout=q_layout if uses_sibling_positions else None,
            )

        # Evaluate the conversion.
        report = JointToCartesianConversionReport(ik_round_trip={}, trajectory={}, wrench_zero_check={})
        if config.evaluate:
            report = evaluate_conversion(
                kinematics,
                {
                    key: _convert_matrix_to_radians(matrices[key], feature_layouts[key], units)
                    for key in keys_to_convert
                    if "position" in feature_layouts[key].fields
                },
                converted_vectors,
                next(iter(feature_layouts.values())),
                config.evaluation_stride,
                episode_indices=episode_index,
                jacobian_frame_reference=config.jacobian_frame_reference,
                feature_layouts=feature_layouts,
            )
            _log_report(report)

        # Write the new dataset.
        new_dataset = _write_converted_dataset(
            dataset,
            config,
            output_dir,
            repo_id,
            {key: matrices[key] for key in keys_to_convert},
            converted_vectors,
            converted_names,
        )
        shutil.copy(kinematics_urdf, Path(new_dataset.root) / "meta" / "kinematics.urdf")

    # Recompute stats and write provenance.
    recompute_stats(new_dataset)
    written = _read_parquet_columns(new_dataset, [*keys_to_convert, "episode_index"])
    _rewrite_episode_stats(
        new_dataset,
        keys_to_convert,
        {key: written[key] for key in keys_to_convert},
        np.asarray(written["episode_index"], dtype=np.int64).reshape(-1),
    )
    _write_provenance(
        new_dataset,
        config,
        target_frame,
        joint_map,
        units,
        converted_names,
        report if config.evaluate else None,
    )
    _plot_episodes(new_dataset, config.plot_episodes)
    return new_dataset, report


def _validate_config(
    config: JointToCartesianConversionConfig, dataset: LeRobotDataset, output_dir: str | Path
) -> None:
    if config.target_frame is None and not config.interactive:
        raise ValueError("target_frame is required when interactive is false.")
    if not config.urdf:
        raise ValueError("urdf is required.")
    if not Path(config.urdf).is_file():
        raise ValueError(f"URDF not found: {config.urdf}")
    if config.joint_units not in _JOINT_ANGLE_UNITS:
        raise ValueError(f"joint_units must be one of {_JOINT_ANGLE_UNITS}, got {config.joint_units!r}.")
    if config.jacobian_frame_reference not in _JACOBIAN_FRAME_REFERENCES:
        raise ValueError(
            f"jacobian_frame_reference must be one of {_JACOBIAN_FRAME_REFERENCES}, got {config.jacobian_frame_reference!r}."
        )
    if config.evaluation_stride < 1:
        raise ValueError("evaluation_stride must be >= 1.")
    if config.plot_episodes < 0:
        raise ValueError("plot_episodes must be >= 0.")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output dataset directory already exists: {output}")
    if output.resolve() == Path(dataset.root).resolve():
        raise ValueError("joints_to_cartesian refuses to overwrite the input dataset.")
    if config.feature_keys is not None:
        if not config.feature_keys:
            raise ValueError("feature_keys is empty.")
        invalid = [key for key in config.feature_keys if key not in _CONVERTIBLE_FEATURE_KEYS]
        if invalid:
            raise ValueError(f"feature_keys must be observation.state and/or action, got {invalid}.")
        missing = [key for key in config.feature_keys if key not in dataset.meta.features]
        if missing:
            raise ValueError(f"Dataset is missing {missing}.")
    elif not any(key in dataset.meta.features for key in _CONVERTIBLE_FEATURE_KEYS):
        raise ValueError("Dataset is missing observation.state and action.")
    if config.convert_fields is not None:
        if not config.convert_fields:
            raise ValueError("convert_fields is empty.")
        invalid = [field_name for field_name in config.convert_fields if field_name not in VALID_JOINT_FIELDS]
        if invalid:
            raise ValueError(f"convert_fields must be position, velocity, and/or effort, got {invalid}.")


def _resolve_frame_and_map(
    joints: list[UrdfJoint],
    dataset_joint_names: Sequence[str],
    config: JointToCartesianConversionConfig,
) -> tuple[list[UrdfJoint], dict[str, str], str]:
    # Resolve the target frame and generate the kinematic chain to it.
    if config.target_frame:
        chain = movable_joint_chain_to_frame(joints, config.target_frame)
        target_frame = config.target_frame
    else:
        chain = _infer_kinematic_chain(joints, dataset_joint_names)
        target_frame = select_target_frame(frame_candidates(joints, chain), None)
        chain = movable_joint_chain_to_frame(joints, target_frame)

    # Validate the kinematic chain.
    if not chain:
        raise ValueError(f"Frame {target_frame!r} has no movable joints.")

    urdf_chain_joint_names = [joint.name for joint in chain]
    if config.joint_map is not None:
        joint_map = dict(config.joint_map)
    else:
        joint_map = propose_joint_map(dataset_joint_names, urdf_chain_joint_names)
        if config.interactive:
            gripper_joints = [name for name in dataset_joint_names if name not in joint_map]
            if gripper_joints:
                print("Gripper joints: " + ", ".join(gripper_joints))
            joint_map = select_joint_map(joint_map)
    return chain, joint_map, target_frame


def _infer_kinematic_chain(joints: list[UrdfJoint], dataset_joint_names: Sequence[str]) -> list[UrdfJoint]:
    # Get the frame names from the URDF chain.
    frame_names = {joint.parent_link for joint in joints} | {joint.child_link for joint in joints}
    frame_names.update(joint.name for joint in joints)

    # Find the best matching chain.
    best: list[UrdfJoint] | None = None

    # Find longest mappable chain among frames whose names look like an end effector.
    for frame_name in frame_names:
        if not _has_ee_name_hint(frame_name):
            continue
        chain = _resolve_chain_for_frame(joints, frame_name, dataset_joint_names)
        if chain is not None and (best is None or len(chain) > len(best)):
            best = chain

    if best is not None:
        return best

    # Find longest mappable chain among child links.
    for link in {joint.child_link for joint in joints}:
        chain = _resolve_chain_for_frame(joints, link, dataset_joint_names)
        if chain is not None and (best is None or len(chain) > len(best)):
            best = chain

    if best is None:
        raise ValueError("No URDF chain matches the dataset joints.")

    return best


def _resolve_chain_for_frame(
    joints: list[UrdfJoint], frame_name: str, dataset_joint_names: Sequence[str]
) -> list[UrdfJoint] | None:
    # Build the movable joint chain to the frame.
    try:
        chain = movable_joint_chain_to_frame(joints, frame_name)
    except ValueError:
        return None

    # Reject frames with no movable joints.
    if not chain:
        return None

    # Match every URDF chain joint to a dataset joint.
    try:
        propose_joint_map(dataset_joint_names, [joint.name for joint in chain])
    except ValueError:
        return None
    return chain


def select_target_frame(candidates: list[str], preselected: str | None) -> str:
    """Choose a URDF frame, prompting when none was given.

    Args:
        candidates (list[str]): Frame names, leaf first.
        preselected (str | None): Frame to keep without prompting.

    Returns:
        str: Frame name.
    """
    # Return the preselected frame if provided.
    if preselected is not None:
        return preselected

    # Reject an empty candidate list.
    if not candidates:
        raise ValueError("No frame candidates to select from.")

    # Find frames whose names look like an end effector.
    suggested_target_frame_names = [name for name in candidates if _has_ee_name_hint(name)]

    # Get the user input for the target frame
    default = suggested_target_frame_names[0] if suggested_target_frame_names else candidates[0]
    print("Select the target frame:")
    for index, name in enumerate(candidates):
        marks = []
        if name in suggested_target_frame_names:
            marks.append("suggested")
        if name == default:
            marks.append("default")
        suffix = f" ({', '.join(marks)})" if marks else ""
        print(f"  {index}: {name}{suffix}")
    raw = input(f"Frame [{default}]: ").strip()
    if raw == "":
        return default
    if raw.isdigit():
        choice = int(raw)
        if choice < 0 or choice >= len(candidates):
            raise ValueError(f"Frame index {choice} is out of range.")
        return candidates[choice]
    if raw in candidates:
        return raw
    raise ValueError(f"Frame {raw!r} is not in the candidate list.")


def _has_ee_name_hint(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _EE_NAME_HINTS)


def propose_joint_map(dataset_joint_names: Sequence[str], chain_joint_names: Sequence[str]) -> dict[str, str]:
    """Match each chain joint to the dataset joint with the longest common token suffix.

    Args:
        dataset_joint_names (Sequence[str]): Dataset joint names.
        chain_joint_names (Sequence[str]): URDF joint names, root first.

    Returns:
        dict[str, str]: Dataset joint name to URDF joint name.
    """
    pairs: list[tuple[int, str, str]] = []
    for urdf_name in chain_joint_names:
        for dataset_name in dataset_joint_names:
            length = _common_token_suffix_length(dataset_name, urdf_name)
            if length >= 1:
                pairs.append((length, dataset_name, urdf_name))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    mapping: dict[str, str] = {}
    used_dataset: set[str] = set()
    used_urdf: set[str] = set()
    for length, dataset_name, urdf_name in pairs:
        if dataset_name in used_dataset or urdf_name in used_urdf:
            continue
        rivals = [
            other
            for other_length, other, other_urdf in pairs
            if other_urdf == urdf_name and other not in used_dataset and other_length == length
        ]
        # ``right_joint1`` and ``right_finger_joint1`` share the token ``joint1``.
        # The name with fewer tokens is the closer match.
        if len(rivals) > 1:
            fewest = min(_token_count(name) for name in rivals)
            rivals = [name for name in rivals if _token_count(name) == fewest]
        if len(rivals) > 1:
            raise ValueError(
                f"URDF joint {urdf_name!r} matches {rivals} with the same suffix length. Pass joint_map."
            )
        chosen = rivals[0]
        mapping[chosen] = urdf_name
        used_dataset.add(chosen)
        used_urdf.add(urdf_name)
    unmatched = [name for name in chain_joint_names if name not in used_urdf]
    if unmatched:
        raise ValueError(f"No dataset joint matches URDF chain joints: {unmatched}")
    inverse = {urdf_name: dataset_name for dataset_name, urdf_name in mapping.items()}
    return {inverse[urdf_name]: urdf_name for urdf_name in chain_joint_names}


def _common_token_suffix_length(left: str, right: str) -> int:
    left_tokens = left.split("_")
    right_tokens = right.split("_")
    length = 0
    for left_token, right_token in zip(reversed(left_tokens), reversed(right_tokens), strict=False):
        if left_token != right_token:
            break
        length += 1
    return length


def _token_count(name: str) -> int:
    return len(name.split("_"))


def select_joint_map(proposal: dict[str, str]) -> dict[str, str]:
    """Confirm a joint map, or swap assignments by line number.

    Args:
        proposal (dict[str, str]): Dataset joint name to URDF joint name.

    Returns:
        dict[str, str]: Accepted map.
    """
    mapping = dict(proposal)
    dataset_names = list(mapping)
    while True:
        print("Proposed joint map (dataset joint -> URDF joint):")
        items = list(mapping.items())
        for index, (dataset_name, urdf_name) in enumerate(items):
            print(f"  {index}: {dataset_name} -> {urdf_name}")
        raw = input("Press Enter to accept, or a line number to reassign: ").strip()
        if raw == "" or raw.lower() in {"y", "yes"}:
            return mapping
        if not raw.isdigit() or int(raw) >= len(items):
            print("Enter a line number, or press Enter to accept.")
            continue
        line = int(raw)
        print("Dataset joints:")
        for index, name in enumerate(dataset_names):
            print(f"  {index}: {name}")
        choice = input("Dataset joint number: ").strip()
        if not choice.isdigit() or int(choice) >= len(dataset_names):
            print("Invalid joint number.")
            continue
        chosen = dataset_names[int(choice)]
        current = items[line][0]
        if chosen != current:
            mapping[chosen], mapping[current] = mapping[current], mapping[chosen]


def _resolve_feature_keys(config: JointToCartesianConversionConfig, dataset: LeRobotDataset) -> list[str]:
    # Find all features in the dataset that can be converted.
    available_features = [key for key in _CONVERTIBLE_FEATURE_KEYS if key in dataset.meta.features]

    # Return all available features if no feature keys are requested.
    if config.feature_keys is None:
        return available_features

    # Return only the requested features if requested.
    requested_features = set(config.feature_keys)
    return [key for key in _CONVERTIBLE_FEATURE_KEYS if key in requested_features]


def _union_joint_names(layouts: Sequence[JointSpaceLayout]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for layout in layouts:
        for name in layout.joint_names:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _resolve_q_feature(
    key: str,
    layout: JointSpaceLayout,
    feature_layouts: Mapping[str, JointSpaceLayout],
    position_layouts: Mapping[str, JointSpaceLayout],
) -> tuple[str | None, JointSpaceLayout | None]:
    # Skip features that are not converting Cartesian quantities.
    if not layout.has_fields_to_convert():
        return None, None

    # Use the feature being converted as the q source.
    if "position" in layout.fields:
        return key, layout

    # Search a sibling feature for a full arm position block.
    for source_key, source_layout in {**feature_layouts, **position_layouts}.items():
        if "position" not in source_layout.fields:
            continue
        missing = [name for name in layout.arm_joints if name not in source_layout.joint_names]
        if missing:
            continue
        return source_key, source_layout
    raise ValueError(
        f"{key} has no joint positions, and no other observation.state/action feature provides them. "
        "Pose, twist, and wrench conversion need joint positions for forward kinematics."
    )


def _order_arm_dataset_joints(
    joint_map: Mapping[str, str], chain_urdf_names: Sequence[str], dataset_joint_names: Sequence[str]
) -> list[str]:
    # Reject names that are not dataset joints or that collide on one URDF joint.
    unknown = [name for name in joint_map if name not in dataset_joint_names]
    if unknown:
        raise ValueError(f"joint_map names are not dataset joints: {unknown}")
    if len(set(joint_map.values())) != len(joint_map):
        raise ValueError("joint_map assigns two dataset joints to one URDF joint.")

    # Reject URDF joints that are not on the chain to the target frame.
    chain_set = set(chain_urdf_names)
    extra = [urdf_name for urdf_name in joint_map.values() if urdf_name not in chain_set]
    if extra:
        raise ValueError(f"joint_map URDF joints are not on the chain to the target frame: {extra}")

    # Return dataset joint names in URDF chain order.
    inverse: dict[str, str] = {}
    for dataset_name, urdf_name in joint_map.items():
        inverse[urdf_name] = dataset_name
    missing = [name for name in chain_urdf_names if name not in inverse]
    if missing:
        raise ValueError(f"Joint map is missing URDF chain joints: {missing}")
    return [inverse[name] for name in chain_urdf_names]


def _read_parquet_columns(dataset: LeRobotDataset, columns: list[str]) -> dict[str, np.ndarray]:
    # Concatenate every parquet shard.
    data_dir = Path(dataset.root) / DATA_DIR
    files = sorted(data_dir.glob("*/*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {data_dir}")
    table = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)

    # Stack vector columns; leave scalar columns as 1-D arrays.
    loaded: dict[str, np.ndarray] = {}
    for column in columns:
        values = table[column].to_numpy()
        if len(values) == 0:
            loaded[column] = np.zeros((0,))
            continue
        first = values[0]
        if isinstance(first, (np.ndarray, list, tuple)):
            loaded[column] = np.stack([np.asarray(row, dtype=np.float64) for row in values])
        else:
            loaded[column] = np.asarray(values)
    return loaded


def _convert_joint_matrix(
    matrix: np.ndarray,
    layout: JointSpaceLayout,
    kinematics: RobotKinematics,
    units: str,
    config: JointToCartesianConversionConfig,
    base_frame: str,
    q_matrix: np.ndarray | None = None,
    q_layout: JointSpaceLayout | None = None,
) -> tuple[np.ndarray, list[str]]:
    convert_set = set(layout.resolve_convert_fields())
    names: list[str] | None = None
    vectors: list[np.ndarray] = []

    # Convert each packed joint row.
    for index, row in enumerate(tqdm(matrix, desc="Converting joint frames")):
        q_row = None if q_matrix is None else q_matrix[index]
        sample = convert_joint_frame(
            kinematics,
            _resolve_row_q_deg(row, layout, units, q_row, q_layout),
            _gather_velocity_rad(row, layout, units) if "velocity" in convert_set else None,
            _gather_optional_field(row, layout, "effort") if "effort" in convert_set else None,
            layout,
            gripper_by_field={
                field_name: _gather_joint_values(row, layout, field_name, layout.gripper_joints)
                for field_name in convert_set
            },
            kept_by_field={
                field_name: _gather_joint_values(row, layout, field_name, layout.joint_names)
                for field_name in layout.fields
                if field_name not in convert_set
            },
            jacobian_frame_reference=config.jacobian_frame_reference,
            subtract_gravity=config.subtract_gravity,
            base_frame=base_frame,
        )

        # Keep the first frame's channel names for every later frame.
        sample_names = sample.feature_names()
        if names is None:
            names = _list_converted_channel_names(layout)
            if sample_names != names:
                raise RuntimeError("Cartesian channel names do not match the field layout.")
        elif sample_names != names:
            raise RuntimeError("Cartesian channel names changed between frames.")
        vectors.append(sample.to_vector())
    if names is None:
        raise ValueError("No frames to convert.")
    return np.stack(vectors).astype(np.float32), names


def _resolve_row_q_deg(
    row: np.ndarray,
    layout: JointSpaceLayout,
    units: str,
    q_row: np.ndarray | None,
    q_layout: JointSpaceLayout | None,
) -> np.ndarray | None:
    # Skip features that are not converting Cartesian quantities.
    if not layout.has_fields_to_convert():
        return None

    # Use this row's position field when it exists.
    if "position" in layout.fields:
        return _gather_position_degs(row, layout, units)

    # Use a sibling row's arm positions.
    if q_row is None or q_layout is None:
        raise ValueError(
            "Converting pose, twist, or wrench needs joint positions on this feature "
            "or on another loaded observation.state/action feature."
        )
    missing = [name for name in layout.arm_joints if (name, "position") not in q_layout.index]
    if missing:
        raise ValueError(f"Position source is missing arm joints {missing}.")
    values = np.array(
        [q_row[q_layout.index[(name, "position")]] for name in layout.arm_joints], dtype=np.float64
    )
    if units == "deg":
        return values
    return np.rad2deg(values)


def _gather_position_degs(row: np.ndarray, layout: JointSpaceLayout, units: str) -> np.ndarray:
    # Convert gathered arm positions to degrees.
    values = _gather_joint_values(row, layout, "position", layout.arm_joints)
    if units == "deg":
        return values
    return np.rad2deg(values)


def _gather_velocity_rad(row: np.ndarray, layout: JointSpaceLayout, units: str) -> np.ndarray | None:
    # Convert gathered arm velocity to rad/s.
    if "velocity" not in layout.fields:
        return None
    values = _gather_joint_values(row, layout, "velocity", layout.arm_joints)
    if units == "deg":
        return np.deg2rad(values)
    return values


def _gather_optional_field(row: np.ndarray, layout: JointSpaceLayout, field_name: str) -> np.ndarray | None:
    # Return the arm values for a field that may be absent.
    if field_name not in layout.fields:
        return None
    return _gather_joint_values(row, layout, field_name, layout.arm_joints)


def _gather_joint_values(
    row: np.ndarray, layout: JointSpaceLayout, field_name: str, joints: Sequence[str]
) -> np.ndarray:
    # Pull named joints for one field from a packed row.
    if not joints:
        return np.zeros(0, dtype=np.float64)
    return np.array([row[layout.index[(name, field_name)]] for name in joints], dtype=np.float64)


def convert_joint_frame(
    kinematics: RobotKinematics,
    q_deg: np.ndarray | None,
    qd: np.ndarray | None,
    tau: np.ndarray | None,
    layout: JointSpaceLayout,
    *,
    gripper_by_field: dict[str, np.ndarray],
    kept_by_field: dict[str, np.ndarray] | None = None,
    jacobian_frame_reference: str = "local_world_aligned",
    subtract_gravity: bool = True,
    base_frame: str | None = None,
) -> CartesianFrameSample:
    """Map one joint sample to pose, twist, wrench, and copied joint channels.

    Args:
        kinematics (RobotKinematics): Chain kinematics.
        q_deg (np.ndarray | None): Arm joint positions in degrees.
        qd (np.ndarray | None): Arm joint velocities in rad/s.
        tau (np.ndarray | None): Arm joint torques in N·m.
        layout (JointSpaceLayout): Source channel layout.

    Returns:
        CartesianFrameSample: Cartesian sample.
    """
    convert_set = set(layout.resolve_convert_fields())
    needs_fk = "position" in convert_set
    needs_jac = "velocity" in convert_set or "effort" in convert_set
    jacobian = None
    transform = None
    q_values: np.ndarray | None = None

    # Compute FK and the Jacobian for the requested fields.
    if needs_fk or needs_jac:
        if q_deg is None:
            raise ValueError("Joint positions are required to convert pose, twist, or wrench.")
        q_values = np.asarray(q_deg, dtype=np.float64)
    if needs_jac:
        if q_values is None:
            raise ValueError("Joint positions are required to convert pose, twist, or wrench.")
        jacobian = kinematics.frame_jacobian(q_values, jacobian_frame_reference)
    external_tau = None if tau is None else np.asarray(tau, dtype=np.float64)

    # Subtract gravity from measured torques.
    if "effort" in convert_set and external_tau is not None and subtract_gravity:
        if base_frame is None:
            raise ValueError("base_frame is required when subtract_gravity is true.")
        external_tau = external_tau - _compute_gravity_torques(kinematics, base_frame)
    if needs_fk:
        if q_values is None:
            raise ValueError("Joint positions are required to convert pose, twist, or wrench.")
        transform = kinematics.forward_kinematics(q_values)
    kept = {} if kept_by_field is None else kept_by_field
    quantities: list[FramePoseRot6D | FrameTwist | FrameWrench | GripperChannels] = []

    # Append pose, twist, wrench, or copied joint channels.
    for field_name in layout.fields:
        if field_name not in convert_set:
            if field_name not in kept:
                raise ValueError(f"Unconverted field {field_name!r} has no joint values to copy.")
            quantities.append(
                GripperChannels(
                    names=tuple(layout.joint_names),
                    values=np.asarray(kept[field_name], dtype=np.float64),
                    entity_type=field_name,
                )
            )
            continue
        if field_name == "position":
            if transform is None:
                raise ValueError("Forward kinematics is required to convert position.")
            quantities.append(
                FramePoseRot6D(translation=transform[:3, 3].copy(), rotation=transform[:3, :3].copy())
            )
        elif field_name == "velocity":
            if qd is None:
                raise ValueError("Velocity field is present but no joint velocity was provided.")
            if jacobian is None:
                raise ValueError("Jacobian is required to convert velocity.")
            twist = jacobian @ np.asarray(qd, dtype=np.float64)
            quantities.append(FrameTwist(linear=twist[:3], angular=twist[3:]))
        elif field_name == "effort":
            if external_tau is None:
                raise ValueError("Effort field is present but no joint torque was provided.")
            if jacobian is None:
                raise ValueError("Jacobian is required to convert effort.")
            # Min-norm wrench. A 6-row Jacobian with fewer joints does not determine a unique wrench.
            solved, _, _, _ = np.linalg.lstsq(jacobian.T, external_tau, rcond=None)
            quantities.append(FrameWrench(force=solved[:3], torque=solved[3:]))
        else:
            raise ValueError(f"Unsupported joint field {field_name!r}.")
        passed = gripper_by_field.get(field_name, np.zeros(0))
        if passed.size:
            quantities.append(
                GripperChannels(
                    names=tuple(layout.gripper_joints),
                    values=np.asarray(passed, dtype=np.float64),
                    entity_type=field_name,
                )
            )
    return CartesianFrameSample(quantities=tuple(quantities), fields=tuple(layout.fields))


def _compute_gravity_torques(kinematics: RobotKinematics, base_frame: str) -> np.ndarray:
    # Torques at the chain parent match the joint gravity term.
    torques = kinematics.robot.static_gravity_compensation_torques_dict(base_frame)
    values = np.array([torques[name] for name in kinematics.joint_names], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Gravity compensation at frame {base_frame!r} returned non-finite torques.")
    return values


def _convert_matrix_to_radians(matrix: np.ndarray, layout: JointSpaceLayout, units: str) -> np.ndarray:
    # Convert position and velocity columns from degrees to radians.
    if units == "rad":
        return matrix
    converted = np.array(matrix, dtype=np.float64, copy=True)
    for field_name in ("position", "velocity"):
        if field_name not in layout.fields:
            continue
        for joint in layout.joint_names:
            column = layout.index[(joint, field_name)]
            converted[:, column] = np.deg2rad(converted[:, column])
    return converted


def _write_converted_dataset(
    dataset: LeRobotDataset,
    config: JointToCartesianConversionConfig,
    output_dir: str | Path,
    repo_id: str,
    original: Mapping[str, np.ndarray],
    converted: Mapping[str, np.ndarray],
    names: Mapping[str, list[str]],
) -> LeRobotDataset:
    # Replace converted features with Cartesian vectors.
    replace_features: dict[str, tuple[np.ndarray, dict]] = {}
    for key, vectors in converted.items():
        feature = dataset.meta.features[key]
        replace_features[key] = (vectors, _build_vector_feature(names[key], feature["dtype"]))

    # Keep the original joint vectors under observation.joint_state / action.joints.
    add_features: dict[str, tuple[np.ndarray, dict]] = {}
    if config.keep_joint_features:
        for key in converted:
            dest = _JOINT_FEATURE_KEYS[key]
            if dest in dataset.meta.features:
                raise ValueError(f"{dest} already exists; cannot keep the original joint features.")
            feature = dataset.meta.features[key]
            add_features[dest] = (_cast_to_feature_dtype(original[key], feature), dict(feature))
    return modify_features(
        dataset,
        add_features=add_features or None,
        replace_features=replace_features,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def _build_vector_feature(names: list[str], dtype_name: str) -> dict[str, Any]:
    # Build a 1-D feature info dict.
    return {"dtype": dtype_name, "shape": (len(names),), "names": list(names)}


def _cast_to_feature_dtype(values: np.ndarray, feature: Mapping[str, Any]) -> np.ndarray:
    # Cast values to the declared feature dtype.
    return np.asarray(values, dtype=np.dtype(feature["dtype"]))


def evaluate_conversion(
    kinematics: RobotKinematics,
    joint_space_features: dict[str, np.ndarray],
    cartesian_features: dict[str, np.ndarray],
    layout: JointSpaceLayout,
    stride: int,
    *,
    episode_indices: np.ndarray | None = None,
    jacobian_frame_reference: str = "local_world_aligned",
    feature_layouts: dict[str, JointSpaceLayout] | None = None,
) -> JointToCartesianConversionReport:
    """Compare converted frames with an IK round-trip and trajectory checks.

    Args:
        kinematics (RobotKinematics): Chain kinematics.
        joint_space_features (dict[str, np.ndarray]): Original joint rows, radians.
        cartesian_features (dict[str, np.ndarray]): Converted rows.
        layout (JointSpaceLayout): Layout used for trajectory checks.
        stride (int): IK sample period.

    Returns:
        JointToCartesianConversionReport: Evaluation summary.
    """
    per_feature = (
        feature_layouts if feature_layouts is not None else dict.fromkeys(joint_space_features, layout)
    )

    # Evaluate IK round-trip and trajectory on converted pose features.
    pose_keys = [
        key
        for key, feature_layout in per_feature.items()
        if key in cartesian_features
        and key in joint_space_features
        and "position" in feature_layout.resolve_convert_fields()
        and "position" in feature_layout.fields
    ]
    if pose_keys:
        errors, position_errors_mm = _compute_ik_errors(
            kinematics,
            {key: joint_space_features[key] for key in pose_keys},
            {key: cartesian_features[key] for key in pose_keys},
            {key: per_feature[key] for key in pose_keys},
            stride,
        )
        error_array = np.stack(errors)
        rmse = np.sqrt(np.mean(np.square(error_array), axis=0))
        ik_round_trip = {
            "stride": stride,
            "per_joint_rmse_rad": {
                name: float(value)
                for name, value in zip(per_feature[pose_keys[0]].arm_joints, rmse, strict=True)
            },
            "max_abs_error_rad": float(np.max(np.abs(error_array))),
            "fraction_above_1deg": float(
                np.mean(np.any(np.abs(error_array) > _MAX_IK_JOINT_ERROR_RAD, axis=1))
            ),
            "fk_recheck_position_rmse_mm": float(np.sqrt(np.mean(np.square(position_errors_mm)))),
        }
        key = OBS_STATE if OBS_STATE in pose_keys else pose_keys[0]
        feature_layout = per_feature[key]
        if episode_indices is None:
            episode_indices = np.zeros(len(cartesian_features[key]), dtype=np.int64)
        trajectory = _compute_trajectory_report(
            kinematics,
            joint_space_features[key],
            cartesian_features[key],
            feature_layout,
            episode_indices,
            jacobian_frame_reference,
        )
    else:
        ik_round_trip = {}
        trajectory = {}

    # Evaluate wrench zero-check on converted effort features.
    effort_keys = [
        key
        for key, feature_layout in per_feature.items()
        if key in cartesian_features
        and key in joint_space_features
        and "effort" in feature_layout.resolve_convert_fields()
        and "effort" in feature_layout.fields
        and "position" in feature_layout.fields
    ]
    if effort_keys:
        key = OBS_STATE if OBS_STATE in effort_keys else effort_keys[0]
        if episode_indices is None:
            episode_indices = np.zeros(len(cartesian_features[key]), dtype=np.int64)
        wrench_zero_check = _compute_wrench_zero_check(
            joint_space_features[key], cartesian_features[key], per_feature[key], episode_indices
        )
    else:
        wrench_zero_check = {}
    return JointToCartesianConversionReport(
        ik_round_trip=ik_round_trip, trajectory=trajectory, wrench_zero_check=wrench_zero_check
    )


def _compute_ik_errors(
    kinematics: RobotKinematics,
    joint_space_features: dict[str, np.ndarray],
    cartesian_features: dict[str, np.ndarray],
    feature_layouts: dict[str, JointSpaceLayout],
    stride: int,
) -> tuple[list[np.ndarray], list[float]]:
    errors: list[np.ndarray] = []
    position_errors_mm: list[float] = []

    # Solve IK from each converted pose and compare to the recorded joints.
    for key, joints in joint_space_features.items():
        feature_layout = feature_layouts[key]
        names = _list_converted_channel_names(feature_layout)
        pose_slice = _slice_ee_channels(names, ".position", 9)
        arm_columns = [feature_layout.index[(name, "position")] for name in feature_layout.arm_joints]
        recorded = joints[:, arm_columns]
        cartesian = cartesian_features[key]
        for frame in range(0, len(recorded), stride):
            seed = recorded[frame - 1] if frame > 0 else recorded[frame]
            target = _build_pose_matrix(cartesian[frame, pose_slice])
            solved_deg = kinematics.inverse_kinematics(
                np.rad2deg(seed),
                target,
                position_weight=_IK_POSITION_WEIGHT,
                orientation_weight=_IK_ORIENTATION_WEIGHT,
                max_iters=_IK_MAX_ITERS,
            )
            solved_rad = np.deg2rad(np.asarray(solved_deg[: len(arm_columns)], dtype=np.float64))
            errors.append(solved_rad - recorded[frame])
            achieved = kinematics.forward_kinematics(solved_deg)
            position_errors_mm.append(float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]) * 1000.0))
    if not errors:
        raise ValueError("IK round-trip had no frames to evaluate.")
    return errors, position_errors_mm


def _list_converted_channel_names(layout: JointSpaceLayout) -> list[str]:
    # Match convert_joint_frame block order: converted quantity then gripper, or original joints.
    names: list[str] = []
    convert_set = set(layout.resolve_convert_fields())
    for field_name in layout.fields:
        if field_name in convert_set:
            names.extend(_list_cartesian_field_names(field_name))
            names.extend(f"{joint}.{field_name}" for joint in layout.gripper_joints)
        else:
            names.extend(f"{joint}.{field_name}" for joint in layout.joint_names)
    return names


def _list_cartesian_field_names(field_name: str) -> list[str]:
    # Return ee_ channel names for one converted field.
    if field_name == "position":
        return FramePoseRot6D.feature_names()
    if field_name == "velocity":
        return FrameTwist.feature_names()
    if field_name == "effort":
        return FrameWrench.feature_names()
    raise ValueError(f"Unsupported joint field {field_name!r}.")


def _slice_ee_channels(names: list[str], suffix: str, count: int) -> slice:
    # Slice consecutive ee_ channels that share a suffix.
    indices = [index for index, name in enumerate(names) if name.startswith("ee_") and name.endswith(suffix)]
    if len(indices) < count:
        raise ValueError(f"Expected at least {count} '{suffix}' channels, found {len(indices)}.")
    return slice(indices[0], indices[0] + count)


def _build_pose_matrix(values: np.ndarray) -> np.ndarray:
    # Build a 4x4 transform from a 9-D pose vector.
    pose = FramePoseRot6D.from_vector(values[:9])
    transform = np.eye(4)
    transform[:3, :3] = pose.rotation
    transform[:3, 3] = pose.translation
    return transform


def _compute_trajectory_report(
    kinematics: RobotKinematics,
    joints: np.ndarray,
    cartesian: np.ndarray,
    layout: JointSpaceLayout,
    episode_indices: np.ndarray,
    jacobian_frame_reference: str,
) -> dict[str, Any]:
    names = _list_converted_channel_names(layout)
    pose = cartesian[:, _slice_ee_channels(names, ".position", 9)]
    translations = pose[:, :3]
    rotations = _extract_rotations_from_pose(pose)

    # Measure the largest translation and rotation jump in each episode.
    max_translation = 0.0
    max_rotation = 0.0
    for episode in np.unique(episode_indices):
        indices = np.flatnonzero(episode_indices == episode)
        if len(indices) < 2:
            continue
        jumps = np.linalg.norm(np.diff(translations[indices], axis=0), axis=1)
        max_translation = max(max_translation, float(jumps.max()))
        max_rotation = max(max_rotation, _compute_max_rotation_jump_deg(rotations[indices]))

    # Count near-singular frames and frames outside joint limits.
    arm_columns = [layout.index[(name, "position")] for name in layout.arm_joints]
    recorded = joints[:, arm_columns]
    near_singular = 0
    outside_limits = 0
    limits = [kinematics.joint_limits(name) for name in kinematics.joint_names]
    for position in recorded:
        jacobian = kinematics.frame_jacobian(np.rad2deg(position), jacobian_frame_reference)
        if float(np.linalg.cond(jacobian)) > _SINGULAR_JACOBIAN_COND_THRESHOLD:
            near_singular += 1
        if _exceeds_joint_limits(position, limits):
            outside_limits += 1
    return {
        "max_translation_jump_m": max_translation,
        "max_rotation_jump_deg": max_rotation,
        "workspace_min_m": [float(value) for value in translations.min(axis=0)],
        "workspace_max_m": [float(value) for value in translations.max(axis=0)],
        "near_singular_frames": near_singular,
        "frames_outside_joint_limits": outside_limits,
    }


def _extract_rotations_from_pose(pose: np.ndarray) -> np.ndarray:
    # Reconstruct rotation matrices from the 6-D rotation columns.
    column_0 = pose[:, 3:6]
    column_1 = pose[:, 6:9]
    column_0 = column_0 / np.linalg.norm(column_0, axis=1, keepdims=True)
    column_1 = column_1 - np.sum(column_0 * column_1, axis=1, keepdims=True) * column_0
    column_1 = column_1 / np.linalg.norm(column_1, axis=1, keepdims=True)
    column_2 = np.cross(column_0, column_1)
    return np.stack([column_0, column_1, column_2], axis=-1)


def _compute_max_rotation_jump_deg(rotations: np.ndarray) -> float:
    # Return the largest geodesic rotation between consecutive frames.
    relative = np.einsum("nji,njk->nik", rotations[:-1], rotations[1:])
    traces = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
    cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosines)).max())


def _exceeds_joint_limits(position: np.ndarray, limits: list[np.ndarray]) -> bool:
    # Return True if any joint is outside its limit.
    for value, bound in zip(position, limits, strict=True):
        lower = float(bound[0])
        upper = float(bound[1])
        if (np.isfinite(lower) and value < lower - 1e-4) or (np.isfinite(upper) and value > upper + 1e-4):
            return True
    return False


def _compute_wrench_zero_check(
    joints: np.ndarray,
    cartesian: np.ndarray,
    layout: JointSpaceLayout,
    episode_indices: np.ndarray,
) -> dict[str, Any]:
    empty = {"frames": 0, "mean_force_n": 0.0, "mean_torque_nm": 0.0}
    if "effort" not in layout.fields:
        return empty

    # Average wrench on still frames with an open gripper.
    names = _list_converted_channel_names(layout)
    wrench = cartesian[:, _slice_ee_channels(names, ".effort", 6)]
    arm_columns = [layout.index[(name, "position")] for name in layout.arm_joints]
    recorded = joints[:, arm_columns]
    still = _mask_still_frames(joints, recorded, layout, episode_indices)
    gripper_open = _mask_open_gripper(joints, layout)
    mask = still & gripper_open
    count = int(np.count_nonzero(mask))
    if count == 0:
        return empty
    return {
        "frames": count,
        "mean_force_n": float(np.mean(np.linalg.norm(wrench[mask, :3], axis=1))),
        "mean_torque_nm": float(np.mean(np.linalg.norm(wrench[mask, 3:], axis=1))),
    }


def _mask_still_frames(
    joints: np.ndarray,
    recorded: np.ndarray,
    layout: JointSpaceLayout,
    episode_indices: np.ndarray,
) -> np.ndarray:
    # Use recorded velocity when present.
    if "velocity" in layout.fields:
        columns = [layout.index[(name, "velocity")] for name in layout.arm_joints]
        return np.max(np.abs(joints[:, columns]), axis=1) <= _MAX_STILL_JOINT_DELTA_RAD

    # Fall back to per-episode joint deltas.
    still = np.zeros(len(recorded), dtype=bool)
    for episode in np.unique(episode_indices):
        indices = np.flatnonzero(episode_indices == episode)
        if len(indices) < 2:
            continue
        delta = np.max(np.abs(np.diff(recorded[indices], axis=0)), axis=1)
        still[indices[1:]] = delta <= _MAX_STILL_JOINT_DELTA_RAD
    return still


def _mask_open_gripper(joints: np.ndarray, layout: JointSpaceLayout) -> np.ndarray:
    # Treat values above the gripper midpoint as open.
    if "position" not in layout.fields or not layout.gripper_joints:
        return np.ones(len(joints), dtype=bool)
    column = layout.index[(layout.gripper_joints[0], "position")]
    gripper = joints[:, column]
    if float(gripper.max() - gripper.min()) < 1e-6:
        return np.ones(len(gripper), dtype=bool)
    midpoint = float(gripper.min() + gripper.max()) / 2.0
    return gripper >= midpoint


def _log_report(report: JointToCartesianConversionReport) -> None:
    ik = report.ik_round_trip

    if ik:
        logging.info("IK round-trip (stride=%s)", ik["stride"])
        logging.info("  %-24s %s", "joint", "rmse_rad")
        for name, value in ik["per_joint_rmse_rad"].items():
            logging.info("  %-24s %.6g", name, value)
        logging.info("  %-24s %.6g", "max_abs_error_rad", ik["max_abs_error_rad"])
        logging.info("  %-24s %.6g", "fraction_above_1deg", ik["fraction_above_1deg"])
        logging.info("  %-24s %.6g", "fk_recheck_position_rmse_mm", ik["fk_recheck_position_rmse_mm"])

    if report.trajectory:
        logging.info("Trajectory")
        for name, value in report.trajectory.items():
            logging.info("  %-24s %s", name, value)

    if report.wrench_zero_check:
        logging.info("Wrench zero-check")
        for name, value in report.wrench_zero_check.items():
            logging.info("  %-24s %s", name, value)


def _rewrite_episode_stats(
    dataset: LeRobotDataset,
    feature_names: list[str],
    matrices: dict[str, np.ndarray],
    episode_index: np.ndarray,
) -> None:
    # Compute per-episode stats for the converted features.
    features = {name: dataset.meta.features[name] for name in feature_names}
    per_episode: dict[int, dict[str, Any]] = {}
    for episode in np.unique(episode_index):
        mask = episode_index == episode
        episode_data = {name: matrices[name][mask] for name in feature_names}
        stats = compute_episode_stats(episode_data, features)
        listed = {
            name: {stat: np.asarray(value).tolist() for stat, value in values.items()}
            for name, values in stats.items()
        }
        per_episode[int(episode)] = flatten_dict({"stats": listed})

    # Load existing episode metadata.
    episodes_dir = Path(dataset.root) / "meta" / "episodes"
    files = sorted(episodes_dir.glob("*/*.parquet"))
    if not files:
        return
    table = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    # Replace stats columns for the converted features.
    drop = [
        column
        for column in table.columns
        if any(column.startswith(f"stats/{name}/") for name in feature_names)
    ]
    table = table.drop(columns=drop)
    for key in sorted({key for stats in per_episode.values() for key in stats}):
        table[key] = [per_episode[int(episode)][key] for episode in table["episode_index"]]

    # Write episode metadata into a single shard.
    if "meta/episodes/chunk_index" in table.columns:
        table["meta/episodes/chunk_index"] = 0
        table["meta/episodes/file_index"] = 0
    episode_dataset = datasets.Dataset.from_pandas(table, preserve_index=False)
    destination = Path(dataset.root) / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    write_episodes(episode_dataset, Path(dataset.root))
    for path in files:
        if path.resolve() != destination.resolve() and path.exists():
            path.unlink()


def _write_provenance(
    dataset: LeRobotDataset,
    config: JointToCartesianConversionConfig,
    target_frame: str,
    joint_map: dict[str, str],
    units: str,
    channel_names: dict[str, list[str]],
    report: JointToCartesianConversionReport | None,
) -> None:
    gripper_channels = []
    for names in channel_names.values():
        for name in names:
            if not name.startswith("ee_") and name not in gripper_channels:
                gripper_channels.append(name)
    payload = {
        "urdf": {"name": Path(config.urdf).name, "sha256": _sha256(config.urdf)},
        "target_frame": target_frame,
        "joint_map": joint_map,
        "gripper_channels": gripper_channels,
        "joint_units": units,
        "orientation": "rotation_6d_columns",
        "jacobian_frame_reference": config.jacobian_frame_reference,
        "subtract_gravity": config.subtract_gravity,
        "feature_keys": list(channel_names),
        "convert_fields": config.convert_fields,
        "channel_names": channel_names,
        "evaluation": None if report is None else report.to_dict(),
    }
    path = Path(dataset.root) / "meta" / "cartesian_conversion.json"
    path.write_text(json.dumps(_to_json_format(payload), indent=2) + "\n")


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _to_json_format(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_format(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_format(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _plot_episodes(dataset: LeRobotDataset, plot_episodes: int) -> None:
    # Skip plotting if the number of episodes to plot is zero.
    if plot_episodes <= 0:
        return

    from lerobot.scripts.lerobot_dataset_plot import (
        plot_episode,
        resolve_feature_keys,
        resolve_image_observation_keys,
        resolve_obs_plot_groups,
    )

    try:
        image_keys = resolve_image_observation_keys(dataset, None)
    except ValueError as exc:
        logging.warning("Skipping conversion plots: %s", exc)
        return

    # Get the number of episodes to plot.
    count = min(plot_episodes, dataset.meta.total_episodes)

    # Plot the episodes.
    feature_keys = resolve_feature_keys(dataset, None)
    groups = resolve_obs_plot_groups(["position", "velocity", "effort"])
    output_dir = str(Path(dataset.root) / "conversion_plots")
    for episode_index in range(count):
        plot_episode(
            dataset,
            episode_index,
            image_keys,
            feature_keys,
            groups,
            18,
            output_dir,
            130,
            1e-4,
        )
