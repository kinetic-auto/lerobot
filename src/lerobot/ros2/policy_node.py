#!/usr/bin/env python3
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

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_srvs.srv import Trigger

from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.ros2.observation import (
    build_action,
    build_observation_state,
    load_checkpoint_metadata,
    map_joint_names,
    policy_state_action_layout_from_checkpoint,
    resize_image_with_pad,
    resolve_checkpoint_dir,
)
from lerobot.utils.constants import OBS_IMAGES

DEFAULT_DEVICE = "cuda"
DEFAULT_CONTROL_FREQUENCY = 30.0
DEFAULT_OBSERVATION_TIMEOUT_SECONDS = 1.0
DEFAULT_ROBOT_JOINT_PREFIX = "follower_r_"
DEFAULT_MODEL_JOINT_PREFIX = "right_"
DEFAULT_CAMERA_NAMES = ["chest", "wrist_r"]
DEFAULT_CAMERA_TOPICS = [
    "cam_chest/image_raw/compressed",
    "cam_wrist_r/image_raw/compressed",
]


@dataclass
class LoadedPolicy:
    """Policy, processors, and the layout needed for one tick.
    """
    policy: Any
    preprocessor: Any
    postprocessor: Any
    layout: Any
    image_sizes: Dict[str, Tuple[int, int]]
    task: str


class LerobotPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("lerobot_policy_node")

        # Declare parameters
        checkpoint = self._declare_required_text_parameter("checkpoint")
        self._device = self._declare_text_parameter("device", DEFAULT_DEVICE)
        control_frequency = self._declare_float_parameter(
            "control_frequency", DEFAULT_CONTROL_FREQUENCY
        )
        self._observation_timeout_seconds = self._declare_float_parameter(
            "observation_timeout_seconds", DEFAULT_OBSERVATION_TIMEOUT_SECONDS
        )
        self._robot_prefix = self._declare_text_parameter(
            "robot_joint_prefix", DEFAULT_ROBOT_JOINT_PREFIX
        )
        self._model_prefix = self._declare_text_parameter(
            "model_joint_prefix", DEFAULT_MODEL_JOINT_PREFIX
        )
        self._task_parameter = self._declare_text_parameter("task", "")
        action_step_count = int(self._declare_float_parameter("n_action_steps", 0.0))
        if control_frequency <= 0:
            raise ValueError(f"control_frequency must be > 0, got {control_frequency}")
        if self._observation_timeout_seconds <= 0:
            raise ValueError(
                f"observation_timeout_seconds must be > 0, got {self._observation_timeout_seconds}"
            )

        # Load the checkpoint and the policy state-action layout.
        ckpt_policy = self._load_policy(checkpoint, action_step_count, self._task_parameter)
        self._policy = ckpt_policy.policy
        self._preprocessor = ckpt_policy.preprocessor
        self._postprocessor = ckpt_policy.postprocessor
        self._layout = ckpt_policy.layout
        self._image_sizes = ckpt_policy.image_sizes
        self._task = ckpt_policy.task

        self._joint_state: Optional[JointState] = None
        self._images: Dict[str, Tuple[bytes, Any]] = {}
        self._decoded: Dict[str, Tuple[bytes, np.ndarray]] = {}

        # Subscribe to cameras and joint states.
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        camera_names = self._declare_string_list_parameter("camera_names", DEFAULT_CAMERA_NAMES)
        camera_topics = self._declare_string_list_parameter("camera_topics", DEFAULT_CAMERA_TOPICS)
        self._subscribe_cameras(camera_names, camera_topics, CompressedImage, qos_profile_sensor_data)

        self.create_subscription(
            JointState,
            self._declare_text_parameter("joint_state_topic", "joint_states"),
            self._handle_joint_state_callback,
            reliable,
        )

        # Publish policy actions.
        self._action_publisher = self.create_publisher(
            JointState,
            self._declare_text_parameter("policy_action_topic", "policy_action"),
            reliable,
        )
        self.create_service(Trigger, "~/reset", self._handle_reset_callback)
        self.create_timer(1.0 / control_frequency, self._handle_tick_callback)

        # Logging
        self.get_logger().info(f"joint names {list(self._layout.joint_names)}")
        self.get_logger().info(f"image keys {list(self._image_sizes)}")

        action_steps = getattr(self._policy.config, "n_action_steps", "unset")
        self.get_logger().info(f"n_action_steps {action_steps}")

    def _declare_required_text_parameter(self, name: str) -> str:
        """Declare a text parameter that must be set.
        """
        self.declare_parameter(name, "")
        value = str(self.get_parameter(name).value)
        if not value:
            raise ValueError(f"parameter {name} is required")
        return value

    def _declare_text_parameter(self, name: str, default: str) -> str:
        """Declare a text parameter and return its value.
        """
        self.declare_parameter(name, default)
        return str(self.get_parameter(name).value)

    def _declare_float_parameter(self, name: str, default: float) -> float:
        """Declare a float parameter and return its value.
        """
        self.declare_parameter(name, default)
        return float(self.get_parameter(name).value)

    def _declare_string_list_parameter(
        self, name: str, default: Optional[List[str]] = None
    ) -> List[str]:
        """Declare a string-list parameter and drop empty entries.
        """
        self.declare_parameter(name, default if default is not None else [""])
        return [str(item) for item in self.get_parameter(name).value if str(item)]

    def _load_policy(self, checkpoint: str, action_step_count: int, task: str) -> LoadedPolicy:
        """Load the checkpoint, processors, and policy state-action layout.
        """
        # Check if the device is available
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device is {self._device} but CUDA is not available")

        ckpt_dir = resolve_checkpoint_dir(checkpoint)
        metadata = load_checkpoint_metadata(ckpt_dir)
        config = PreTrainedConfig.from_pretrained(ckpt_dir)

        # Load the policy
        policy = get_policy_class(config.type).from_pretrained(str(ckpt_dir))
        policy.to(self._device)

        # Set the policy to evaluation mode
        policy.eval()

        # Load the preprocessor and postprocessor
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": self._device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        # Load the policy state-action layout
        policy_io_layout = policy_state_action_layout_from_checkpoint(ckpt_dir)

        # Check if the packed widths of the layout match the checkpoint
        self._validate_state_and_action_lengths(config, policy_io_layout)

        # Get the image sizes
        image_sizes = self._get_image_sizes(config)

        # Validate the action step count
        if action_step_count > 0:
            chunk_size = getattr(policy.config, "chunk_size", None)
            if chunk_size is not None and action_step_count > int(chunk_size):
                raise ValueError(
                    f"n_action_steps {action_step_count} is larger than chunk_size {chunk_size}"
                )
            policy.config.n_action_steps = int(action_step_count)

        # Warm up the policy
        self._warmup_policy(policy)

        # Get the task description
        task_description = task or str(metadata.get("task_description") or "")

        return LoadedPolicy(policy, preprocessor, postprocessor, policy_io_layout, image_sizes, task_description)

    def _validate_state_and_action_lengths(self, config: Any, layout: Any) -> None:
        """Reject a layout whose state or action length differs from the checkpoint.
        """
        state = config.input_features.get("observation.state")
        if state is not None:
            expected = int(state.shape[0])
            packed = len(layout.joint_names) * len(layout.state_types)
            if packed != expected:
                raise ValueError(
                    f"policy state-action layout packs observation.state to {packed} "
                    f"but the checkpoint expects {expected}"
                )

        action = config.output_features.get("action")
        if action is not None:
            expected = int(action.shape[0])
            packed = len(layout.joint_names) * len(layout.action_types)
            if packed != expected:
                raise ValueError(
                    f"policy state-action layout packs action to {packed} "
                    f"but the checkpoint expects {expected}"
                )

    def _get_image_sizes(self, config: Any) -> Dict[str, Tuple[int, int]]:
        """Map each checkpoint camera name to its height and width.
        """
        sizes = {}
        prefix = OBS_IMAGES + "."
        for key, feature in config.input_features.items():
            if feature.type != FeatureType.VISUAL or not str(key).startswith(prefix):
                continue
            shape = tuple(int(dimension) for dimension in feature.shape)
            if len(shape) == 3 and shape[0] in (1, 3):
                height, width = shape[1], shape[2]
            elif len(shape) == 3 and shape[2] in (1, 3):
                height, width = shape[0], shape[1]
            else:
                raise ValueError(f"unsupported image shape for {key}: {shape}")
            sizes[str(key)[len(prefix):]] = (height, width)
        if not sizes:
            raise ValueError("checkpoint has no observation.images features")
        return sizes

    def _warmup_policy(self, policy: Any) -> None:
        """warm up the policy before the first prediction.
        """
        policy.reset()

    def _subscribe_cameras(
        self,
        camera_names: List[str],
        camera_topics: List[str],
        message_type: type,
        quality_of_service: Any,
    ) -> None:
        """Subscribe to each camera the checkpoint requires.
        """
        if len(camera_names) != len(camera_topics):
            raise ValueError(
                f"camera_names ({len(camera_names)}) and camera_topics "
                f"({len(camera_topics)}) differ in length"
            )
        topics = dict(zip(camera_names, camera_topics, strict=True))
        for name in camera_names:
            if name not in self._image_sizes:
                self.get_logger().info(
                    f"camera {name} is not in the checkpoint input features; ignoring it"
                )
        missing = [name for name in self._image_sizes if name not in topics]
        if missing:
            raise ValueError(
                f"checkpoint cameras {missing} have no entry in camera_names/camera_topics"
            )
        for name in self._image_sizes:
            self.create_subscription(
                message_type,
                topics[name],
                lambda message, camera_name=name: self._handle_img_callback(camera_name, message),
                quality_of_service,
            )

    def _handle_img_callback(self, name: str, message: CompressedImage) -> None:
        """Store the latest compressed frame for one camera.
        """
        self._images[name] = (bytes(message.data), self.get_clock().now())

    def _handle_joint_state_callback(self, message: JointState) -> None:
        """Store the latest joint state.
        """
        self._joint_state = message

    def _handle_reset_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Clear the policy action queue.
        """
        self._policy.reset()
        response.success = True
        response.message = "policy reset"
        return response

    def _handle_tick_callback(self) -> None:
        """Predict one action and publish it.
        """
        # Wait for the cameras
        if any(name not in self._images for name in self._image_sizes):
            self.get_logger().info("waiting for cameras", throttle_duration_sec=5.0)
            return

        # Wait for the joint states
        if self._joint_state is None:
            self.get_logger().info("waiting for joint_states", throttle_duration_sec=5.0)
            return

        # Warn on stale cameras
        self._warn_on_stale_cameras()

        # Build the observation and predict the action
        try:
            observation = self._build_observation()
            action = self._predict_action(observation)
        except Exception as e:
            self.get_logger().error(
                f"failed to predict policy_action: {e}",
                throttle_duration_sec=5.0,
            )
            return

        # Publish the policy action
        self._publish_policy_action(action, self._joint_state.header.stamp)

    def _warn_on_stale_cameras(self) -> None:
        """Log when a stored camera frame is older than the timeout.
        """
        now = self.get_clock().now()
        for name, (_payload, received_at) in self._images.items():
            elapsed_duration = (now - received_at).nanoseconds / 1e9
            if elapsed_duration > self._observation_timeout_seconds:
                self.get_logger().warning(
                    f"camera {name} is {elapsed_duration:.2f}s old",
                    throttle_duration_sec=5.0,
                )

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Pack the latest images and joint values into a policy observation.
        """
        observation: Dict[str, np.ndarray] = {}

        # Build the image observations
        for name, (height, width) in self._image_sizes.items():
            payload, _received_at = self._images[name]
            observation[f"observation.images.{name}"] = resize_image_with_pad(
                self._decode_compressed_image(name, payload),
                height,
                width,
            )

        # Build the joint state values
        joint_state_values = {}
        for joint_state_type in self._layout.state_types:
            if joint_state_type not in ("position", "effort", "velocity"):
                raise ValueError(f"unsupported state block {joint_state_type!r}")
            joint_state_values[joint_state_type] = self._joint_block_in_model_order(joint_state_type)
        observation["observation.state"] = build_observation_state(
            joint_state_values,
            self._layout.state_types,
        )

        return observation

    def _decode_compressed_image(self, name: str, payload: bytes) -> np.ndarray:
        """Decode one stored JPEG into an RGB image.
        """
        # Check if the image is already decoded
        cached = self._decoded.get(name)
        if cached is not None and cached[0] is payload:
            return cached[1]

        # Decode the image
        encoded = np.frombuffer(payload, dtype=np.uint8).copy()
        image = cv.imdecode(encoded, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {name}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Store the decoded image
        self._decoded[name] = (payload, image)

        return image

    def _joint_block_in_model_order(self, joint_field: str) -> List[float]:
        """Read one JointState field and return it in checkpoint joint order.
        """
        # Read the joint field values
        joint_field_values = list(getattr(self._joint_state, joint_field))
        robot_joint_names = [str(name) for name in self._joint_state.name]

        # Build the values in checkpoint order
        values_in_checkpoint_order = []
        missing_joint_names = []
        for checkpoint_joint_name in self._layout.joint_names:
            robot_joint_name = map_joint_names(
                [checkpoint_joint_name],
                self._model_prefix,
                self._robot_prefix,
            )[0]
            if robot_joint_name not in robot_joint_names:
                missing_joint_names.append(robot_joint_name)
                continue
            index = robot_joint_names.index(robot_joint_name)
            if index >= len(joint_field_values):
                missing_joint_names.append(robot_joint_name)
                continue
            values_in_checkpoint_order.append(float(joint_field_values[index]))

        # Raise if a joint is missing
        if missing_joint_names:
            raise KeyError(f"joint_states is missing {joint_field} for {missing_joint_names}")

        return values_in_checkpoint_order

    def _publish_policy_action(self, action: Any, stamp: Any) -> None:
        """Publish the position and effort values as a JointState.
        """
        # Build the action blocks
        blocks = build_action(
            action.detach().cpu().numpy(),
            len(self._layout.joint_names),
            self._layout.action_types,
        )

        # Build the JointState message
        message = JointState()
        message.header.stamp = stamp
        message.name = map_joint_names(
            self._layout.joint_names,
            self._model_prefix,
            self._robot_prefix,
        )

        if "position" in blocks:
            message.position = [float(value) for value in blocks["position"]]
        if "effort" in blocks:
            message.effort = [float(value) for value in blocks["effort"]]
        if "velocity" in blocks:
            message.velocity = [float(value) for value in blocks["velocity"]]

        # Publish the policy action
        self._action_publisher.publish(message)

    def _predict_action(self, observation: Dict[str, np.ndarray]) -> Any:
        """Run one lerobot predict_action call.
        """
        return predict_action(
            observation,
            self._policy,
            torch.device(self._device),
            self._preprocessor,
            self._postprocessor,
            use_amp=False,
            task=self._task or None,
        )


def _requests_help(args: Optional[List[str]]) -> bool:
    """Return whether the process was started with a help flag.
    """
    command_arguments = sys.argv[1:] if args is None else args
    return any(argument in ("-h", "--help") for argument in command_arguments)


def _print_usage() -> None:
    """Print how to start the policy node.
    """
    print(
        "usage: lerobot-ros2-policy-node --help\n"
        "       lerobot-ros2-policy-node --ros-args -p checkpoint:=<directory> [-p device:=cuda]\n"
        "\n"
        "  --help, -h    Show this help and exit.\n"
        "  checkpoint    Required. Checkpoint directory, or its pretrained_model child.\n"
        "\n"
        "The kinetic-door-policy compose service supplies the checkpoint, device, and namespace."
    )


def main(args: Optional[List[str]] = None) -> None:
    if _requests_help(args):
        _print_usage()
        return

    rclpy.init(args=args)
    node = None
    try:
        node = LerobotPolicyNode()
        node.get_logger().info("Node started - Ctrl+C to exit")
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Shutting down...")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
