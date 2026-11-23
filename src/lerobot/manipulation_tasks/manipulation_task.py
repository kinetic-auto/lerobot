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

"""Manipulation task that loads the policy of a requested task on demand."""

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, bind_relative_anchor

from .configuration_manipulation_task import KineticManipulationTaskConfig, override_task_config


class KineticManipulationTask:
    """Manipulation task that loads the policy of a requested task."""

    def __init__(self, task_config_directory: str | Path | None = None, device: str | None = None) -> None:
        """Create a task with no policy loaded.

        Args:
            task_config_directory (str | Path | None, optional): Directory of the task YAML files.
                Defaults to the packaged configs.
            device (str | None, optional): Policy device. Defaults to cuda when available, else cpu.
        """
        self._task_config_directory = task_config_directory
        self._device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._task_config: KineticManipulationTaskConfig | None = None
        self._policy: PreTrainedPolicy | None = None
        self._preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self._postprocessor: PolicyProcessorPipeline[Any, Any] | None = None

    def prepare_task(self, task_name: str, task_config_overrides: dict[str, Any] | None = None) -> None:
        """Load the policy and processors of a task.

        Args:
            task_name (str): Task name.
            task_config_overrides (dict[str, Any] | None, optional): Task config field name to value.
                Defaults to None.
        """
        task_config = KineticManipulationTaskConfig.load_from_yaml(task_name, self._task_config_directory)
        override_task_config(task_config, task_config_overrides or {})

        if task_config == self._task_config:
            self.reset()
            return

        policy, preprocessor, postprocessor = self._load_policy_and_processors(
            task_config.policy_checkpoint_path
        )
        self.unload_task()
        self._task_config = task_config
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.reset()

    def _load_policy_and_processors(
        self, policy_checkpoint_path: str
    ) -> tuple[
        PreTrainedPolicy,
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        PolicyProcessorPipeline[Any, Any],
    ]:
        policy_config = PreTrainedConfig.from_pretrained(policy_checkpoint_path)
        policy_config.device = self._device
        policy = get_policy_class(policy_config.type).from_pretrained(
            policy_checkpoint_path, config=policy_config
        )
        policy.to(self._device).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=policy_checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": self._device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        bind_relative_anchor(policy, preprocessor)
        return policy, preprocessor, postprocessor

    def unload_task(self) -> None:
        """Release the policy and processors of the loaded task."""
        self._task_config = None
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self) -> None:
        """Reset the policy and its processors."""
        policy, preprocessor, postprocessor = self._get_loaded_policy_and_processors("reset")
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    def predict_action_chunk(self, observation: dict[str, np.ndarray]) -> Tensor:
        """Predict an action chunk.

        Args:
            observation (dict[str, np.ndarray]): Observation of the current step.

        Returns:
            Tensor: Absolute action chunk, (chunk_size, action_dim).
        """
        policy, preprocessor, postprocessor = self._get_loaded_policy_and_processors("predict_action_chunk")
        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                dict(observation), torch.device(self._device), task=self.task_description
            )
            batch = preprocessor(batch)
            if hasattr(policy, "predict_action_chunk_with_auxiliary_outputs"):
                action_chunk, _ = policy.predict_action_chunk_with_auxiliary_outputs(batch)
            else:
                action_chunk = policy.predict_action_chunk(batch)
            return postprocessor.process_action(action_chunk)[0]

    def _get_loaded_policy_and_processors(
        self, method_name: str
    ) -> tuple[
        PreTrainedPolicy,
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        PolicyProcessorPipeline[Any, Any],
    ]:
        if self._policy is None or self._preprocessor is None or self._postprocessor is None:
            raise RuntimeError(f"Call prepare_task(task_name) before {method_name}().")
        return self._policy, self._preprocessor, self._postprocessor

    @property
    def active_task_name(self) -> str | None:
        """Name of the loaded task."""
        return None if self._task_config is None else self._task_config.task_name

    @property
    def task_description(self) -> str:
        """Task text given to the policy."""
        return "" if self._task_config is None else self._task_config.task_description

    @property
    def policy_type(self) -> str:
        """Policy type of the loaded task."""
        return self._get_loaded_policy_and_processors("policy_type")[0].config.type

    @property
    def camera_image_shapes(self) -> dict[str, tuple[int, ...]]:
        """Camera key to image shape, (C, H, W)."""
        image_features = self._get_loaded_policy_and_processors("camera_image_shapes")[
            0
        ].config.image_features
        return {camera_key: tuple(feature.shape) for camera_key, feature in image_features.items()}

    @property
    def state_dimension(self) -> int | None:
        """Width of the state."""
        robot_state_feature = self._get_loaded_policy_and_processors("state_dimension")[
            0
        ].config.robot_state_feature
        return None if robot_state_feature is None else robot_state_feature.shape[0]

    @property
    def chunk_size(self) -> int | None:
        """Steps per action chunk."""
        return getattr(self._get_loaded_policy_and_processors("chunk_size")[0].config, "chunk_size", None)

    @property
    def state_history_length(self) -> int:
        """State steps per history, current step included."""
        return getattr(
            self._get_loaded_policy_and_processors("state_history_length")[0].config,
            "state_history_length",
            1,
        )

    @property
    def image_history_length(self) -> int:
        """Image steps per history, current step included."""
        return getattr(
            self._get_loaded_policy_and_processors("image_history_length")[0].config,
            "image_history_length",
            1,
        )

    @property
    def observation_history_stride(self) -> int:
        """Frames between consecutive history steps."""
        policy_config = self._get_loaded_policy_and_processors("observation_history_stride")[0].config
        return getattr(policy_config, "observation_history_stride", 1)
