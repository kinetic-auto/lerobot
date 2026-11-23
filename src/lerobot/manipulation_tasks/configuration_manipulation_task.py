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

"""Configuration of a manipulation task."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus

logger = logging.getLogger(__name__)

PACKAGED_TASK_CONFIG_DIRECTORY = Path(__file__).parent / "configs"


@dataclass
class KineticManipulationTaskConfig:
    """Configuration of a manipulation task.

    Args:
        task_name: Task name.
        task_description: Task text given to the policy.
        policy_checkpoint_path: Directory of the policy checkpoint and its processors.
    """

    task_name: str
    task_description: str
    policy_checkpoint_path: str

    @classmethod
    def load_from_yaml(
        cls, task_name: str, task_config_directory: str | Path | None = None
    ) -> "KineticManipulationTaskConfig":
        """Load the configuration of a task from its YAML file.

        Args:
            task_name (str): Task name.
            task_config_directory (str | Path | None, optional): Directory of the task YAML files.
                Defaults to the packaged configs.

        Returns:
            KineticManipulationTaskConfig: Task configuration.
        """
        directory = (
            PACKAGED_TASK_CONFIG_DIRECTORY if task_config_directory is None else Path(task_config_directory)
        )
        task_config_path = directory / f"{task_name}.yaml"
        if not task_config_path.is_file():
            raise ValueError(f"No task config for {task_name!r} at {task_config_path}.")
        task_config = draccus.parse(cls, str(task_config_path), args=[])
        if task_config.task_name != task_name:
            raise ValueError(f"{task_config_path} holds task {task_config.task_name!r}, not {task_name!r}.")
        return task_config


def override_task_config(
    task_config: KineticManipulationTaskConfig, task_config_overrides: dict[str, Any]
) -> None:
    """Apply field overrides to a task configuration.

    Args:
        task_config (KineticManipulationTaskConfig): Task configuration to update.
        task_config_overrides (dict[str, Any]): Field name to value.
    """
    for field_name, value in task_config_overrides.items():
        if not hasattr(task_config, field_name):
            logger.warning(f"Unknown field '{field_name}' in the task config overrides; skipping")
            continue
        setattr(task_config, field_name, value)
        logger.info(f"Task config override: {field_name} = {value}")
