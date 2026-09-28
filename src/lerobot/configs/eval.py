# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot import envs, policies  # noqa: F401

from . import parser
from .default import DatasetConfig, EvalConfig
from .policies import PreTrainedConfig

logger = getLogger(__name__)


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    # Explicit consent to execute remote code from the Hub (required for hub environments).
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)

        else:
            logger.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type if self.policy is not None else 'scratch'}"
            else:
                self.job_name = (
                    f"{self.env.type}_{self.policy.type if self.policy is not None else 'scratch'}"
                )

            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


OPEN_LOOP_SPLITS = ("train", "val", "test", "all")


@dataclass
class OpenLoopEvalConfig:
    # Dataset whose recorded episodes are replayed. Only `repo_id`, `root`, `revision`, `video_backend`,
    # `episodes` (explicit episode selection), `exclude_features` and `task_override` are used; the
    # training-only fields (`image_transforms`, `split_ratio`, `eval_split`, `use_imagenet_stats`) are ignored.
    dataset: DatasetConfig
    # Checkpoint to evaluate, given as `--policy.path=<checkpoint>/pretrained_model`.
    policy: PreTrainedConfig | None = None
    # Episodes are drawn from the checkpoint's split_info.json: "train", "val", "test" or "all".
    split: str = "all"
    # Episodes evaluated per split; zero or less evaluates every episode of the split.
    episodes_per_split: int = 3
    # Seeds the episode sampling only.
    seed: int | None = 42
    # Defaults to `<checkpoint step dir>/eval_open_loop` next to a local checkpoint.
    output_dir: Path | None = None
    # Write per-episode joint plots and the per-joint MAE summary plot (requires matplotlib).
    save_plots: bool = True
    dpi: int = 150
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)
        if self.policy is None:
            raise ValueError("Open-loop evaluation needs a trained checkpoint: pass `--policy.path=<dir>`.")
        if self.split not in OPEN_LOOP_SPLITS:
            raise ValueError(f"split must be one of {OPEN_LOOP_SPLITS}, got {self.split!r}")
        if self.dataset.streaming:
            raise ValueError(
                "Open-loop evaluation replays frames by index and does not support dataset.streaming=true."
            )
        if not self.output_dir:
            self.output_dir = self._default_output_dir()

    def _default_output_dir(self) -> Path:
        if self.policy is None:
            raise ValueError("A policy config is required to derive the default output_dir.")
        pretrained_path = self.policy.pretrained_path
        if pretrained_path is not None and Path(pretrained_path).is_dir():
            return Path(pretrained_path).parent / "eval_open_loop"
        now = dt.datetime.now()
        return Path("outputs/eval_open_loop") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.policy.type}"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
