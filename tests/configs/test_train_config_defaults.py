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

import re
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


def test_train_config_resolves_local_run_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(root="/datasets/pick-cube"),
        policy=ACTConfig(device="cpu"),
    )

    cfg.validate()

    assert re.fullmatch(r"act_\d{8}_\d{6}", cfg.job_name)
    assert cfg.output_dir == Path("model_zoo") / "pick-cube" / cfg.job_name
    assert cfg.wandb.project == "pick-cube"
    assert cfg.env_eval_freq == 0
    assert cfg.save_freq == 10_000


def test_train_config_keeps_explicit_run_values(tmp_path):
    output_dir = tmp_path / "custom"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/dataset"),
        policy=ACTConfig(device="cpu"),
        job_name="custom-job",
        output_dir=output_dir,
        wandb=WandBConfig(project="custom-project"),
    )

    cfg.validate()

    assert cfg.job_name == "custom-job"
    assert cfg.output_dir == output_dir
    assert cfg.wandb.project == "custom-project"
