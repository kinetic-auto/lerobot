# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import draccus
import pytest

# Importing lerobot_train eagerly pulls in lerobot.datasets, which needs the
# `dataset` extra. The base CI tier runs without it, so skip the whole module there.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.act.configuration_act import (
    ACTConfig,  # noqa: E402, F401  (registers --policy.type act)
)
from lerobot.scripts.lerobot_train import (  # noqa: E402
    _expand_resume_path_in_argv,
    _remote_target_in_argv,
    train,
)


def _set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["lerobot-train", *args])


def test_remote_target_detected_space_separated(monkeypatch):
    _set_argv(monkeypatch, "--policy.type", "act", "--job.target", "a10g-small")
    assert _remote_target_in_argv() is True


def test_remote_target_detected_equals(monkeypatch):
    _set_argv(monkeypatch, "--job.target=t4-small")
    assert _remote_target_in_argv() is True


def test_local_string_is_not_remote(monkeypatch):
    _set_argv(monkeypatch, "--job.target", "local")
    assert _remote_target_in_argv() is False


def test_no_target_is_not_remote(monkeypatch):
    _set_argv(monkeypatch, "--policy.type", "act")
    assert _remote_target_in_argv() is False


def test_train_dispatches_to_submit_when_remote(monkeypatch):
    """A remote --job.target short-circuits train() to the HF Jobs submitter."""
    import lerobot.scripts.lerobot_train as train_module

    captured = []
    monkeypatch.setattr(train_module, "submit_to_hf", lambda cfg: captured.append(cfg) or "submitted")
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "a10g-small"],
    )
    # Returns the submitter's result and never enters the local training path.
    assert train(cfg) == "submitted"
    assert captured == [cfg]


def test_expand_resume_run_path(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    pretrained = run_dir / "checkpoints" / "last" / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text("{}")
    _set_argv(monkeypatch, f"--resume={run_dir}")

    _expand_resume_path_in_argv()

    assert "--resume=true" in sys.argv
    assert f"--config_path={pretrained / 'train_config.json'}" in sys.argv
    assert f"--output_dir={run_dir}" in sys.argv


def test_expand_resume_keeps_explicit_output_dir(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    pretrained = run_dir / "checkpoints" / "last" / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text("{}")
    output_dir = tmp_path / "other"
    _set_argv(monkeypatch, f"--resume={run_dir}", f"--output_dir={output_dir}")

    _expand_resume_path_in_argv()

    assert sys.argv.count(f"--output_dir={output_dir}") == 1
    assert f"--output_dir={run_dir}" not in sys.argv


def test_expand_resume_leaves_boolean_and_rejects_config_path(monkeypatch, tmp_path):
    _set_argv(monkeypatch, "--resume=true")
    _expand_resume_path_in_argv()
    assert sys.argv == ["lerobot-train", "--resume=true"]

    _set_argv(monkeypatch, f"--resume={tmp_path}", "--config_path=checkpoint.json")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _expand_resume_path_in_argv()
