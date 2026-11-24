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

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from lerobot.common.train_monitor import TrainMonitor, build_monitor_message, metadata_rows
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.io_utils import load_json, write_json


def _config(tmp_path: Path) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/dataset"),
        policy=ACTConfig(device="cpu"),
        job_name="custom-job",
        output_dir=tmp_path / "run",
        log_freq=200,
    )
    cfg.validate()
    cfg.monitor.slack_channel = "C1"
    return cfg


def test_thread_message_omits_the_parent_title():
    parent_text, parent_blocks = build_monitor_message("Training custom-job", [("job_name", "custom-job")])
    reply_text, reply_blocks = build_monitor_message(None, [("step", "1000"), ("loss", "0.2500")])

    assert parent_text.startswith("Training custom-job")
    assert "job_name: custom-job" in parent_text
    assert parent_blocks[0]["text"]["text"] == "*Training custom-job*"
    assert reply_text == "step: 1000\nloss: 0.2500"
    assert "Training" not in reply_text
    assert all("Training" not in str(block) for block in reply_blocks)


def test_metadata_rows_include_run_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.common.train_monitor.git_sha", lambda: "abc1234")
    monkeypatch.setattr("lerobot.common.train_monitor.host_name", lambda: "train-box")
    rows = dict(metadata_rows(_config(tmp_path)))

    assert rows["job_name"] == "custom-job"
    assert rows["dataset"] == "user/dataset"
    assert rows["policy"] == "act"
    assert rows["log_freq"] == "200"
    assert rows["save_freq"] == "10000"
    assert rows["hostname"] == "train-box"
    assert rows["git_sha"] == "abc1234"
    assert "chunk_size" in rows
    assert "backbone" in rows


def test_parent_message_posts_metadata_and_saves_thread(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.common.train_monitor.git_sha", lambda: "abc1234")
    monkeypatch.setattr("lerobot.common.train_monitor.host_name", lambda: "train-box")
    cfg = _config(tmp_path)
    client = MagicMock()
    client.post_message.return_value = {"ts": "1710000000.000100", "channel": "C1"}
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())

    monitor.start(0)

    kwargs = client.post_message.call_args.kwargs
    assert kwargs["channel"] == "C1"
    assert "thread_ts" not in kwargs
    assert kwargs["text"].startswith("Training custom-job")
    assert "job_name: custom-job" in kwargs["text"]
    assert "dataset: user/dataset" in kwargs["text"]
    assert kwargs["blocks"][0]["text"]["text"] == "*Training custom-job*"
    saved = load_json(cfg.output_dir / "monitor" / "slack_thread.json")
    assert saved == {"thread_ts": "1710000000.000100", "channel": "C1"}


def test_resume_replies_in_the_existing_thread(tmp_path):
    cfg = _config(tmp_path)
    cfg.resume = True
    write_json(
        {"thread_ts": "1710000000.000100", "channel": "C1"},
        cfg.output_dir / "monitor" / "slack_thread.json",
    )
    client = MagicMock()
    client.post_message.return_value = {"ts": "1710000000.000200", "channel": "C1"}
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())

    monitor.start(1000)

    assert client.post_message.call_count == 1
    kwargs = client.post_message.call_args.kwargs
    assert kwargs["thread_ts"] == "1710000000.000100"
    assert "Training" not in kwargs["text"]
    assert "status: resumed" in kwargs["text"]
    assert "step: 1000" in kwargs["text"]
    assert "output_dir:" in kwargs["text"]
    saved = load_json(cfg.output_dir / "monitor" / "slack_thread.json")
    assert saved["thread_ts"] == "1710000000.000100"


def test_finish_reply_has_duration_and_losses_without_a_title(tmp_path):
    cfg = _config(tmp_path)
    client = MagicMock()
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())
    monitor.thread_ts = "1710000000.000100"
    monitor.channel = "C1"
    monitor.latest_train_loss = 0.25
    monitor.latest_test_loss = 0.4

    monitor.finish(1000)

    kwargs = client.post_message.call_args.kwargs
    assert kwargs["thread_ts"] == "1710000000.000100"
    assert "Training" not in kwargs["text"]
    assert "step: 1000" in kwargs["text"]
    assert "duration:" in kwargs["text"]
    assert "output_dir:" in kwargs["text"]
    assert "train_loss: 0.2500" in kwargs["text"]
    assert "test_loss: 0.4000" in kwargs["text"]


def test_missing_token_does_not_raise(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL", raising=False)
    cfg = _config(tmp_path)
    cfg.monitor.slack_channel = None
    monitor = TrainMonitor(cfg, writer=MagicMock())

    with caplog.at_level(logging.WARNING):
        monitor.start(0)

    assert "SLACK_BOT_TOKEN" in caplog.text
    assert not (cfg.output_dir / "monitor" / "slack_thread.json").exists()


def test_slack_api_error_does_not_raise(tmp_path, caplog):
    cfg = _config(tmp_path)
    client = MagicMock()
    client.post_message.side_effect = RuntimeError("slack down")
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())

    with caplog.at_level(logging.ERROR):
        monitor.start(0)

    assert "Failed to post training Slack notification" in caplog.text
    assert not (cfg.output_dir / "monitor" / "slack_thread.json").exists()


def test_stdlib_slack_client_posts_json(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    cfg = _config(tmp_path)
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true, "ts": "1.2", "channel": "C1"}'

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monitor = TrainMonitor(cfg, writer=MagicMock())
    monitor.start(0)

    assert captured["url"] == "https://slack.com/api/chat.postMessage"
    assert captured["auth"] == "Bearer xoxb-test"
    body = json.loads(captured["body"])
    assert body["channel"] == "C1"
    assert body["text"].startswith("Training custom-job")
    assert "thread_ts" not in body


def test_scalar_events_record_train_and_test_loss(tmp_path):
    cfg = _config(tmp_path)
    monitor = TrainMonitor(cfg, client=MagicMock())

    monitor.log_scalars({"loss": 1.5}, step=200)
    monitor.log_scalars({"loss": 0.4}, step=10000, mode="test")
    monitor.finish(10000)

    events = list((cfg.output_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert len(events) == 1
    raw = events[0].read_bytes()
    assert b"train/loss" in raw
    assert b"test/loss" in raw
    assert b"brain.Event:2" in raw


def test_log_scalars_keeps_latest_loss(tmp_path):
    cfg = _config(tmp_path)
    writer = MagicMock()
    monitor = TrainMonitor(cfg, client=MagicMock(), writer=writer)

    monitor.log_scalars({"loss": 1.5, "samples_per_s": 10, "note": "skip"}, step=200)
    monitor.log_scalars({"loss": 0.4}, step=10000, mode="test")

    writer.add_scalar.assert_any_call("train/loss", 1.5, 200)
    writer.add_scalar.assert_any_call("train/samples_per_s", 10.0, 200)
    writer.add_scalar.assert_any_call("test/loss", 0.4, 10000)
    logged_names = [call.args[0] for call in writer.add_scalar.call_args_list]
    assert "train/note" not in logged_names
    assert monitor.latest_train_loss == 1.5
    assert monitor.latest_test_loss == 0.4
