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

import logging
from pathlib import Path
from unittest.mock import MagicMock

from lerobot.common.train_monitor import (
    TrainMonitor,
    build_kv_message,
    build_monitor_message,
    metadata_rows,
)
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


def _link_urls(blocks: object) -> list[str]:
    urls: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "link" and isinstance(node.get("url"), str):
                urls.append(node["url"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(blocks)
    return urls


def test_kv_message_uses_the_section_icon():
    text, blocks = build_kv_message("model", {"policy": "act"})

    assert text == ":brain: Model\npolicy: act"
    assert blocks[0]["text"]["text"] == "*:brain: Model*"
    assert blocks[1]["type"] == "table"


def test_monitor_line_is_status_progress_and_elapsed():
    text, blocks = build_monitor_message("running", "10%", "2m 4s")

    assert text == ":bar_chart: running  |  10%  |  2m 4s"
    assert blocks == [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*:bar_chart: running  |  10%  |  2m 4s*"},
        }
    ]


def test_metadata_rows_include_run_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.common.train_monitor.git_sha", lambda: "abc1234")
    monkeypatch.setattr("lerobot.common.train_monitor.host_name", lambda: "train-box")
    rows = dict(metadata_rows(_config(tmp_path)))

    assert rows["dataset"] == "user/dataset"
    assert rows["policy"] == "act"
    assert rows["log_freq"] == "200"
    assert rows["save_freq"] == "10000"
    assert rows["output_dir"] == "model_zoo/dataset/custom-job"
    assert rows["tensorboard"] == "model_zoo/dataset/custom-job/tensorboard"
    assert rows["s3"] == "s3://lerobot-train-outputs/model_zoo/dataset/custom-job/"
    assert rows["hostname"] == "train-box"
    assert rows["git_sha"] == "abc1234"
    assert "chunk_size" in rows
    assert "backbone" in rows


def test_parent_message_posts_metadata_and_saves_thread(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.common.train_monitor.git_sha", lambda: "abc1234")
    monkeypatch.setattr("lerobot.common.train_monitor.host_name", lambda: "train-box")
    cfg = _config(tmp_path)
    client = MagicMock()
    client.chat_postMessage.side_effect = [
        {"ts": "1710000000.000100", "channel": "C1"},
        {"ts": "1710000000.000110", "channel": "C1"},
        {"ts": "1710000000.000120", "channel": "C1"},
        {"ts": "1710000000.000130", "channel": "C1"},
        {"ts": "1710000000.000200", "channel": "C1"},
    ]
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())

    monitor.start(0)

    posts = client.chat_postMessage.call_args_list
    parent = posts[0].kwargs
    assert parent["channel"] == "C1"
    assert "thread_ts" not in parent
    assert parent["text"] == ":robot_face: Training custom-job"
    assert parent["blocks"] == [
        {"type": "section", "text": {"type": "mrkdwn", "text": "*:robot_face: Training custom-job*"}}
    ]
    assert sum(block["type"] == "table" for block in parent["blocks"]) == 0
    headings = [post.kwargs["blocks"][0]["text"]["text"] for post in posts[1:4]]
    assert headings == [
        "*:brain: Model*",
        "*:file_folder: Data*",
        "*:gear: Training*",
    ]
    for post in posts[1:4]:
        assert post.kwargs["thread_ts"] == "1710000000.000100"
        assert post.kwargs["blocks"][1]["type"] == "table"
    training = posts[3].kwargs
    assert "dataset: user/dataset" in posts[2].kwargs["text"]
    s3_links = [url for url in _link_urls(training["blocks"]) if "s3.console.aws.amazon.com" in url]
    assert s3_links == [
        "https://s3.console.aws.amazon.com/s3/buckets/lerobot-train-outputs"
        "?region=us-west-1&prefix=model_zoo/dataset/custom-job/"
    ]
    monitor_reply = posts[4].kwargs
    assert monitor_reply["thread_ts"] == "1710000000.000100"
    assert monitor_reply["text"].startswith(":bar_chart: running  |  0%  |  ")
    assert [block["type"] for block in monitor_reply["blocks"]] == ["section"]
    assert "loss" not in monitor_reply["text"]
    saved = load_json(cfg.output_dir / "monitor" / "slack_thread.json")
    assert saved == {
        "thread_ts": "1710000000.000100",
        "channel": "C1",
        "progress_ts": "1710000000.000200",
    }


def test_resume_updates_the_existing_reply(tmp_path):
    cfg = _config(tmp_path)
    cfg.resume = True
    write_json(
        {
            "thread_ts": "1710000000.000100",
            "channel": "C1",
            "progress_ts": "1710000000.000200",
        },
        cfg.output_dir / "monitor" / "slack_thread.json",
    )
    client = MagicMock()
    client.chat_update.return_value = {"ok": True, "ts": "1710000000.000200", "channel": "C1"}
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())

    monitor.start(1000)

    client.chat_postMessage.assert_not_called()
    kwargs = client.chat_update.call_args.kwargs
    assert kwargs["ts"] == "1710000000.000200"
    assert kwargs["channel"] == "C1"
    assert "thread_ts" not in kwargs
    assert "Training" not in kwargs["text"]
    assert kwargs["text"].startswith(":bar_chart: resumed  |  1%  |  ")
    assert [block["type"] for block in kwargs["blocks"]] == ["section"]
    saved = load_json(cfg.output_dir / "monitor" / "slack_thread.json")
    assert saved["thread_ts"] == "1710000000.000100"
    assert saved["progress_ts"] == "1710000000.000200"


def test_finish_reply_has_duration_and_losses_without_a_title(tmp_path):
    cfg = _config(tmp_path)
    client = MagicMock()
    client.chat_update.return_value = {"ok": True}
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())
    monitor.thread_ts = "1710000000.000100"
    monitor.progress_ts = "1710000000.000200"
    monitor.channel = "C1"
    monitor.latest_train_loss = 0.25
    monitor.latest_test_loss = 0.4

    monitor.finish(1000)

    client.chat_postMessage.assert_not_called()
    kwargs = client.chat_update.call_args.kwargs
    assert kwargs["ts"] == "1710000000.000200"
    assert "thread_ts" not in kwargs
    assert "Training" not in kwargs["text"]
    assert kwargs["text"].startswith(":bar_chart: finished  |  1%  |  ")
    assert [block["type"] for block in kwargs["blocks"]] == ["section"]
    assert "train_loss" not in kwargs["text"]
    assert "eval_loss" not in kwargs["text"]
    assert "test_loss" not in kwargs["text"]


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
    client.chat_postMessage.side_effect = RuntimeError("slack down")
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())

    with caplog.at_level(logging.ERROR):
        monitor.start(0)

    assert "Failed to post training Slack notification" in caplog.text
    assert not (cfg.output_dir / "monitor" / "slack_thread.json").exists()


def test_web_client_posts_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    cfg = _config(tmp_path)
    created: dict[str, object] = {}

    class _Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat_postMessage(self, **kwargs: object) -> dict[str, object]:  # noqa: N802
            self.calls.append(kwargs)
            return {"ok": True, "ts": f"1.{len(self.calls)}", "channel": "C1"}

    def fake_web_client(token: str) -> _Client:
        created["token"] = token
        client = _Client()
        created["client"] = client
        return client

    monkeypatch.setattr("lerobot.common.train_monitor._web_client", fake_web_client)
    monitor = TrainMonitor(cfg, writer=MagicMock())
    monitor.start(0)

    assert created["token"] == "xoxb-test"
    client = created["client"]
    assert isinstance(client, _Client)
    assert client.calls[0]["channel"] == "C1"
    assert str(client.calls[0]["text"]).startswith(":robot_face: Training custom-job")
    assert "thread_ts" not in client.calls[0]
    assert client.calls[1]["thread_ts"] == "1.1"


def test_checkpoints_update_one_thread_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lerobot.common.train_monitor.render_loss_plot",
        lambda *_args, **_kwargs: b"\x89PNG",
    )
    cfg = _config(tmp_path)
    client = MagicMock()
    client.chat_postMessage.side_effect = [
        {"ts": "1710000000.000100", "channel": "C1"},
        {"ts": "1710000000.000110", "channel": "C1"},
        {"ts": "1710000000.000120", "channel": "C1"},
        {"ts": "1710000000.000130", "channel": "C1"},
        {"ts": "1710000000.000200", "channel": "C1"},
    ]
    client.files_upload_v2.side_effect = [
        {"files": [{"id": "F1"}]},
        {"files": [{"id": "F2"}]},
        {"files": [{"id": "F3"}]},
    ]
    client.chat_update.return_value = {"ok": True}
    monitor = TrainMonitor(cfg, client=client, writer=MagicMock())

    monitor.start(0)

    posts = client.chat_postMessage.call_args_list
    assert len(posts) == 5
    assert "thread_ts" not in posts[0].kwargs
    assert posts[4].kwargs["thread_ts"] == "1710000000.000100"

    monitor.log_scalars({"loss": 0.9}, step=1)
    assert client.chat_update.call_count == 0

    monitor.log_scalars({"eval_loss": 0.8}, step=2, mode="eval")
    assert client.chat_postMessage.call_count == 5
    first = client.chat_update.call_args.kwargs
    assert first["ts"] == "1710000000.000200"
    assert first["channel"] == "C1"
    assert "thread_ts" not in first
    assert first["text"].startswith(":bar_chart: running  |  0%  |  ")
    assert "train_loss" not in first["text"]
    assert "eval_loss" not in first["text"]
    image = next(block for block in first["blocks"] if block["type"] == "image")
    assert image["alt_text"] == "loss"
    assert image["slack_file"] == {"id": "F1"}
    assert client.files_upload_v2.call_args_list[0].kwargs["request_file_info"] is False
    client.files_info.assert_not_called()
    client.files_delete.assert_not_called()

    monitor.log_scalars({"loss": 0.4}, step=3, mode="test")
    second = client.chat_update.call_args.kwargs
    assert second["ts"] == "1710000000.000200"
    assert second["text"].startswith(":bar_chart: running  |  0%  |  ")
    assert "test_loss" not in second["text"]
    image = next(block for block in second["blocks"] if block["type"] == "image")
    assert image["slack_file"] == {"id": "F2"}
    client.files_delete.assert_called_once_with(file="F1")

    monitor.finish(3)
    assert client.chat_postMessage.call_count == 5
    finished = client.chat_update.call_args.kwargs
    assert finished["ts"] == "1710000000.000200"
    assert finished["text"].startswith(":bar_chart: finished  |  0%  |  ")
    assert "train_loss" not in finished["text"]
    assert "Training" not in finished["text"]
    client.files_delete.assert_any_call(file="F2")


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
