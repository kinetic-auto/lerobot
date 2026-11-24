#!/usr/bin/env python

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
"""Slack-thread and TensorBoard monitor for offline training.

The parent Slack message carries run metadata. The closing reply stays on that
message's ``thread_ts``. Train and test losses are written as TensorBoard scalars
under ``{output_dir}/tensorboard``. Slack is posted with the standard library, and
the event file is written without the tensorboard package, so the train image does
not need extra dependencies. A Slack or TensorBoard failure is logged and swallowed
so training continues.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.utils.io_utils import load_json, write_json

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)

SLACK_TOKEN_ENV = "SLACK_BOT_TOKEN"
SLACK_CHANNEL_ENV = "SLACK_CHANNEL"
THREAD_FILENAME = "slack_thread.json"
_SLACK_FIELDS_PER_SECTION = 10


def git_sha() -> str | None:
    """Short HEAD of the working tree, or None when git is unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return sha or None


def host_name() -> str:
    return socket.gethostname()


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def metadata_rows(cfg: TrainPipelineConfig) -> list[tuple[str, str]]:
    """Curated run metadata for the parent Slack message."""
    trainable = cfg.trainable_config
    dataset = str(cfg.dataset.root) if cfg.dataset.root else str(cfg.dataset.repo_id)
    rows: list[tuple[str, str]] = [
        ("job_name", str(cfg.job_name)),
        ("dataset", dataset),
        ("policy", str(getattr(trainable, "type", type(trainable).__name__))),
    ]
    for name in ("chunk_size", "n_action_steps", "kl_weight"):
        value = getattr(trainable, name, None)
        if value is not None:
            rows.append((name, str(value)))
    backbone = cfg.backbone or getattr(trainable, "vision_backbone", None)
    if backbone:
        rows.append(("backbone", str(backbone)))
    rows.extend(
        [
            ("batch_size", str(cfg.batch_size)),
            ("steps", str(cfg.steps)),
            ("seed", str(cfg.seed)),
            ("log_freq", str(cfg.log_freq)),
            ("save_freq", str(cfg.save_freq)),
            ("mixed_precision", str(cfg.accelerator.mixed_precision)),
            ("output_dir", str(cfg.output_dir)),
            ("hostname", host_name()),
        ]
    )
    sha = git_sha()
    if sha:
        rows.append(("git_sha", sha))
    return rows


def build_monitor_message(title: str | None, rows: list[tuple[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback text and Block Kit sections.

    A title is included only for the parent message. Thread replies pass
    ``title=None`` so the channel title is not repeated.
    """
    kv_lines = [f"{key}: {value}" for key, value in rows]
    blocks: list[dict[str, Any]] = []
    if title:
        fallback = "\n".join([title, *kv_lines])
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}})
    else:
        fallback = "\n".join(kv_lines)
    fields = [{"type": "mrkdwn", "text": f"*{key}*\n{value}"} for key, value in rows]
    for start in range(0, len(fields), _SLACK_FIELDS_PER_SECTION):
        blocks.append({"type": "section", "fields": fields[start : start + _SLACK_FIELDS_PER_SECTION]})
    return fallback, blocks


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _varint(value: int) -> bytes:
    out = bytearray()
    remaining = value
    while True:
        bits = remaining & 0x7F
        remaining >>= 7
        if remaining:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _length_delimited(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _crc32c_table() -> list[int]:
    table: list[int] = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC32C_TABLE = _crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _scalar_event(wall_time: float, step: int, tag: str, value: float) -> bytes:
    """Protobuf Event with one Summary simple_value. No tensorboard dependency."""
    value_msg = _length_delimited(1, tag.encode()) + _varint((2 << 3) | 5) + struct.pack("<f", value)
    summary = _length_delimited(1, value_msg)
    return (
        _varint((1 << 3) | 1)
        + struct.pack("<d", wall_time)
        + _varint((2 << 3) | 0)
        + _varint(step)
        + _length_delimited(5, summary)
    )


def _file_version_event(wall_time: float) -> bytes:
    return _varint((1 << 3) | 1) + struct.pack("<d", wall_time) + _length_delimited(3, b"brain.Event:2")


class _ScalarEventWriter:
    """Writes TensorBoard scalar events with the standard library."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname().replace(":", "_")
        self._path = log_dir / f"events.out.tfevents.{int(time.time())}.{host}"
        self._file = self._path.open("wb")
        self._write_record(_file_version_event(time.time()))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._write_record(_scalar_event(time.time(), int(step), tag, float(value)))

    def flush(self) -> None:
        if not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def _write_record(self, payload: bytes) -> None:
        header = struct.pack("<Q", len(payload))
        self._file.write(header)
        self._file.write(struct.pack("<I", _masked_crc32c(header)))
        self._file.write(payload)
        self._file.write(struct.pack("<I", _masked_crc32c(payload)))


class _UrlSlackClient:
    """Posts chat messages with urllib."""

    def __init__(self, token: str, timeout: float = 5) -> None:
        self._token = token
        self._timeout = timeout

    def post_message(self, **kwargs: Any) -> dict[str, Any]:
        request = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=json.dumps(kwargs).encode(),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Slack HTTP {exc.code}: {detail}") from exc
        if not isinstance(payload, dict) or not payload.get("ok"):
            error = payload.get("error") if isinstance(payload, dict) else "invalid response"
            raise RuntimeError(f"Slack API error: {error}")
        return payload


class TrainMonitor:
    """Writes train and test loss scalars to TensorBoard and posts a Slack thread."""

    def __init__(self, cfg: TrainPipelineConfig, *, client: Any = None, writer: Any = None) -> None:
        if cfg.output_dir is None:
            raise ValueError("TrainMonitor requires cfg.output_dir.")
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.monitor_dir = self.output_dir / "monitor"
        self._thread_path = self.monitor_dir / THREAD_FILENAME
        self._client = client
        self._writer = writer
        self._slack_unavailable = False
        self._tensorboard_unavailable = False
        self.thread_ts: str | None = None
        self.channel: str | None = self._configured_channel()
        self.started_at = time.perf_counter()
        self.latest_train_loss: float | None = None
        self.latest_test_loss: float | None = None

    def start(self, step: int = 0) -> None:
        """Open TensorBoard and post the parent message, or a resume reply."""
        self._open_writer()
        if self.cfg.resume:
            saved = self._read_thread()
            if saved is not None:
                self.thread_ts = str(saved["thread_ts"])
                saved_channel = saved.get("channel")
                if saved_channel:
                    self.channel = str(saved_channel)
                self._post_thread(
                    [
                        ("status", "resumed"),
                        ("step", str(step)),
                        ("output_dir", str(self.output_dir)),
                    ]
                )
                logger.info("Continuing Slack thread %s", self.thread_ts)
                return
            logger.info("No Slack thread file at %s; posting a new parent message.", self._thread_path)
        self._post_parent()

    def log_scalars(self, metrics: dict[str, Any], step: int, mode: str = "train") -> None:
        """Write numeric metrics to TensorBoard. Non-numeric values are skipped."""
        writer = self._writer_or_none()
        if mode == "train":
            loss = _as_float(metrics.get("loss"))
            if loss is not None:
                self.latest_train_loss = loss
        elif mode == "test":
            test_loss = _as_float(metrics.get("loss"))
            if test_loss is not None:
                self.latest_test_loss = test_loss
        if writer is None:
            return
        for key, value in metrics.items():
            number = _as_float(value)
            if number is None:
                continue
            writer.add_scalar(f"{mode}/{key}", number, step)
        writer.flush()

    def finish(self, step: int) -> None:
        """Post the closing thread reply and close the TensorBoard writer."""
        rows = [
            ("step", str(step)),
            ("duration", format_duration(time.perf_counter() - self.started_at)),
            ("output_dir", str(self.output_dir)),
        ]
        if self.latest_train_loss is not None:
            rows.append(("train_loss", f"{self.latest_train_loss:.4f}"))
        if self.latest_test_loss is not None:
            rows.append(("test_loss", f"{self.latest_test_loss:.4f}"))
        self._post_thread(rows)
        writer = self._writer_or_none()
        if writer is not None:
            writer.close()
        self._writer = None
        self._tensorboard_unavailable = True

    def _post_parent(self) -> None:
        title = f"Training {self.cfg.job_name}"
        rows = metadata_rows(self.cfg)
        fallback, blocks = build_monitor_message(title, rows)
        response = self._post(fallback, blocks, thread_ts=None)
        if response is None:
            return
        thread_ts = response.get("ts")
        if not thread_ts:
            logger.warning("Slack parent message did not return a thread timestamp.")
            return
        self.thread_ts = str(thread_ts)
        channel = response.get("channel") or self.channel
        if channel:
            self.channel = str(channel)
        write_json({"thread_ts": self.thread_ts, "channel": self.channel}, self._thread_path)
        logger.info("Slack thread %s in %s", self.thread_ts, self.channel)

    def _post_thread(self, rows: list[tuple[str, str]]) -> None:
        fallback, blocks = build_monitor_message(None, rows)
        self._post(fallback, blocks, thread_ts=self.thread_ts)

    def _post(self, text: str, blocks: list[dict[str, Any]], *, thread_ts: str | None) -> Any | None:
        client = self._resolve_client()
        channel = self.channel or self._configured_channel()
        if client is None or not channel:
            return None
        kwargs: dict[str, Any] = {"channel": channel, "text": text, "blocks": blocks}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        try:
            return client.post_message(**kwargs)
        except Exception:
            logger.exception("Failed to post training Slack notification")
            return None

    def _configured_channel(self) -> str | None:
        channel = (self.cfg.monitor.slack_channel or os.environ.get(SLACK_CHANNEL_ENV) or "").strip()
        return channel or None

    def _resolve_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if self._slack_unavailable:
            return None
        token = os.environ.get(SLACK_TOKEN_ENV, "").strip()
        channel = self._configured_channel()
        if not token:
            logger.warning("%s is not set; Slack notifications are disabled.", SLACK_TOKEN_ENV)
            self._slack_unavailable = True
            return None
        if not channel:
            logger.warning(
                "Slack channel is not set (--monitor.slack_channel or %s); Slack notifications are disabled.",
                SLACK_CHANNEL_ENV,
            )
            self._slack_unavailable = True
            return None
        self.channel = channel
        self._client = _UrlSlackClient(token)
        return self._client

    def _open_writer(self) -> None:
        self._writer_or_none()
        if self._writer is not None:
            logger.info("TensorBoard logs: %s", self.output_dir / "tensorboard")

    def _writer_or_none(self) -> Any | None:
        if self._writer is not None:
            return self._writer
        if self._tensorboard_unavailable:
            return None
        self._writer = _ScalarEventWriter(self.output_dir / "tensorboard")
        return self._writer

    def _read_thread(self) -> dict[str, Any] | None:
        if not self._thread_path.is_file():
            return None
        try:
            data = load_json(self._thread_path)
        except Exception:
            logger.exception("Could not read %s", self._thread_path)
            return None
        if not isinstance(data, dict) or not data.get("thread_ts"):
            return None
        return data
