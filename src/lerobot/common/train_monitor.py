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

The parent Slack message is the run title and is not edited. Model, data, and
training config are each one thread reply. The monitor reply is one line,
``icon status  |  progress  |  elapsed``, updated in place at eval, test, and
finish, and it carries the loss plot. Train and test losses
are also written as TensorBoard scalars under ``{output_dir}/tensorboard``. The
event file is written without the tensorboard package. A Slack or TensorBoard
failure is logged and swallowed so training continues.
"""

from __future__ import annotations

import io
import logging
import os
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from lerobot.utils.io_utils import load_json, write_json

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)

SLACK_TOKEN_ENV = "SLACK_BOT_TOKEN"
SLACK_CHANNEL_ENV = "SLACK_CHANNEL"
THREAD_FILENAME = "slack_thread.json"
_FILE_READY_ATTEMPTS = 5
_FILE_READY_DELAY_S = 0.5
_TABLE_CELL_LIMIT = 400
_AWS_REGION = "us-west-1"
_S3_OUTPUT_BUCKET = "lerobot-train-outputs"
# Section name selects the heading icon. Order here is the thread order.
SECTION_ICONS = {
    "model": ":brain:",
    "data": ":file_folder:",
    "training": ":gear:",
    "monitor": ":bar_chart:",
}
# Same separator as console visit alerts (``title  |  system  |  location``).
_TITLE_SEP = "  |  "
_LossSeries = list[tuple[int, float]]
# (key, value) or (key, value, url). An s3:// value is linked when url is omitted.
MonitorRow = tuple[str, str] | tuple[str, str, str]


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


def published_run_dir(cfg: TrainPipelineConfig) -> Path:
    """Run directory printed by setup_lambda_train.sh, independent of a temp output dir."""
    return Path("model_zoo") / cfg.dataset_name / str(cfg.job_name)


def _s3_output_uri(run_dir: Path) -> str:
    return f"s3://{_S3_OUTPUT_BUCKET}/{run_dir.as_posix()}/"


def _s3_console_url(s3_uri: str) -> str:
    """AWS S3 console URL for an ``s3://`` URI."""
    without_scheme = s3_uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not bucket:
        return s3_uri
    prefix = quote(f"{key.rstrip('/')}/" if key else "", safe="/")
    return (
        f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}?region={_AWS_REGION}&prefix={prefix}"
    )


def _row_parts(row: MonitorRow) -> tuple[str, str, str | None]:
    if len(row) == 3:
        return row[0], row[1], row[2] or None
    return row[0], row[1], None


def _raw_table_cell(text: str) -> dict[str, Any]:
    return {"type": "raw_text", "text": str(text)[:_TABLE_CELL_LIMIT]}


def _link_table_cell(url: str, text: str) -> dict[str, Any]:
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": [
                    {
                        "type": "link",
                        "url": str(url)[:_TABLE_CELL_LIMIT],
                        "text": str(text)[:_TABLE_CELL_LIMIT],
                    }
                ],
            }
        ],
    }


def _value_table_cell(value: str, url: str | None) -> dict[str, Any]:
    """Plain text, or one link when the value is ``s3://`` or ``url`` is set."""
    if url:
        return _link_table_cell(url, value)
    if value.startswith("s3://"):
        return _link_table_cell(_s3_console_url(value), value[len("s3://") :])
    return _raw_table_cell(value)


def _kv_table_block(rows: list[MonitorRow]) -> dict[str, Any] | None:
    """Two-column Slack table. Keys stay plain text."""
    if not rows:
        return None
    table_rows: list[list[dict[str, Any]]] = []
    for row in rows:
        key, value, url = _row_parts(row)
        table_rows.append([_raw_table_cell(key), _value_table_cell(value, url)])
    return {
        "type": "table",
        "column_settings": [{"is_wrapped": True}, {"is_wrapped": True}],
        "rows": table_rows,
    }


def _heading_block(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": f"*{text}*"}}


def metadata_sections(cfg: TrainPipelineConfig) -> dict[str, dict[str, str]]:
    """Model, data, and training fields. Keys match ``SECTION_ICONS``."""
    trainable = cfg.trainable_config
    dataset = str(cfg.dataset.root) if cfg.dataset.root else str(cfg.dataset.repo_id)
    model: dict[str, str] = {
        "policy": str(getattr(trainable, "type", type(trainable).__name__)),
    }
    for name in ("chunk_size", "n_action_steps", "kl_weight"):
        value = getattr(trainable, name, None)
        if value is not None:
            model[name] = str(value)
    backbone = cfg.backbone or getattr(trainable, "vision_backbone", None)
    if backbone:
        model["backbone"] = str(backbone)
    mapping = getattr(trainable, "normalization_mapping", None)
    if mapping:
        model["normalization"] = ", ".join(f"{key}:{value}" for key, value in mapping.items())

    data: dict[str, str] = {"dataset": dataset}
    transforms = cfg.dataset.image_transforms
    if transforms.enable:
        data["image_transforms"] = f"true, max {transforms.max_num_transforms}"
    else:
        data["image_transforms"] = "false"
    if cfg.dataset.split_ratio is not None:
        labels = ("train", "val", "test")
        parts = [
            f"{label} {_format_split_weight(weight)}"
            for label, weight in zip(labels, cfg.dataset.split_ratio, strict=False)
        ]
        data["split"] = ", ".join(parts)

    run_dir = published_run_dir(cfg)
    training: dict[str, str] = {
        "batch_size": str(cfg.batch_size),
        "steps": str(cfg.steps),
        "seed": str(cfg.seed),
        "log_freq": str(cfg.log_freq),
        "save_freq": str(cfg.save_freq),
        "mixed_precision": str(cfg.accelerator.mixed_precision),
        "wandb": str(cfg.wandb.enable).lower(),
        "output_dir": run_dir.as_posix(),
        "tensorboard": f"{run_dir.as_posix()}/tensorboard",
        "s3": _s3_output_uri(run_dir),
        "hostname": host_name(),
    }
    sha = git_sha()
    if sha:
        training["git_sha"] = sha
    return {"model": model, "data": data, "training": training}


def metadata_rows(cfg: TrainPipelineConfig) -> list[tuple[str, str]]:
    """Flat key/value view of the config sections. ``s3`` is still the URI."""
    rows: list[tuple[str, str]] = []
    for fields in metadata_sections(cfg).values():
        rows.extend(fields.items())
    return rows


def build_kv_message(icon_key: str, fields: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    """One headed key/value table for ``model``, ``data``, or ``training``.

    An ``s3://`` value becomes the single console link. The monitor reply is
    ``build_monitor_message`` instead of this table.
    """
    heading = f"{SECTION_ICONS[icon_key]} {icon_key.capitalize()}"
    lines = [heading, *[f"{key}: {value}" for key, value in fields.items()]]
    blocks: list[dict[str, Any]] = [_heading_block(heading)]
    table = _kv_table_block(list(fields.items()))
    if table:
        blocks.append(table)
    return "\n".join(lines), blocks


def build_parent_message(cfg: TrainPipelineConfig) -> tuple[str, list[dict[str, Any]]]:
    """Run title only. Config tables are thread replies."""
    title = f":robot_face: Training {cfg.job_name}"
    return title, [_heading_block(title)]


def format_progress(step: int, total: int) -> str:
    """Progress as a whole percent of ``total`` steps."""
    if total <= 0:
        return "0%"
    percent = round(100 * step / total)
    return f"{min(max(percent, 0), 100)}%"


def build_monitor_message(status: str, progress: str, elapsed: str) -> tuple[str, list[dict[str, Any]]]:
    """One line: ``:bar_chart: status  |  progress  |  elapsed``.

    Visit alerts use the same join for a title plus two fields. This line
    replaces the monitor key/value table.
    """
    line = _TITLE_SEP.join([f"{SECTION_ICONS['monitor']} {status}", progress, elapsed])
    return line, [_heading_block(line)]


def render_loss_plot(
    train: _LossSeries,
    eval_points: _LossSeries,
    test: _LossSeries,
) -> bytes | None:
    """One PNG of train, eval, and test loss. None when there is nothing to draw."""
    if not train and not eval_points and not test:
        return None
    try:
        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; Slack loss plot is skipped.")
        return None
    except Exception:
        logger.exception("Could not initialize matplotlib; Slack loss plot is skipped.")
        return None

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    try:
        for points, label in ((train, "train"), (eval_points, "eval"), (test, "test")):
            if not points:
                continue
            steps, losses = zip(*points, strict=True)
            ax.plot(steps, losses, marker="o", label=label)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _mapping_get(payload: Any, key: str) -> Any:
    """Read one field from a dict or a SlackResponse."""
    getter = getattr(payload, "get", None)
    if not callable(getter):
        return None
    return getter(key)


def _uploaded_file_id(uploaded: Any) -> str | None:
    """File id from files.upload or from files.completeUploadExternal.

    ``files.info`` is not required. The bot token only needs ``files:write``.
    """
    file_id = _mapping_get(_mapping_get(uploaded, "file"), "id")
    if isinstance(file_id, str) and file_id:
        return file_id
    files = _mapping_get(uploaded, "files")
    if isinstance(files, list) and files:
        file_id = _mapping_get(files[0], "id")
        if isinstance(file_id, str) and file_id:
            return file_id
    return None


def _format_split_weight(weight: float) -> str:
    """One decimal for tenths, so a zero val weight stays ``0.0``."""
    scaled = weight * 10
    if abs(scaled - round(scaled)) < 1e-6:
        return f"{weight:.1f}"
    return f"{weight:g}"


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


def _web_client(token: str) -> Any:
    from slack_sdk import WebClient

    return WebClient(token=token, timeout=5)


class TrainMonitor:
    """Writes loss scalars to TensorBoard and keeps one editable Slack monitor reply."""

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
        self.progress_ts: str | None = None
        self.channel: str | None = self._configured_channel()
        self.started_at = time.perf_counter()
        self.latest_train_loss: float | None = None
        self.latest_eval_loss: float | None = None
        self.latest_test_loss: float | None = None
        self._train_points: _LossSeries = []
        self._eval_points: _LossSeries = []
        self._test_points: _LossSeries = []
        self._plot_file_id: str | None = None

    def start(self, step: int = 0) -> None:
        """Open TensorBoard, post the title, then one reply per config section."""
        self._open_writer()
        if self.cfg.resume:
            saved = self._read_thread()
            if saved is not None:
                self._restore_thread(saved)
                fallback, blocks = self._monitor_message(step, status="resumed")
                if self.progress_ts:
                    self._update_reply(fallback, blocks)
                else:
                    self._post_progress(fallback, blocks)
                logger.info("Continuing Slack thread %s", self.thread_ts)
                return
            logger.info("No Slack thread file at %s; posting a new parent message.", self._thread_path)
        self._post_parent()
        self._post_config_replies()
        self._post_progress(*self._monitor_message(step, status="running"))

    def log_scalars(self, metrics: dict[str, Any], step: int, mode: str = "train") -> None:
        """Write numeric metrics to TensorBoard. Non-numeric values are skipped.

        Train loss is kept for the plot. Eval and test also refresh the single
        Slack reply. Train log steps do not upload a plot.
        """
        refresh = False
        if mode == "train":
            loss = _as_float(metrics.get("loss"))
            if loss is not None:
                self.latest_train_loss = loss
                self._train_points.append((step, loss))
        elif mode == "eval":
            eval_loss = _as_float(metrics.get("eval_loss"))
            if eval_loss is not None:
                self.latest_eval_loss = eval_loss
                self._eval_points.append((step, eval_loss))
                refresh = True
        elif mode == "test":
            test_loss = _as_float(metrics.get("loss"))
            if test_loss is not None:
                self.latest_test_loss = test_loss
                self._test_points.append((step, test_loss))
                refresh = True
        writer = self._writer_or_none()
        if writer is not None:
            for key, value in metrics.items():
                number = _as_float(value)
                if number is None:
                    continue
                writer.add_scalar(f"{mode}/{key}", number, step)
            writer.flush()
        if refresh and self.progress_ts:
            self._update_reply(*self._monitor_message(step, status="running"))

    def finish(self, step: int) -> None:
        """Update the monitor reply and close TensorBoard."""
        fallback, blocks = self._monitor_message(step, status="finished")
        if self.progress_ts:
            self._update_reply(fallback, blocks)
        else:
            self._post_progress(fallback, blocks)
        writer = self._writer_or_none()
        if writer is not None:
            writer.close()
        self._writer = None
        self._tensorboard_unavailable = True

    def _monitor_message(self, step: int, *, status: str) -> tuple[str, list[dict[str, Any]]]:
        elapsed = format_duration(time.perf_counter() - self.started_at)
        return build_monitor_message(status, format_progress(step, self.cfg.steps), elapsed)

    def _post_parent(self) -> None:
        fallback, blocks = build_parent_message(self.cfg)
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
        self._write_thread()
        logger.info("Slack thread %s in %s", self.thread_ts, self.channel)

    def _post_config_replies(self) -> None:
        for icon_key, fields in metadata_sections(self.cfg).items():
            if not self.thread_ts:
                return
            fallback, blocks = build_kv_message(icon_key, fields)
            self._post(fallback, blocks, thread_ts=self.thread_ts)

    def _post_progress(self, fallback: str, blocks: list[dict[str, Any]]) -> None:
        if not self.thread_ts:
            return
        response = self._post(fallback, blocks, thread_ts=self.thread_ts)
        if response is None:
            return
        progress_ts = response.get("ts")
        if not progress_ts:
            logger.warning("Slack progress reply did not return a timestamp.")
            return
        self.progress_ts = str(progress_ts)
        self._write_thread()

    def _update_reply(self, fallback: str, blocks: list[dict[str, Any]]) -> None:
        if not self.progress_ts:
            return
        blocks = list(blocks)
        new_file_id = self._upload_loss_plot()
        image_id = new_file_id or self._plot_file_id
        if image_id:
            blocks.append(
                {
                    "type": "image",
                    "slack_file": {"id": image_id},
                    "alt_text": "loss",
                }
            )
        # A just-uploaded image can be rejected for a moment. Retry the same
        # message instead of calling files.info, which needs files:read.
        attempts = _FILE_READY_ATTEMPTS if new_file_id else 1
        response = None
        for attempt in range(attempts):
            response = self._update(
                fallback,
                blocks,
                ts=self.progress_ts,
                quiet=attempt + 1 < attempts,
            )
            if response is not None:
                break
            time.sleep(_FILE_READY_DELAY_S)
        if response is None:
            return
        if new_file_id and self._plot_file_id and new_file_id != self._plot_file_id:
            self._delete_file(self._plot_file_id)
        if new_file_id:
            self._plot_file_id = new_file_id
            self._write_thread()

    def _upload_loss_plot(self) -> str | None:
        png = render_loss_plot(self._train_points, self._eval_points, self._test_points)
        if png is None:
            return None
        client = self._resolve_client()
        if client is None:
            return None
        try:
            # request_file_info calls files.info, which needs files:read.
            uploaded = client.files_upload_v2(
                filename="loss.png",
                content=png,
                title="loss",
                request_file_info=False,
            )
        except Exception:
            logger.exception("Failed to upload Slack loss plot")
            return None
        file_id = _uploaded_file_id(uploaded)
        if file_id is None:
            logger.warning("Slack file upload did not return a file id.")
            return None
        return file_id

    def _delete_file(self, file_id: str) -> None:
        client = self._resolve_client()
        if client is None:
            return
        try:
            client.files_delete(file=file_id)
        except Exception:
            logger.exception("Failed to delete Slack file %s", file_id)

    def _post(self, text: str, blocks: list[dict[str, Any]], *, thread_ts: str | None) -> Any | None:
        client = self._resolve_client()
        channel = self.channel or self._configured_channel()
        if client is None or not channel:
            return None
        kwargs: dict[str, Any] = {"channel": channel, "text": text, "blocks": blocks}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        try:
            return client.chat_postMessage(**kwargs)
        except Exception:
            logger.exception("Failed to post training Slack notification")
            return None

    def _update(
        self,
        text: str,
        blocks: list[dict[str, Any]],
        *,
        ts: str,
        quiet: bool = False,
    ) -> Any | None:
        client = self._resolve_client()
        channel = self.channel or self._configured_channel()
        if client is None or not channel:
            return None
        try:
            return client.chat_update(channel=channel, ts=ts, text=text, blocks=blocks)
        except Exception:
            if quiet:
                logger.debug("Slack progress update failed; retrying", exc_info=True)
            else:
                logger.exception("Failed to update training Slack notification")
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
        try:
            self._client = _web_client(token)
        except ImportError:
            logger.warning("slack-sdk is not installed; Slack notifications are disabled.")
            self._slack_unavailable = True
            return None
        return self._client

    def _restore_thread(self, saved: dict[str, Any]) -> None:
        self.thread_ts = str(saved["thread_ts"])
        saved_channel = saved.get("channel")
        if saved_channel:
            self.channel = str(saved_channel)
        progress_ts = saved.get("progress_ts")
        if progress_ts:
            self.progress_ts = str(progress_ts)
        plot_file_id = saved.get("plot_file_id")
        if plot_file_id:
            self._plot_file_id = str(plot_file_id)

    def _write_thread(self) -> None:
        payload: dict[str, str | None] = {
            "thread_ts": self.thread_ts,
            "channel": self.channel,
        }
        if self.progress_ts:
            payload["progress_ts"] = self.progress_ts
        if self._plot_file_id:
            payload["plot_file_id"] = self._plot_file_id
        write_json(payload, self._thread_path)

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
