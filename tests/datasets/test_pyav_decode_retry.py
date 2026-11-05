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

import pytest
import torch

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

from lerobot.datasets import video_utils  # noqa: E402
from lerobot.datasets.video_utils import (  # noqa: E402
    PYAV_RETRY_SEEK_BACK_S,
    FrameTimestampError,
    decode_video_frames_pyav,
)

TOLERANCE_S = 1e-4


def make_frame(value: int) -> torch.Tensor:
    return torch.full((1, 2, 2), value, dtype=torch.int64)


def stub_decoder(monkeypatch, decoded_timestamps_per_pass: list[list[float]]) -> list[float]:
    """Make each decode pass return the given timestamps and record the extra seek-back of every pass."""
    seek_back_per_pass: list[float] = []

    def fake_decode(video_path, first_ts, last_ts, extra_seek_back_s, is_depth, log_loaded_timestamps):
        pass_index = len(seek_back_per_pass)
        seek_back_per_pass.append(extra_seek_back_s)
        decoded_timestamps = decoded_timestamps_per_pass[pass_index]
        return [make_frame(round(t * 10)) for t in decoded_timestamps], decoded_timestamps

    monkeypatch.setattr(video_utils, "_decode_pyav_frames_after_seek", fake_decode)
    return seek_back_per_pass


def test_decode_does_not_retry_when_the_first_pass_covers_the_timestamps(monkeypatch):
    seek_back_per_pass = stub_decoder(monkeypatch, [[6.4, 6.5]])

    frames = decode_video_frames_pyav("video.mp4", [6.5], TOLERANCE_S, is_depth=True)

    assert seek_back_per_pass == [0.0]
    assert frames.tolist() == [[[[65, 65], [65, 65]]]]


def test_decode_retries_from_an_earlier_keyframe_when_a_timestamp_is_missing(monkeypatch):
    # The first pass lands on a keyframe that drops the 6.5 s picture, the retry decodes it.
    seek_back_per_pass = stub_decoder(monkeypatch, [[6.4, 6.6], [6.3, 6.4, 6.5, 6.6]])

    frames = decode_video_frames_pyav("video.mp4", [6.5], TOLERANCE_S, is_depth=True)

    assert seek_back_per_pass == [0.0, PYAV_RETRY_SEEK_BACK_S]
    assert frames.tolist() == [[[[65, 65], [65, 65]]]]


def test_decode_warns_when_only_the_retry_finds_the_timestamp(monkeypatch, caplog):
    stub_decoder(monkeypatch, [[6.4, 6.6], [6.4, 6.5, 6.6]])

    with caplog.at_level("WARNING"):
        decode_video_frames_pyav("video.mp4", [6.5], TOLERANCE_S, is_depth=True)

    assert "video.mp4" in caplog.text


def test_decode_retries_when_the_first_pass_decodes_no_frame(monkeypatch):
    seek_back_per_pass = stub_decoder(monkeypatch, [[], [6.5]])

    decode_video_frames_pyav("video.mp4", [6.5], TOLERANCE_S, is_depth=True)

    assert seek_back_per_pass == [0.0, PYAV_RETRY_SEEK_BACK_S]


def test_decode_raises_when_the_retry_still_misses_a_timestamp(monkeypatch):
    seek_back_per_pass = stub_decoder(monkeypatch, [[6.4, 6.6], [6.4, 6.6]])

    with pytest.raises(FrameTimestampError, match="violate the tolerance"):
        decode_video_frames_pyav("video.mp4", [6.5], TOLERANCE_S, is_depth=True)

    assert seek_back_per_pass == [0.0, PYAV_RETRY_SEEK_BACK_S]


def test_decode_raises_when_no_pass_decodes_a_frame(monkeypatch):
    stub_decoder(monkeypatch, [[], []])

    with pytest.raises(FrameTimestampError, match="No frames could be decoded"):
        decode_video_frames_pyav("video.mp4", [6.5], TOLERANCE_S, is_depth=True)
