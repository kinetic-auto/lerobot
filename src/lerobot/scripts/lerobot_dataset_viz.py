#!/usr/bin/env python

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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Requires: pip install 'lerobot[dataset_viz]'  (includes dataset + viz extras)

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize a local collected dataset (no Hub repo-id needed):
```
local$ lerobot-dataset-viz \
    --root /path/to/car-door-opening-20260910 \
    --list-episodes

local$ lerobot-dataset-viz \
    --root /path/to/car-door-opening-20260910 \
    --episode-index 0
```
Opens the Rerun web viewer in a browser.

- Visualize a Hub dataset:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876

local$ rerun rerun+http://IP:GRPC_PORT/proxy
```

- Visualize data in Foxglove with a seekable, scrubbable timeline:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --display-mode foxglove

# then open the Foxglove app and connect to ws://127.0.0.1:8765
```
This starts a Foxglove WebSocket server that serves the episode on demand from the on-disk dataset,
so you can play/pause and scrub anywhere in the episode using Foxglove's playback controls.

"""

import argparse
import gc
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import tqdm

from lerobot.configs import DEPTH_MILLIMETER_UNIT
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, SUCCESS
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

DEFAULT_FOXGLOVE_PORT = 8765
DEFAULT_RERUN_PORT = 9090
PREFERRED_SCALAR_GROUPS = ("position", "velocity", "effort")


def infer_repo_id(repo_id: str | None, root: str | None) -> str:
    """Return a repo id, inferring it from the local folder name when omitted."""
    if repo_id:
        return repo_id
    if not root:
        raise ValueError("Provide --repo-id or --root")
    return Path(root).expanduser().resolve().name


def resolve_dataset_root(root: str | None) -> str | None:
    """Expand a local dataset path and require ``meta/info.json`` when set.

    Object-store URIs are returned unchanged so ``Path`` does not mangle them.
    """
    if root is None:
        return None
    if "://" in root:
        return root
    root_path = Path(root).expanduser().resolve()
    info_path = root_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"No LeRobot dataset at {root_path}: missing {info_path}")
    return str(root_path)


def get_feature_names(dataset: LeRobotDataset, key: str) -> list[str]:
    """Return per-dimension names for a feature from the dataset metadata.

    Only flat-list ``names`` metadata is used. Dict-style ``names`` and missing names fall back to ``{key}_{i}`` indices.
    """
    feature = dataset.features[key]
    dim = feature["shape"][-1]

    names = feature.get("names")
    if isinstance(names, list) and len(names) == dim:
        return [str(name) for name in names]

    return [f"{key}_{d}" for d in range(dim)]


def group_feature_dims(names: list[str]) -> list[tuple[str, list[int], list[str]]]:
    """Split feature dimensions into groups by the trailing name suffix.

    ``right_joint1.position`` lands in group ``position``. Names without a ``.``
    stay in a single unnamed group so the original one-plot layout is preserved.

    Returns a list of ``(group_name, indices, series_names)``.
    """
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        suffix = name.rsplit(".", 1)[-1] if "." in name else ""
        groups.setdefault(suffix, []).append(i)

    ordered = [key for key in PREFERRED_SCALAR_GROUPS if key in groups]
    ordered.extend(key for key in groups if key not in ordered)

    return [(key, groups[key], [names[i] for i in groups[key]]) for key in ordered]


def scalar_entity_path(base: str, group_name: str) -> str:
    return f"{base}/{group_name}" if group_name else base


def select_video_frames(
    frame_timestamps_ns: np.ndarray,
    from_s: float,
    to_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick video-file timestamps that fall in ``[from_s, to_s)`` and return episode-relative seconds."""
    frame_s = frame_timestamps_ns.astype(np.float64) * 1e-9
    mask = (frame_s >= from_s - 1e-4) & (frame_s < to_s + 1e-4)
    selected_ns = np.asarray(frame_timestamps_ns[mask], dtype=np.int64)
    if selected_ns.size == 0:
        return selected_ns, np.zeros(0, dtype=np.float64)
    relative_s = selected_ns.astype(np.float64) * 1e-9 - from_s
    return selected_ns, relative_s


def _can_stream_videos(dataset: LeRobotDataset) -> bool:
    """True when every camera is an RGB video file (no per-frame decode needed)."""
    if not dataset.meta.camera_keys:
        return False
    return all(
        key in dataset.meta.video_keys and key not in dataset.meta.depth_keys
        for key in dataset.meta.camera_keys
    )


def _ffmpeg_executable() -> str | None:
    """Prefer PATH ffmpeg, then the binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        return None


def _extract_video_clip(src: Path, from_s: float, to_s: float, dest: Path) -> bool:
    """Cut ``[from_s, to_s)`` from ``src`` with stream-copy. Returns True on success."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{from_s:.6f}",
        "-to",
        f"{to_s:.6f}",
        "-i",
        str(src),
        "-c",
        "copy",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return dest.is_file() and dest.stat().st_size > 0


def _as_2d_float(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def log_episode_videos(dataset: LeRobotDataset, episode_index: int) -> None:
    """Log each RGB camera as an MP4 asset plus frame references (no Python decode)."""
    import rerun as rr

    clip_dir = Path(tempfile.mkdtemp(prefix="lerobot_viz_clips_"))
    ep = dataset.meta.episodes[episode_index]
    for key in dataset.meta.camera_keys:
        if key not in dataset.meta.video_keys or key in dataset.meta.depth_keys:
            continue
        src = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        if not src.is_file():
            logging.warning("Missing video for %s: %s", key, src)
            continue
        from_s = float(ep[f"videos/{key}/from_timestamp"])
        to_s = float(ep[f"videos/{key}/to_timestamp"])
        clip = clip_dir / f"{key.replace('.', '_')}.mp4"
        used_clip = _extract_video_clip(src, from_s, to_s, clip)
        if used_clip:
            video_path = clip
        else:
            logging.warning("Could not cut a clip for %s; sending the full video file %s", key, src.name)
            video_path = src
        logging.info("Logging %s from %s (%.1fs)", key, video_path.name, to_s - from_s)
        video_asset = rr.AssetVideo(path=str(video_path))
        rr.log(key, video_asset, static=True)
        frame_ns = video_asset.read_frame_timestamps_nanos()
        if used_clip:
            selected_ns = frame_ns
            relative_s = selected_ns.astype(np.float64) * 1e-9
            if relative_s.size:
                relative_s = relative_s - relative_s[0]
        else:
            selected_ns, relative_s = select_video_frames(frame_ns, from_s, to_s)
        if selected_ns.size == 0:
            logging.warning("No frames selected for %s in [%.3f, %.3f)", key, from_s, to_s)
            continue
        frame_index = np.rint(relative_s * float(dataset.fps)).astype(np.int64)
        rr.send_columns(
            key,
            indexes=[
                rr.TimeColumn("timestamp", duration=relative_s),
                rr.TimeColumn("frame_index", sequence=frame_index),
            ],
            columns=rr.VideoFrameReference.columns_nanos(selected_ns),
        )


def log_episode_scalars(dataset: LeRobotDataset) -> None:
    """Log action/state (and optional reward flags) from parquet via send_columns."""
    import rerun as rr

    hf = dataset.hf_dataset.with_format("numpy")
    timestamps = np.asarray(hf["timestamp"], dtype=np.float64).reshape(-1)
    frame_index = np.asarray(hf["frame_index"], dtype=np.int64).reshape(-1)
    indexes = [
        rr.TimeColumn("timestamp", duration=timestamps),
        rr.TimeColumn("frame_index", sequence=frame_index),
    ]

    for origin, key in ((ACTION, ACTION), ("state", OBS_STATE)):
        if key not in hf.column_names:
            continue
        values = _as_2d_float(hf[key])
        names = get_feature_names(dataset, key)
        for group_name, dim_indices, series_names in group_feature_dims(names):
            entity = scalar_entity_path(origin, group_name)
            grouped = np.ascontiguousarray(values[:, dim_indices])
            n, dim = grouped.shape
            columns = rr.Scalars.columns(scalars=grouped.reshape(-1))
            if dim > 1:
                columns = columns.partition([dim] * n)
            rr.log(entity, rr.SeriesLines(names=series_names), static=True)
            rr.send_columns(entity, indexes=indexes, columns=columns)

    for key in (DONE, REWARD, SUCCESS):
        if key not in hf.column_names:
            continue
        values = np.asarray(hf[key], dtype=np.float64).reshape(-1)
        rr.send_columns(key, indexes=indexes, columns=rr.Scalars.columns(scalars=values))


def format_episode_table(meta: LeRobotDatasetMetadata) -> str:
    """Return a human-readable episode listing for a local or Hub dataset."""
    lines = [
        f"Dataset:  {meta.repo_id}",
        f"Root:     {meta.root}",
        f"Robot:    {meta.robot_type}",
        f"FPS:      {meta.fps}",
        f"Episodes: {meta.total_episodes}",
        f"Frames:   {meta.total_frames}",
        "",
        f"{'ep':>4}  {'frames':>7}  {'sec':>7}  {'tasks':<32}  videos",
    ]
    fps = float(meta.fps) if meta.fps else 1.0
    for i in range(meta.total_episodes):
        if meta.episodes is None:
            break
        ep = meta.episodes[i]
        length = int(ep["length"])
        duration = length / fps
        tasks = ep.get("tasks", [])
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        if isinstance(tasks, list):
            task_s = ", ".join(str(t) for t in tasks)
        else:
            task_s = str(tasks)
        video_bits: list[str] = []
        for vid_key in meta.video_keys:
            rel = meta.get_video_file_path(i, vid_key)
            label = vid_key.rsplit(".", 1)[-1]
            if (meta.root / rel).is_file():
                video_bits.append(label)
            else:
                video_bits.append(f"{label}(missing)")
        lines.append(
            f"{i:4d}  {length:7d}  {duration:7.1f}  {task_s:<32}  {','.join(video_bits)}"
        )
    return "\n".join(lines)


def check_chw_float32(frame: torch.Tensor) -> None:
    """
    Check if a frame is a channel-first, float32 tensor.
    """
    assert frame.dtype == torch.float32
    assert frame.ndim == 3
    c, h, w = frame.shape
    assert c < h and c < w, f"expect channel first images, but instead {frame.shape}"


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    check_chw_float32(chw_float32_torch)
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def to_hwc_float32_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    check_chw_float32(chw_float32_torch)
    hwc_float32_numpy = chw_float32_torch.permute(1, 2, 0).numpy()
    return hwc_float32_numpy


def build_blueprint_from_dataset(dataset: LeRobotDataset):
    """Build a Rerun blueprint laying out camera images and time series for the given dataset.

    Camera images and scalar signals (action, state, reward, done, success) are arranged in a grid.
    The per-dimension series names for ``action`` and ``state`` are applied directly
    via blueprint overrides.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=key, name=key) for key in dataset.meta.camera_keys]

    # Style multi-dimensional signals (action, state) with per-dimension names.
    # Packed pos/vel/effort vectors get one plot per suffix so 21-dim arms stay readable.
    for origin, key in ((ACTION, ACTION), ("state", OBS_STATE)):
        if key in dataset.features:
            names = get_feature_names(dataset, key)
            for group_name, _indices, series_names in group_feature_dims(names):
                entity = scalar_entity_path(origin, group_name)
                styling = rr.SeriesLines(names=series_names)
                views.append(rrb.TimeSeriesView(origin=entity, name=entity, overrides={entity: styling}))
    for key in (DONE, REWARD, SUCCESS):
        if key in dataset.features:
            views.append(rrb.TimeSeriesView(origin=key, name=key))

    return rrb.Blueprint(rrb.Grid(*views))


def _log_episode_decoded_frames(
    dataset: LeRobotDataset,
    batch_size: int,
    num_workers: int,
    display_compressed_images: bool,
) -> None:
    """Fallback for image/depth datasets: decode frames in Python and log JPEGs."""
    import rerun as rr

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    depth_meter = 1000.0 if dataset.depth_output_unit == DEPTH_MILLIMETER_UNIT else 1.0
    depth_ranges = {}
    for key in dataset.meta.depth_keys:
        stats = (dataset.meta.stats or {}).get(key)
        if not stats:
            continue
        lo = stats["q01"] if "q01" in stats else stats["min"]
        hi = stats["q99"] if "q99" in stats else stats["max"]
        depth_ranges[key] = (float(np.asarray(lo).item()), float(np.asarray(hi).item()))

    first_index = None
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()

        for i in range(len(batch["index"])):
            frame_index = batch["index"][i].item() - first_index
            rr.set_time("frame_index", sequence=int(frame_index))
            rr.set_time("timestamp", duration=float(batch["timestamp"][i].item()))

            for key in dataset.meta.camera_keys:
                if key in dataset.meta.depth_keys:
                    depth = to_hwc_float32_numpy(batch[key][i])
                    depth_entity = rr.DepthImage(
                        depth,
                        meter=depth_meter,
                        colormap=rr.components.Colormap.Viridis,
                        depth_range=depth_ranges.get(key),
                    )
                    rr.log(key, entity=depth_entity)
                else:
                    img = to_hwc_uint8_numpy(batch[key][i])
                    img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                    rr.log(key, entity=img_entity)

            if ACTION in batch:
                names = get_feature_names(dataset, ACTION)
                values = batch[ACTION][i].numpy()
                for group_name, indices, _series in group_feature_dims(names):
                    rr.log(scalar_entity_path(ACTION, group_name), rr.Scalars(values[indices]))

            if OBS_STATE in batch:
                names = get_feature_names(dataset, OBS_STATE)
                values = batch[OBS_STATE][i].numpy()
                for group_name, indices, _series in group_feature_dims(names):
                    rr.log(scalar_entity_path("state", group_name), rr.Scalars(values[indices]))

            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if SUCCESS in batch:
                rr.log(SUCCESS, rr.Scalars(batch[SUCCESS][i].item()))


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int | None = None,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    display_mode: str = "rerun",
    host: str = "127.0.0.1",
    autoplay: bool = True,
    **kwargs,
) -> Path | None:
    if display_mode == "foxglove":
        from lerobot.utils.foxglove_visualization import serve_foxglove_dataset_playback

        logging.info("Starting Foxglove server")
        serve_foxglove_dataset_playback(
            dataset,
            episode_index,
            host=host,
            port=web_port if web_port is not None else DEFAULT_FOXGLOVE_PORT,
            compress_images=display_compressed_images,
            autoplay=autoplay,
        )
        return None

    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    from lerobot.utils.import_utils import require_package

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    serve_web = (mode == "distant") or (mode == "local" and not save)
    blueprint = build_blueprint_from_dataset(dataset)
    rr.init(f"{repo_id}/episode_{episode_index}", default_blueprint=blueprint)
    gc.collect()

    server_uri = None
    local_web_port = web_port if web_port is not None else DEFAULT_RERUN_PORT
    if serve_web:
        server_uri = rr.serve_grpc(grpc_port=grpc_port, server_memory_limit="2GiB")
        if mode == "distant":
            logging.info("Connect with: rerun rerun+http://IP:%s/proxy", grpc_port)
            rr.serve_web_viewer(open_browser=False, web_port=local_web_port, connect_to=server_uri)

    logging.info("Logging to Rerun")
    if _can_stream_videos(dataset):
        log_episode_videos(dataset, episode_index)
        log_episode_scalars(dataset)
    else:
        _log_episode_decoded_frames(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            display_compressed_images=display_compressed_images,
        )

    # Flush so the spawned viewer actually receives the last batches.
    recording = rr.get_data_recording()
    if recording is not None:
        recording.flush()

    # save .rrd locally
    if mode == "local" and save:
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    if serve_web and mode == "local":
        logging.info("Opening Rerun in the browser at http://127.0.0.1:%s", local_web_port)
        rr.serve_web_viewer(open_browser=True, web_port=local_web_port, connect_to=server_uri)

    if serve_web:
        logging.info("Logged episode %s. Viewer is open — press play in Rerun, Ctrl-C to exit.", episode_index)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`). Optional when `--root` points at a local dataset folder.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Episode to visualize. Required unless `--list-episodes` is set.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset directory (the folder that contains `meta/`). Also accepts an object-store URI for storage formats that read in place (e.g. `--root s3://bucket/dataset`). Converting to Path would mangle URIs, so this stays a string. By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="Print episode index, length, duration, tasks, and video files, then exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers used only when frames must be decoded (image/depth datasets).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' opens the Rerun web viewer in a browser. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help=(
            "Web/WebSocket port. For rerun `--mode distant` it is the web viewer port (default 9090); "
            "for `--display-mode foxglove` it is the server bind port (default 8765)."
        ),
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="If set, display compressed (JPEG) images instead of uncompressed ones.",
    )

    parser.add_argument(
        "--display-mode",
        type=str,
        default="rerun",
        choices=["rerun", "foxglove"],
        help=(
            "Visualization backend. 'rerun' uses the Rerun viewer (--mode/--save/--*-port apply). "
            "'foxglove' starts a Foxglove WebSocket server that serves the episode as a seekable, "
            "scrubbable timeline; connect the Foxglove app to ws://HOST:PORT (--host/--web-port)."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Host to bind the Foxglove WebSocket server to when `--display-mode foxglove` is set "
            "(127.0.0.1 for local only, 0.0.0.0 for all interfaces)."
        ),
    )
    parser.add_argument(
        "--no-autoplay",
        dest="autoplay",
        action="store_false",
        help=(
            "For `--display-mode foxglove`: don't start playing automatically when a client "
            "connects; wait for play to be pressed in the Foxglove app instead."
        ),
    )

    args = parser.parse_args()

    if args.repo_id is None and args.root is None:
        parser.error("Provide --root (local dataset folder) or --repo-id (Hub dataset)")
    if not args.list_episodes and args.episode_index is None:
        parser.error("--episode-index is required unless --list-episodes is set")

    if args.display_mode == "foxglove":
        rerun_only = ("mode", "save", "output_dir", "grpc_port", "batch_size", "num_workers")
        ignored = [name for name in rerun_only if getattr(args, name) != parser.get_default(name)]
        if ignored:
            logging.warning(
                "These flags only apply to `--display-mode rerun` and are ignored with "
                "`--display-mode foxglove`: %s.",
                ", ".join(f"--{name.replace('_', '-')}" for name in ignored),
            )

    kwargs = vars(args)
    repo_id = infer_repo_id(kwargs.pop("repo_id"), kwargs.get("root"))
    root = resolve_dataset_root(kwargs.pop("root"))
    tolerance_s = kwargs.pop("tolerance_s")
    list_episodes = kwargs.pop("list_episodes")

    init_logging()
    if list_episodes:
        logging.info("Loading dataset metadata")
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        print(format_episode_table(meta))
        return

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **kwargs)


if __name__ == "__main__":
    main()
