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
"""Save a static episode overview: camera keyframe strips above synced 1-D signal plots.

Requires: pip install 'lerobot[dataset_viz]'

Example:
```
lerobot-dataset-plot \
    --dataset-path /path/to/my_dataset \
    --feature-keys observation.state action
```
"""

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _matplotlib_available, require_package
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

FIGURE_WIDTH_INCHES = 24.0
FIGURE_BASE_HEIGHT_INCHES = 12.0
FIGURE_STRIP_HEIGHT_INCHES = 2.2
NON_PLOTTABLE_DTYPES = ("video", "image")
DEFAULT_FEATURE_KEYS = (OBS_STATE, ACTION)
OBS_PLOT_GROUPS = ("position", "velocity", "effort")
_pyplot = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Local dataset folder path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episode indices to plot. Default: all.",
    )
    parser.add_argument(
        "--feature-keys",
        type=str,
        nargs="+",
        default=None,
        help="1-D feature keys (observation or action). Default: observation.state action.",
    )
    parser.add_argument(
        "--obs-plot-groups",
        type=str,
        nargs="+",
        default=None,
        help="Dimension groups to plot. Default: position velocity effort.",
    )
    parser.add_argument(
        "--image-observations",
        type=str,
        nargs="+",
        default=None,
        help="Image observation keys. Default: all RGB video cameras.",
    )
    parser.add_argument(
        "--num-image-samples",
        type=int,
        default=18,
        help="Number of images sampled per episode. Must be >= 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for `episode_<i>.png` files. Default: `<dataset-path>/episode_plots/`.",
    )
    parser.add_argument("--dpi", type=int, default=130, help="Saved figure resolution.")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Timestamp tolerance forwarded to LeRobotDataset and video decoding.",
    )
    args = parser.parse_args()

    init_logging()
    dataset = load_dataset(args.dataset_path, args.episodes, args.tolerance_s)
    image_observation_keys = resolve_image_observation_keys(dataset, args.image_observations)
    feature_keys = resolve_feature_keys(dataset, args.feature_keys)
    obs_plot_groups = resolve_obs_plot_groups(args.obs_plot_groups)
    episode_indices = resolve_episode_indices_to_plot(dataset, args.episodes)
    output_dir = (
        args.output_dir if args.output_dir is not None else os.path.join(str(dataset.root), "episode_plots")
    )

    for episode_index in episode_indices:
        output_path = plot_episode(
            dataset,
            episode_index,
            image_observation_keys,
            feature_keys,
            obs_plot_groups,
            args.num_image_samples,
            output_dir,
            args.dpi,
            args.tolerance_s,
        )
        logging.info("Saved %s", output_path)


def load_dataset(dataset_path: str, episodes: list[int] | None, tolerance_s: float) -> LeRobotDataset:
    """Load a local LeRobot dataset.

    Args:
        dataset_path (str): Local dataset folder.
        episodes (list[int] | None): Episode indices.
        tolerance_s (float): Timestamp tolerance.

    Returns:
        LeRobotDataset: Loaded dataset.
    """
    root = resolve_dataset_root(dataset_path)
    repo_id = os.path.basename(root)
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    validate_episode_indices(episodes, metadata.total_episodes)
    return LeRobotDataset(repo_id, root=root, episodes=episodes, tolerance_s=tolerance_s)


def resolve_dataset_root(root: str | None) -> str | None:
    """Expand a local dataset path and require ``meta/info.json`` when set.

    Args:
        root (str | None): Dataset folder.

    Returns:
        str | None: Resolved folder.
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
    """Return per-dimension feature names.

    Args:
        dataset (LeRobotDataset): Dataset.
        key (str): Feature key.

    Returns:
        list[str]: Dimension names.
    """
    feature = dataset.features[key]
    dim = feature["shape"][-1]
    names = feature.get("names")
    if isinstance(names, list) and len(names) == dim:
        return [str(name) for name in names]
    return [f"{key}_{d}" for d in range(dim)]


def group_feature_dims(
    names: list[str], plot_groups: list[str] | None = None
) -> list[tuple[str, list[int], list[str]]]:
    """Split feature dimensions by trailing name suffix.

    Args:
        names (list[str]): Dimension names.
        plot_groups (list[str] | None): Groups to keep. Default: ``OBS_PLOT_GROUPS``.

    Returns:
        list[tuple[str, list[int], list[str]]]: Group name, indices, and series names.
    """
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        suffix = name.rsplit(".", 1)[-1] if "." in name else ""
        groups.setdefault(suffix, []).append(i)
    requested = list(OBS_PLOT_GROUPS if plot_groups is None else plot_groups)
    selected = [key for key in requested if key in groups]
    ordered = selected if selected else list(groups)
    return [(key, groups[key], [names[i] for i in groups[key]]) for key in ordered]


def validate_episode_indices(episodes: list[int] | None, total_episodes: int) -> None:
    """Validate episode indices.

    Args:
        episodes (list[int] | None): Episode indices.
        total_episodes (int): Episode count.
    """
    if episodes is None:
        return
    out_of_range = [index for index in episodes if index < 0 or index >= total_episodes]
    if out_of_range:
        raise ValueError(f"Episode indices out of range [0, {total_episodes}): {out_of_range}")


def resolve_episode_indices_to_plot(dataset: LeRobotDataset, episodes: list[int] | None) -> list[int]:
    """Resolve episode indices to plot.

    Args:
        dataset (LeRobotDataset): Dataset.
        episodes (list[int] | None): Episode indices.

    Returns:
        list[int]: Episode indices.
    """
    if episodes is None:
        return list(range(dataset.meta.total_episodes))
    return list(episodes)


def resolve_image_observation_keys(
    dataset: LeRobotDataset, image_observations: list[str] | None
) -> list[str]:
    """Resolve RGB video feature keys.

    Args:
        dataset (LeRobotDataset): Dataset.
        image_observations (list[str] | None): Image feature keys.

    Returns:
        list[str]: Image feature keys.
    """
    available = [
        key
        for key in dataset.meta.camera_keys
        if key in dataset.meta.video_keys and key not in dataset.meta.depth_keys
    ]
    if image_observations is None:
        if not available:
            raise ValueError(
                "No RGB video cameras in this dataset. Pass --image-observations with RGB video feature keys."
            )
        return available

    invalid = [key for key in image_observations if key not in available]
    if invalid:
        raise ValueError(
            f"Image observation keys are not RGB video features: {invalid}. "
            f"Available RGB video keys: {available}"
        )
    return list(image_observations)


def is_plottable_vector(feature: dict) -> bool:
    """Return whether a feature is a 1-D non-image vector.

    Args:
        feature (dict): Feature spec.

    Returns:
        bool: Plottable flag.
    """
    return feature["dtype"] not in NON_PLOTTABLE_DTYPES and len(feature["shape"]) == 1


def resolve_feature_keys(dataset: LeRobotDataset, feature_keys: list[str] | None) -> list[str]:
    """Resolve 1-D observation or action feature keys.

    Args:
        dataset (LeRobotDataset): Dataset.
        feature_keys (list[str] | None): Feature keys.

    Returns:
        list[str]: Feature keys.
    """
    if feature_keys is None:
        available = [
            key
            for key in DEFAULT_FEATURE_KEYS
            if key in dataset.features and is_plottable_vector(dataset.features[key])
        ]
        if not available:
            raise ValueError(
                f"Default feature keys {list(DEFAULT_FEATURE_KEYS)} are missing or not 1-D vectors. "
                "Pass --feature-keys with observation or action keys."
            )
        return available
    validate_feature_keys(dataset, feature_keys)
    return list(feature_keys)


def resolve_obs_plot_groups(obs_plot_groups: list[str] | None) -> list[str]:
    """Resolve dimension groups to plot.

    Args:
        obs_plot_groups (list[str] | None): Group names.

    Returns:
        list[str]: Group names.
    """
    if obs_plot_groups is None:
        return list(OBS_PLOT_GROUPS)
    return list(obs_plot_groups)


def validate_feature_keys(dataset: LeRobotDataset, feature_keys: list[str]) -> None:
    """Validate 1-D vector feature keys.

    Args:
        dataset (LeRobotDataset): Dataset.
        feature_keys (list[str]): Feature keys.
    """
    plottable = [key for key, feature in dataset.features.items() if is_plottable_vector(feature)]
    invalid = [
        key
        for key in feature_keys
        if key not in dataset.features or not is_plottable_vector(dataset.features[key])
    ]
    if invalid:
        raise ValueError(
            f"Feature keys are missing or not 1-D vector features: {invalid}. "
            f"Plottable keys: {plottable}"
        )


def plot_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    image_observation_keys: list[str],
    feature_keys: list[str],
    obs_plot_groups: list[str],
    num_image_samples: int,
    output_dir: str,
    dpi: int,
    tolerance_s: float,
) -> str:
    """Save one episode overview figure.

    Args:
        dataset (LeRobotDataset): Dataset.
        episode_index (int): Episode index.
        image_observation_keys (list[str]): Image feature keys.
        feature_keys (list[str]): Feature keys.
        obs_plot_groups (list[str]): Dimension groups to plot.
        num_image_samples (int): Image count per strip.
        output_dir (str): Output directory.
        dpi (int): Figure resolution.
        tolerance_s (float): Timestamp tolerance.

    Returns:
        str: Saved figure path.
    """
    timestamps, signals = read_episode_signals(dataset, episode_index, feature_keys)
    split_timestamps = split_episode_timestamps(timestamps, num_image_samples)
    midpoint_timestamps = select_midpoint_timestamps(split_timestamps)
    keyframe_timestamps = snap_to_frame_timestamps(midpoint_timestamps, timestamps)
    keyframe_images = {
        key: decode_keyframes(dataset, episode_index, key, keyframe_timestamps, tolerance_s)
        for key in image_observation_keys
    }
    signal_names = {key: get_feature_names(dataset, key) for key in feature_keys}
    figure = draw_episode_overview(
        dataset.repo_id,
        episode_index,
        timestamps,
        signals,
        signal_names,
        feature_keys,
        obs_plot_groups,
        split_timestamps,
        keyframe_timestamps,
        keyframe_images,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"episode_{episode_index}.png")
    figure.savefig(output_path, dpi=dpi)
    _import_pyplot().close(figure)
    return output_path


def read_episode_signals(
    dataset: LeRobotDataset, episode_index: int, feature_keys: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read episode timestamps and 1-D signals.

    Args:
        dataset (LeRobotDataset): Dataset.
        episode_index (int): Episode index.
        feature_keys (list[str]): Feature keys.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: Timestamps and signals.
    """
    hf = dataset.hf_dataset.with_format("numpy")
    episode_indices = np.asarray(hf["episode_index"]).reshape(-1)
    mask = episode_indices == episode_index
    timestamps = np.asarray(hf["timestamp"], dtype=np.float64).reshape(-1)[mask]
    if timestamps.size == 0:
        raise ValueError(f"Episode {episode_index} has no frames")
    timestamps = timestamps - timestamps[0]
    signals = {}
    for key in feature_keys:
        values = np.asarray(hf[key], dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        signals[key] = values[mask]
    return timestamps, signals


def split_episode_timestamps(timestamps: np.ndarray, num_image_samples: int) -> np.ndarray:
    """Split episode timestamps.

    Args:
        timestamps (np.ndarray): Frame timestamps.
        num_image_samples (int): Image count.

    Returns:
        np.ndarray: Split timestamps.
    """
    if num_image_samples < 1:
        raise ValueError("--num-image-samples must be at least 1")
    return np.linspace(timestamps[0], timestamps[-1], num_image_samples + 1)


def select_midpoint_timestamps(split_timestamps: np.ndarray) -> np.ndarray:
    """Select midpoint timestamps.

    Args:
        split_timestamps (np.ndarray): Split timestamps.

    Returns:
        np.ndarray: Midpoint timestamps.
    """
    return (split_timestamps[:-1] + split_timestamps[1:]) / 2.0


def snap_to_frame_timestamps(target_timestamps: np.ndarray, frame_timestamps: np.ndarray) -> np.ndarray:
    """Snap targets to frame timestamps.

    Args:
        target_timestamps (np.ndarray): Target timestamps.
        frame_timestamps (np.ndarray): Frame timestamps.

    Returns:
        np.ndarray: Snapped timestamps.
    """
    if frame_timestamps.size == 0:
        raise ValueError("Cannot snap timestamps: episode has no frames")
    distances = np.abs(frame_timestamps[:, None] - target_timestamps[None, :])
    return frame_timestamps[distances.argmin(axis=0)]


def decode_keyframes(
    dataset: LeRobotDataset,
    episode_index: int,
    image_observation_key: str,
    keyframe_timestamps: np.ndarray,
    tolerance_s: float,
) -> list[np.ndarray]:
    """Decode keyframe images from a video feature.

    Args:
        dataset (LeRobotDataset): Dataset.
        episode_index (int): Episode index.
        image_observation_key (str): Image feature key.
        keyframe_timestamps (np.ndarray): Keyframe timestamps.
        tolerance_s (float): Timestamp tolerance.

    Returns:
        list[np.ndarray]: Images.
    """
    episodes = dataset.meta.episodes
    if episodes is None:
        raise FileNotFoundError(f"Dataset '{dataset.repo_id}' has no episode metadata")
    from_timestamp = float(episodes[episode_index][f"videos/{image_observation_key}/from_timestamp"])
    video_path = os.path.join(str(dataset.root), str(dataset.meta.get_video_file_path(episode_index, image_observation_key)))
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found for {image_observation_key}: {video_path}")
    video_timestamps = [from_timestamp + float(timestamp) for timestamp in keyframe_timestamps]
    frames = decode_video_frames(video_path, video_timestamps, tolerance_s, return_uint8=True)
    return [frame.permute(1, 2, 0).cpu().numpy() for frame in frames]


def _import_pyplot():
    """Return the pyplot module.

    Returns:
        module: Pyplot.
    """
    global _pyplot
    if _pyplot is not None:
        return _pyplot
    if not _matplotlib_available:
        require_package("matplotlib", extra="dataset_viz")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _pyplot = plt
    return plt


def draw_episode_overview(
    repo_id: str,
    episode_index: int,
    timestamps: np.ndarray,
    signals: dict[str, np.ndarray],
    signal_names: dict[str, list[str]],
    feature_keys: list[str],
    obs_plot_groups: list[str],
    split_timestamps: np.ndarray,
    keyframe_timestamps: np.ndarray,
    keyframe_images: dict[str, list[np.ndarray]],
) -> "Figure":
    """Draw the episode overview figure.

    Args:
        repo_id (str): Dataset name.
        episode_index (int): Episode index.
        timestamps (np.ndarray): Frame timestamps.
        signals (dict[str, np.ndarray]): State signals.
        signal_names (dict[str, list[str]]): Per-key dimension names.
        feature_keys (list[str]): Feature keys.
        obs_plot_groups (list[str]): Dimension groups to plot.
        split_timestamps (np.ndarray): Split timestamps.
        keyframe_timestamps (np.ndarray): Keyframe timestamps.
        keyframe_images (dict[str, list[np.ndarray]]): Per-key images.

    Returns:
        Figure: Overview figure.
    """
    plt = _import_pyplot()
    image_keys = list(keyframe_images)
    figure_height = FIGURE_BASE_HEIGHT_INCHES + FIGURE_STRIP_HEIGHT_INCHES * max(len(image_keys) - 1, 0)
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, figure_height))
    outer_grid = figure.add_gridspec(
        len(image_keys) + 1,
        1,
        height_ratios=[1.4] * len(image_keys) + [5.0],
        hspace=0.12,
        top=0.93,
        bottom=0.06,
        left=0.06,
        right=0.98,
    )

    for row, image_key in enumerate(image_keys):
        draw_keyframe_strip(
            figure,
            outer_grid[row],
            keyframe_images[image_key],
            keyframe_timestamps,
            image_key,
            show_titles=row == 0,
        )
    signal_axes = draw_signal_axes(
        figure,
        outer_grid[-1],
        timestamps,
        signals,
        signal_names,
        feature_keys,
        obs_plot_groups,
        split_timestamps,
    )

    figure.suptitle(
        f"{repo_id}   episode {episode_index}   {len(timestamps)} frames, {timestamps[-1]:.1f}s",
        fontsize=14,
    )
    legend_labels = signal_axes[0].get_legend_handles_labels()[1]
    signal_axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=max(len(legend_labels), 1),
        fontsize=8,
        frameon=False,
    )
    return figure


def draw_keyframe_strip(
    figure: "Figure",
    grid_cell,
    images: list[np.ndarray],
    keyframe_timestamps: np.ndarray,
    image_key: str,
    show_titles: bool,
) -> None:
    """Draw a camera keyframe strip.

    Args:
        figure (Figure): Figure.
        grid_cell: Gridspec cell.
        images (list[np.ndarray]): Images.
        keyframe_timestamps (np.ndarray): Keyframe timestamps.
        image_key (str): Image feature key.
        show_titles (bool): Title flag.
    """
    strip_grid = grid_cell.subgridspec(1, len(images), wspace=0.03)
    for position, (image, keyframe_timestamp) in enumerate(zip(images, keyframe_timestamps, strict=True)):
        axis = figure.add_subplot(strip_grid[0, position])
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        if show_titles:
            axis.set_title(f"{position + 1}   t={keyframe_timestamp:.1f}s", fontsize=8)
        if position == 0:
            axis.set_ylabel(image_key, fontsize=8)
        for spine in axis.spines.values():
            spine.set_color("0.4")


def draw_signal_axes(
    figure: "Figure",
    grid_cell,
    timestamps: np.ndarray,
    signals: dict[str, np.ndarray],
    signal_names: dict[str, list[str]],
    feature_keys: list[str],
    obs_plot_groups: list[str],
    split_timestamps: np.ndarray,
) -> list["Axes"]:
    """Draw state-signal axes.

    Args:
        figure (Figure): Figure.
        grid_cell: Gridspec cell.
        timestamps (np.ndarray): Frame timestamps.
        signals (dict[str, np.ndarray]): State signals.
        signal_names (dict[str, list[str]]): Per-key dimension names.
        feature_keys (list[str]): Feature keys.
        obs_plot_groups (list[str]): Dimension groups to plot.
        split_timestamps (np.ndarray): Split timestamps.

    Returns:
        list[Axes]: Signal axes.
    """
    plt = _import_pyplot()
    signal_rows: list[tuple[str, str, list[int], list[str]]] = []
    for key in feature_keys:
        for group_name, dim_indices, series_names in group_feature_dims(
            signal_names[key], obs_plot_groups
        ):
            signal_rows.append((key, group_name, dim_indices, series_names))

    signal_grid = grid_cell.subgridspec(len(signal_rows), 1, hspace=0.08)
    keyframe_timestamps = select_midpoint_timestamps(split_timestamps)
    axes: list[Axes] = []
    for row, (key, group_name, dim_indices, series_names) in enumerate(signal_rows):
        joint_colors = plt.get_cmap("tab10")(np.arange(len(series_names)) % 10)
        axis = figure.add_subplot(signal_grid[row, 0], sharex=axes[0] if axes else None)
        for joint_index, series_name in enumerate(series_names):
            label = series_name.rsplit(".", 1)[0] if group_name else series_name
            axis.plot(
                timestamps,
                signals[key][:, dim_indices[joint_index]],
                lw=1.0,
                color=joint_colors[joint_index],
                label=label if row == 0 else None,
            )
        draw_interval_markers(axis, split_timestamps, keyframe_timestamps, annotate=row == 0)
        axis.set_ylabel(f"{key}/{group_name}" if group_name else key, fontsize=9)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=8)
        if row < len(signal_rows) - 1:
            axis.tick_params(labelbottom=False)
        axes.append(axis)

    axes[-1].set_xlabel("time (s)", fontsize=10)
    axes[-1].set_xlim(timestamps[0], timestamps[-1])
    return axes


def draw_interval_markers(
    axis: "Axes",
    split_timestamps: np.ndarray,
    keyframe_timestamps: np.ndarray,
    annotate: bool,
) -> None:
    """Draw interval boundary markers.

    Args:
        axis (Axes): Axes.
        split_timestamps (np.ndarray): Split timestamps.
        keyframe_timestamps (np.ndarray): Keyframe timestamps.
        annotate (bool): Annotation flag.
    """
    for split_timestamp in split_timestamps:
        axis.axvline(split_timestamp, color="0.35", lw=0.8, ls="--", alpha=0.7)
    if not annotate:
        return
    for position, keyframe_timestamp in enumerate(keyframe_timestamps):
        axis.annotate(
            str(position + 1),
            xy=(keyframe_timestamp, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="0.25",
        )


if __name__ == "__main__":
    main()
