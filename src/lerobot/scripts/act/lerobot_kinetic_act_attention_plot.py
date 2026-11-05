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
"""Save kinetic_act attention heatmaps with the predicted stage and distance on each keyframe.

The keyframes are the same frames `lerobot-attention-plot` samples. The caption is the
prediction for that frame: chunk step 0 of the stage and object-distance heads. The dataset
label for the same frame is shown underneath.

Requires: pip install 'lerobot[dataset_viz]'

Example:
```
python -m lerobot.scripts.act.lerobot_kinetic_act_attention_plot \
    --dataset-path /path/to/dataset \
    --policy-path /path/to/checkpoints/last/pretrained_model \
    --episodes 0 1 2
```
"""

import argparse
import logging
import os

import numpy as np
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.kinetic_act.modeling_kinetic_act import KineticACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.scripts.act.lerobot_attention_plot import (
    FIGURE_BASE_HEIGHT_INCHES,
    FIGURE_STRIP_HEIGHT_INCHES,
    FIGURE_WIDTH_INCHES,
    AttentionRecorder,
    _import_pyplot,
    draw_signal_axes,
    episode_row_indices,
    frame_image_to_uint8,
    get_feature_names,
    load_act_policy,
    load_dataset,
    overlay_attention,
    read_episode_signals,
    resolve_episode_indices_to_plot,
    resolve_feature_keys,
    resolve_obs_plot_groups,
    resolve_policy_image_keys,
    rollout_to_camera_maps,
    select_midpoint_timestamps,
    snap_to_frame_indices,
    split_episode_timestamps,
)
from lerobot.utils.constants import OBS_PREFIX
from lerobot.utils.utils import init_logging

# observation.stage written by kinetic_label_object_task: 0 approach, 1 grasp, 2 pull.
STAGE_LABELS = ("approach", "grasp", "pull")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Local dataset folder path.")
    parser.add_argument(
        "--policy-path", type=str, required=True, help="Local kinetic_act checkpoint directory."
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None, help="Episode indices. Default: all."
    )
    parser.add_argument(
        "--feature-keys",
        type=str,
        nargs="+",
        default=None,
        help="1-D feature keys for the time plots. Default: observation.state action.",
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
        help="Policy camera keys to overlay. Default: every camera in the checkpoint.",
    )
    parser.add_argument("--num-image-samples", type=int, default=18, help="Keyframes per episode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for `episode_<i>.png`. Default: `<dataset-path>/kinetic_act_attention_plots/`.",
    )
    parser.add_argument("--dpi", type=int, default=130, help="Saved figure resolution.")
    parser.add_argument(
        "--tolerance-s", type=float, default=1e-4, help="Timestamp tolerance for the dataset."
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: checkpoint device.")
    parser.add_argument("--heatmap-alpha", type=float, default=0.5, help="Heatmap blend weight.")
    args = parser.parse_args()

    init_logging()
    dataset = load_dataset(args.dataset_path, args.episodes, args.tolerance_s)
    feature_keys = resolve_feature_keys(dataset, args.feature_keys)
    obs_plot_groups = resolve_obs_plot_groups(args.obs_plot_groups)
    episode_indices = resolve_episode_indices_to_plot(dataset, args.episodes)
    policy, preprocessor = load_act_policy(args.policy_path, dataset.meta, args.device)
    policy = require_kinetic_act_policy(policy)
    image_observation_keys = resolve_policy_image_keys(dataset, policy, args.image_observations)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(str(dataset.root), "kinetic_act_attention_plots")
    )

    for episode_index in episode_indices:
        output_path = plot_episode(
            dataset,
            policy,
            preprocessor,
            episode_index,
            image_observation_keys,
            feature_keys,
            obs_plot_groups,
            args.num_image_samples,
            output_dir,
            args.dpi,
            args.heatmap_alpha,
        )
        logging.info("Saved %s", output_path)


def require_kinetic_act_policy(policy: ACTPolicy) -> KineticACTPolicy:
    """Require a kinetic_act checkpoint that predicts stage and distance.

    Args:
        policy (ACTPolicy): Loaded policy.

    Returns:
        KineticACTPolicy: Policy with both heads.
    """
    if not isinstance(policy, KineticACTPolicy):
        raise TypeError(f"Policy is {type(policy).__name__}, expected KineticACTPolicy.")
    missing_heads = [
        name
        for name, key in (
            ("object distance", policy.config.object_distance_label_key),
            ("stage", policy.config.stage_label_key),
        )
        if key is None
    ]
    if missing_heads:
        raise ValueError(f"Checkpoint is missing the {missing_heads} head.")
    return policy


def plot_episode(
    dataset,
    policy: KineticACTPolicy,
    preprocessor: PolicyProcessorPipeline,
    episode_index: int,
    image_observation_keys: list[str],
    feature_keys: list[str],
    obs_plot_groups: list[str],
    num_image_samples: int,
    output_dir: str,
    dpi: int,
    heatmap_alpha: float,
) -> str:
    """Save one episode figure with heatmaps and the prediction for each keyframe.

    Args:
        dataset: Dataset.
        policy (KineticACTPolicy): kinetic_act policy.
        preprocessor (PolicyProcessorPipeline): Policy preprocessor.
        episode_index (int): Episode index.
        image_observation_keys (list[str]): Camera keys.
        feature_keys (list[str]): Time-plot feature keys.
        obs_plot_groups (list[str]): Dimension groups.
        num_image_samples (int): Keyframes per episode.
        output_dir (str): Output directory.
        dpi (int): Figure resolution.
        heatmap_alpha (float): Heatmap blend weight.

    Returns:
        str: Saved figure path.
    """
    timestamps, signals = read_episode_signals(dataset, episode_index, feature_keys)
    split_timestamps = split_episode_timestamps(timestamps, num_image_samples)
    keyframe_indices = snap_to_frame_indices(select_midpoint_timestamps(split_timestamps), timestamps)
    keyframe_timestamps = timestamps[keyframe_indices]
    dataset_indices = episode_row_indices(dataset, episode_index)[keyframe_indices]

    overlaid: dict[str, list[np.ndarray]] = {key: [] for key in image_observation_keys}
    captions: list[str] = []
    for dataset_index in dataset_indices:
        frame = dataset[int(dataset_index)]
        attention_maps, stage, distance = predict_frame(policy, preprocessor, frame)
        captions.append(format_frame_caption(policy, frame, stage, distance))
        for key in image_observation_keys:
            image_uint8 = frame_image_to_uint8(frame[key])
            overlaid[key].append(overlay_attention(image_uint8, attention_maps[key], heatmap_alpha))

    signal_names = {key: get_feature_names(dataset, key) for key in feature_keys}
    figure = draw_episode(
        dataset.repo_id,
        episode_index,
        timestamps,
        signals,
        signal_names,
        feature_keys,
        obs_plot_groups,
        split_timestamps,
        keyframe_timestamps,
        overlaid,
        captions,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"episode_{episode_index}.png")
    figure.savefig(output_path, dpi=dpi)
    _import_pyplot().close(figure)
    return output_path


def predict_frame(
    policy: KineticACTPolicy, preprocessor: PolicyProcessorPipeline, frame: dict
) -> tuple[dict[str, np.ndarray], int, float]:
    """Run one frame and return attention maps plus the stage and distance for that frame.

    Args:
        policy (KineticACTPolicy): kinetic_act policy.
        preprocessor (PolicyProcessorPipeline): Policy preprocessor.
        frame (dict): One dataset frame.

    Returns:
        tuple[dict[str, np.ndarray], int, float]: Attention maps, stage class, distance in meters.
    """
    observation = {key: value for key, value in frame.items() if key.startswith(OBS_PREFIX)}
    observation["task"] = frame["task"]
    batch = preprocessor(observation)
    with AttentionRecorder(policy.model) as recorder:
        _, auxiliary_outputs = policy.predict_action_chunk_with_auxiliary_outputs(batch)
    # Chunk step 0 is the prediction aligned with this frame.
    stage = int(auxiliary_outputs["stage_logits"][0, 0].argmax(dim=-1).detach().cpu())
    distance = float(auxiliary_outputs["object_distance"][0, 0].detach().cpu())
    return rollout_to_camera_maps(recorder, policy.config), stage, distance


def format_frame_caption(policy: KineticACTPolicy, frame: dict, stage: int, distance: float) -> str:
    """Format the predicted and recorded stage and distance for one keyframe.

    Args:
        policy (KineticACTPolicy): kinetic_act policy.
        frame (dict): Dataset frame.
        stage (int): Predicted stage class.
        distance (float): Predicted distance in meters.

    Returns:
        str: Two-line caption.
    """
    label_stage = scalar_from_frame(frame[policy.config.stage_label_key])
    label_distance = scalar_from_frame(frame[policy.config.object_distance_label_key])
    label_distance_text = "—" if not np.isfinite(label_distance) else f"{label_distance:.2f}m"
    return (
        f"pred {stage_label_string(stage)} {distance:.2f}m\n"
        f"label {stage_label_string(label_stage)} {label_distance_text}"
    )


def stage_label_string(stage: float) -> str:
    """Map a stage class to its name.

    Args:
        stage (float): Stage class. Negative marks a missing label.

    Returns:
        str: Stage name, or the class number when it has no name.
    """
    if not np.isfinite(stage) or stage < 0:
        return "—"
    index = int(stage)
    if 0 <= index < len(STAGE_LABELS):
        return STAGE_LABELS[index]
    return str(index)


def scalar_from_frame(value: Tensor | np.ndarray | float) -> float:
    """Read a one-element label from a dataset frame.

    Args:
        value (Tensor | np.ndarray | float): Frame value.

    Returns:
        float: Scalar label.
    """
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def draw_episode(
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
    captions: list[str],
):
    """Draw heatmaps with a prediction caption on the first camera strip.

    Args:
        repo_id (str): Dataset name.
        episode_index (int): Episode index.
        timestamps (np.ndarray): Frame timestamps.
        signals (dict[str, np.ndarray]): Time-plot signals.
        signal_names (dict[str, list[str]]): Dimension names.
        feature_keys (list[str]): Time-plot keys.
        obs_plot_groups (list[str]): Dimension groups.
        split_timestamps (np.ndarray): Interval boundaries.
        keyframe_timestamps (np.ndarray): Keyframe timestamps.
        keyframe_images (dict[str, list[np.ndarray]]): Overlaid images per camera.
        captions (list[str]): Caption per keyframe.

    Returns:
        Figure: Episode figure.
    """
    plt = _import_pyplot()
    image_keys = list(keyframe_images)
    figure_height = FIGURE_BASE_HEIGHT_INCHES + FIGURE_STRIP_HEIGHT_INCHES * max(len(image_keys) - 1, 0)
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, figure_height))
    outer_grid = figure.add_gridspec(
        len(image_keys) + 1,
        1,
        height_ratios=[2.0] + [1.4] * max(len(image_keys) - 1, 0) + [5.0],
        hspace=0.28,
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
            captions if row == 0 else None,
            image_key,
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
        f"{repo_id}   kinetic_act   episode {episode_index}   {len(timestamps)} frames",
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


def draw_keyframe_strip(figure, grid_cell, images, keyframe_timestamps, captions, image_key) -> None:
    """Draw one camera strip, with prediction captions when provided.

    Args:
        figure: Figure.
        grid_cell: Gridspec cell.
        images (list[np.ndarray]): Images.
        keyframe_timestamps (np.ndarray): Keyframe timestamps.
        captions (list[str] | None): Caption per keyframe. Drawn on this strip when set.
        image_key (str): Camera key.
    """
    strip_grid = grid_cell.subgridspec(1, len(images), wspace=0.03)
    for position, (image, keyframe_timestamp) in enumerate(zip(images, keyframe_timestamps, strict=True)):
        axis = figure.add_subplot(strip_grid[0, position])
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"t={keyframe_timestamp:.1f}s", fontsize=7, pad=2)
        if captions is not None:
            axis.set_xlabel(captions[position], fontsize=6.5, labelpad=1)
        if position == 0:
            axis.set_ylabel(image_key, fontsize=7)
        for spine in axis.spines.values():
            spine.set_color("0.4")


if __name__ == "__main__":
    main()
