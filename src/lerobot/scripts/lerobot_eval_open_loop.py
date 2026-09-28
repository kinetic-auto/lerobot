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
"""Open-loop evaluation of a trained policy on recorded dataset episodes.

Every frame of the selected episodes is replayed through the checkpoint's preprocessor,
`select_action` and postprocessor, and the resulting absolute action is compared with the
recorded one. Episodes are drawn from the `split_info.json` that training writes next to the
checkpoint, so validation and test episodes are evaluated without any manual bookkeeping.

Outputs under `--output_dir` (default `<checkpoint step dir>/eval_open_loop`):
- `metrics_summary.json`: metrics averaged overall and per split
- `metrics_per_episode.csv`: one row per episode, with a `mae_<joint>` column per joint
- `plots/episode_<index>_<split>.png`: recorded, predicted and observed trajectories per joint
- `plots/per_joint_mae_by_split.png`: distribution of per-joint MAE across episodes, per split

Examples:
```
lerobot-eval-open-loop \
    --policy.path=model_zoo/my_dataset/act_20260927_120000/checkpoints/last/pretrained_model \
    --dataset.root=/path/to/my_dataset \
    --split=val

lerobot-eval-open-loop \
    --policy.path=<checkpoint>/pretrained_model \
    --dataset.root=/path/to/my_dataset \
    --dataset.episodes='[3, 7]' \
    --policy.n_action_steps=1
```
"""

import csv
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.configs import parser
from lerobot.configs.eval import OpenLoopEvalConfig
from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.joint_layout import channel_groups
from lerobot.datasets.utils import EpisodeSplit, resolve_excluded_features
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.relative_action_processor import bind_relative_anchor
from lerobot.scripts.lerobot_dataset_plot import _import_pyplot, get_feature_names
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.constants import ACTION, OBS_STATE, SPLIT_INFO
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig

MANUAL_SPLIT_LABEL = "manual"
UNSPLIT_LABEL = "all"
PLOTS_DIR = "plots"
METRICS_SUMMARY_FILENAME = "metrics_summary.json"
METRICS_PER_EPISODE_FILENAME = "metrics_per_episode.csv"
PER_JOINT_MAE_PLOT_FILENAME = "per_joint_mae_by_split.png"
COSINE_EPSILON = 1e-8


def main() -> None:
    register_third_party_plugins()
    evaluate_open_loop()


@parser.wrap()
def evaluate_open_loop(cfg: OpenLoopEvalConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    dataset_metadata = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    excluded_features = resolve_excluded_features(cfg.dataset.exclude_features, dataset_metadata.features)
    dataset_metadata.exclude_features(excluded_features)

    policy, preprocessor, postprocessor = load_policy_and_processors(cfg, dataset_metadata)

    episode_split = load_episode_split(cfg.policy.pretrained_path)
    episodes_to_evaluate = select_episodes_to_evaluate(cfg, episode_split, dataset_metadata.total_episodes)
    logging.info(f"Evaluating {len(episodes_to_evaluate)} episodes: {episodes_to_evaluate}")

    dataset = load_replay_dataset(
        cfg, policy.config, dataset_metadata, excluded_features, episodes_to_evaluate
    )
    joint_names = get_feature_names(dataset, ACTION)
    state_key = dataset_key_for(cfg.rename_map, OBS_STATE)
    state_names = get_feature_names(dataset, state_key) if state_key in dataset.features else []

    all_metrics: list[EpisodeMetrics] = []
    for episode_index, split_label in episodes_to_evaluate:
        replay = replay_episode(
            policy,
            preprocessor,
            postprocessor,
            dataset,
            episode_index,
            split_label,
            state_key,
            cfg.rename_map,
            cfg.dataset.task_override,
            excluded_features,
        )
        metrics = compute_episode_metrics(replay, joint_names)
        all_metrics.append(metrics)
        if cfg.save_plots:
            plot_path = output_dir / PLOTS_DIR / f"episode_{episode_index:04d}_{split_label}.png"
            plot_episode_joints(replay, metrics, joint_names, state_names, plot_path, cfg.dpi)
        logging.info(
            f"episode {episode_index} [{split_label}]: mae={metrics.mae:.4f} rmse={metrics.rmse:.4f}"
        )

    write_metrics_summary(all_metrics, output_dir / METRICS_SUMMARY_FILENAME)
    write_metrics_per_episode(all_metrics, output_dir / METRICS_PER_EPISODE_FILENAME)
    if cfg.save_plots:
        plot_per_joint_mae_by_split(
            all_metrics, joint_names, output_dir / PLOTS_DIR / PER_JOINT_MAE_PLOT_FILENAME, cfg.dpi
        )
    log_summary(all_metrics)


def load_policy_and_processors(
    cfg: OpenLoopEvalConfig, dataset_metadata: LeRobotDatasetMetadata
) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load the policy and its processors from the checkpoint."""
    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset_metadata, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    bind_relative_anchor(policy, preprocessor)
    return policy, preprocessor, postprocessor


def load_episode_split(pretrained_path: str | Path | None) -> EpisodeSplit | None:
    if pretrained_path is None:
        return None
    split_path = Path(pretrained_path) / SPLIT_INFO
    if not split_path.is_file():
        logging.info(f"No {SPLIT_INFO} next to the checkpoint; every episode is labelled '{UNSPLIT_LABEL}'.")
        return None
    logging.info(f"Loaded episode split from {split_path}")
    return EpisodeSplit.from_dict(load_json(split_path))


def select_episodes_to_evaluate(
    cfg: OpenLoopEvalConfig, episode_split: EpisodeSplit | None, total_episodes: int
) -> list[tuple[int, str]]:
    """Select the `(episode_index, split_label)` pairs to evaluate."""
    if cfg.dataset.episodes is not None:
        out_of_range = [index for index in cfg.dataset.episodes if not 0 <= index < total_episodes]
        if out_of_range:
            raise ValueError(
                f"Episode indices outside the dataset range [0, {total_episodes}): {out_of_range}"
            )
        return sorted((index, split_label_of(index, episode_split)) for index in cfg.dataset.episodes)

    if episode_split is None:
        if cfg.split != UNSPLIT_LABEL:
            raise ValueError(f"split={cfg.split!r} needs a {SPLIT_INFO} next to the checkpoint.")
        candidates = {UNSPLIT_LABEL: list(range(total_episodes))}
    else:
        candidates = {
            "train": episodes_within_dataset(episode_split.train_episodes, total_episodes),
            "val": episodes_within_dataset(episode_split.val_episodes, total_episodes),
            "test": episodes_within_dataset(episode_split.test_episodes, total_episodes),
        }
        if cfg.split != UNSPLIT_LABEL:
            candidates = {cfg.split: candidates[cfg.split]}

    sampler = random.Random(cfg.seed)
    selected: list[tuple[int, str]] = []
    for split_label, episodes in candidates.items():
        if not episodes:
            continue
        count = len(episodes) if cfg.episodes_per_split <= 0 else min(len(episodes), cfg.episodes_per_split)
        selected.extend((index, split_label) for index in sampler.sample(episodes, count))
    if not selected:
        raise ValueError(f"No episodes to evaluate for split={cfg.split!r}.")
    return sorted(selected)


def episodes_within_dataset(episodes: list[int], total_episodes: int) -> list[int]:
    out_of_range = [index for index in episodes if not 0 <= index < total_episodes]
    if out_of_range:
        logging.warning(
            f"Ignoring split episodes outside the dataset range [0, {total_episodes}): {out_of_range}"
        )
    return [index for index in episodes if 0 <= index < total_episodes]


def split_label_of(episode_index: int, episode_split: EpisodeSplit | None) -> str:
    if episode_split is None:
        return MANUAL_SPLIT_LABEL
    for split_label, episodes in (
        ("train", episode_split.train_episodes),
        ("val", episode_split.val_episodes),
        ("test", episode_split.test_episodes),
    ):
        if episode_index in episodes:
            return split_label
    return MANUAL_SPLIT_LABEL


def load_replay_dataset(
    cfg: OpenLoopEvalConfig,
    policy_config: "PreTrainedConfig",
    dataset_metadata: LeRobotDatasetMetadata,
    excluded_features: list[str],
    episodes_to_evaluate: list[tuple[int, str]],
) -> LeRobotDataset:
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=sorted({index for index, _ in episodes_to_evaluate}),
        delta_timestamps=observation_delta_timestamps(policy_config, dataset_metadata, cfg.rename_map),
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
    )
    dataset.meta.exclude_features(excluded_features)
    return dataset


def observation_delta_timestamps(
    policy_config: "PreTrainedConfig", dataset_metadata: LeRobotDatasetMetadata, rename_map: dict[str, str]
) -> dict[str, list[float]] | None:
    """Delta timestamps of the policy's observation keys."""
    delta_timestamps = resolve_delta_timestamps(policy_config, dataset_metadata, rename_map)
    if not delta_timestamps:
        return None
    # The recorded action is compared frame by frame, so the policy's action chunk window is dropped.
    observation_only = {
        key: timestamps for key, timestamps in delta_timestamps.items() if rename_map.get(key, key) != ACTION
    }
    return observation_only or None


def dataset_key_for(rename_map: dict[str, str], policy_key: str) -> str:
    """Return the dataset key that `rename_map` maps onto `policy_key`."""
    return next(
        (dataset_key for dataset_key, renamed in rename_map.items() if renamed == policy_key), policy_key
    )


@dataclass
class EpisodeReplay:
    episode_index: int
    split_label: str
    predicted: np.ndarray
    ground_truth: np.ndarray
    observed_state: np.ndarray | None


def replay_episode(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    episode_index: int,
    split_label: str,
    state_key: str,
    rename_map: dict[str, str],
    task_override: str | None,
    excluded_features: list[str],
) -> EpisodeReplay:
    """Run the policy frame by frame over one episode."""
    policy.reset()
    predicted: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    observed_state: list[np.ndarray] = []
    for frame_index in episode_frame_indices(dataset, episode_index):
        item = dataset[frame_index]
        ground_truth.append(item[ACTION].numpy())
        if state_key in item:
            state = item[state_key]
            observed_state.append((state[-1] if state.ndim > 1 else state).numpy())
        batch = _preprocess_dataset_batch(
            default_collate([item]),
            dataset.meta.camera_keys,
            rename_map,
            preprocessor,
            exclude_features=excluded_features,
            task_override=task_override,
        )
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)
        predicted.append(action.detach().cpu().numpy().reshape(-1))
    replay = EpisodeReplay(
        episode_index=episode_index,
        split_label=split_label,
        predicted=np.stack(predicted),
        ground_truth=np.stack(ground_truth),
        observed_state=np.stack(observed_state) if observed_state else None,
    )
    if replay.predicted.shape != replay.ground_truth.shape:
        raise ValueError(
            f"Predicted actions have shape {replay.predicted.shape} but recorded actions have shape "
            f"{replay.ground_truth.shape}; `select_action` must return one action per frame."
        )
    return replay


def episode_frame_indices(dataset: LeRobotDataset, episode_index: int) -> list[int]:
    """Frame indices of one episode in recording order."""
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=[episode_index],
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    return sampler.indices


@dataclass
class EpisodeMetrics:
    episode_index: int
    split_label: str
    num_frames: int
    mse: float
    mae: float
    rmse: float
    cosine_similarity: float
    max_abs_error: float
    max_abs_error_joint: str
    per_joint_mse: dict[str, float]
    per_joint_mae: dict[str, float]
    per_joint_rmse: dict[str, float]
    block_mae: dict[str, float]
    block_rmse: dict[str, float]
    block_cosine: dict[str, float]
    predicted_smoothness_mean: float
    predicted_smoothness_std: float
    ground_truth_smoothness_mean: float
    ground_truth_smoothness_std: float


def compute_episode_metrics(replay: EpisodeReplay, joint_names: list[str]) -> EpisodeMetrics:
    """Compute the error metrics of one replayed episode."""
    error = replay.predicted - replay.ground_truth
    groups = channel_groups(joint_names)
    # The headline metrics cover the position channels; a dataset without any falls back to every channel.
    position_indices = groups.get("position", list(range(len(joint_names))))

    def mean_squared_error(indices: list[int]) -> float:
        return float(np.mean(error[:, indices] ** 2))

    def mean_absolute_error(indices: list[int]) -> float:
        return float(np.mean(np.abs(error[:, indices])))

    def cosine_similarity(indices: list[int]) -> float:
        predicted = replay.predicted[:, indices]
        ground_truth = replay.ground_truth[:, indices]
        norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(ground_truth, axis=1)
        dots = np.sum(predicted * ground_truth, axis=1)
        return float(np.mean(np.where(norms > COSINE_EPSILON, dots / np.maximum(norms, COSINE_EPSILON), 0.0)))

    absolute_error = np.abs(error)
    max_position = np.unravel_index(np.argmax(absolute_error), absolute_error.shape)
    per_joint_mse = {name: float(np.mean(error[:, index] ** 2)) for index, name in enumerate(joint_names)}
    predicted_steps, ground_truth_steps = trajectory_step_norms(replay)
    return EpisodeMetrics(
        episode_index=replay.episode_index,
        split_label=replay.split_label,
        num_frames=int(replay.predicted.shape[0]),
        mse=mean_squared_error(position_indices),
        mae=mean_absolute_error(position_indices),
        rmse=float(np.sqrt(mean_squared_error(position_indices))),
        cosine_similarity=cosine_similarity(position_indices),
        max_abs_error=float(absolute_error[max_position]),
        max_abs_error_joint=joint_names[int(max_position[1])],
        per_joint_mse=per_joint_mse,
        per_joint_mae={
            name: float(np.mean(absolute_error[:, index])) for index, name in enumerate(joint_names)
        },
        per_joint_rmse={name: float(np.sqrt(value)) for name, value in per_joint_mse.items()},
        block_mae={field: mean_absolute_error(indices) for field, indices in groups.items()},
        block_rmse={field: float(np.sqrt(mean_squared_error(indices))) for field, indices in groups.items()},
        block_cosine={field: cosine_similarity(indices) for field, indices in groups.items()},
        predicted_smoothness_mean=float(np.mean(predicted_steps)),
        predicted_smoothness_std=float(np.std(predicted_steps)),
        ground_truth_smoothness_mean=float(np.mean(ground_truth_steps)),
        ground_truth_smoothness_std=float(np.std(ground_truth_steps)),
    )


def trajectory_step_norms(replay: EpisodeReplay) -> tuple[np.ndarray, np.ndarray]:
    """L2 norms of consecutive action differences."""
    if replay.predicted.shape[0] < 2:
        return np.zeros(1), np.zeros(1)
    predicted_steps = np.linalg.norm(np.diff(replay.predicted, axis=0), axis=1)
    ground_truth_steps = np.linalg.norm(np.diff(replay.ground_truth, axis=0), axis=1)
    return predicted_steps, ground_truth_steps


def write_metrics_summary(all_metrics: list[EpisodeMetrics], output_path: Path) -> None:
    by_split: dict[str, list[EpisodeMetrics]] = {}
    for metrics in all_metrics:
        by_split.setdefault(metrics.split_label, []).append(metrics)
    summary = {
        "overall": aggregate_metrics(all_metrics),
        "by_split": {split_label: aggregate_metrics(metrics) for split_label, metrics in by_split.items()},
    }
    write_json(summary, output_path)


def aggregate_metrics(metrics_list: list[EpisodeMetrics]) -> dict[str, Any]:
    """Average the per-episode metrics."""
    if not metrics_list:
        return {}

    def mean_of(attribute: str) -> float:
        return float(np.mean([getattr(metrics, attribute) for metrics in metrics_list]))

    def mean_of_dict(attribute: str) -> dict[str, float]:
        keys = getattr(metrics_list[0], attribute).keys()
        return {
            key: float(np.mean([getattr(metrics, attribute)[key] for metrics in metrics_list]))
            for key in keys
        }

    return {
        "count": len(metrics_list),
        "mean_mse": mean_of("mse"),
        "mean_mae": mean_of("mae"),
        "mean_rmse": mean_of("rmse"),
        "mean_cosine_similarity": mean_of("cosine_similarity"),
        "mean_max_abs_error": mean_of("max_abs_error"),
        "mean_predicted_smoothness": mean_of("predicted_smoothness_mean"),
        "mean_ground_truth_smoothness": mean_of("ground_truth_smoothness_mean"),
        "per_joint_mae": mean_of_dict("per_joint_mae"),
        "per_joint_mse": mean_of_dict("per_joint_mse"),
        "block_mae": mean_of_dict("block_mae"),
        "block_rmse": mean_of_dict("block_rmse"),
    }


def write_metrics_per_episode(all_metrics: list[EpisodeMetrics], output_path: Path) -> None:
    if not all_metrics:
        return
    scalar_columns = [
        "episode_index",
        "split_label",
        "num_frames",
        "mse",
        "mae",
        "rmse",
        "cosine_similarity",
        "max_abs_error",
        "max_abs_error_joint",
        "predicted_smoothness_mean",
        "predicted_smoothness_std",
        "ground_truth_smoothness_mean",
        "ground_truth_smoothness_std",
    ]
    joint_names = list(all_metrics[0].per_joint_mae)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=scalar_columns + [f"mae_{name}" for name in joint_names])
        writer.writeheader()
        for metrics in all_metrics:
            row = {column: getattr(metrics, column) for column in scalar_columns}
            row.update({f"mae_{name}": metrics.per_joint_mae[name] for name in joint_names})
            writer.writerow(row)


def log_summary(all_metrics: list[EpisodeMetrics]) -> None:
    overall = aggregate_metrics(all_metrics)
    logging.info(
        f"overall ({overall['count']} episodes): mae={overall['mean_mae']:.4f} rmse={overall['mean_rmse']:.4f}"
    )
    for split_label in sorted({metrics.split_label for metrics in all_metrics}):
        split_summary = aggregate_metrics(
            [metrics for metrics in all_metrics if metrics.split_label == split_label]
        )
        logging.info(
            f"{split_label} ({split_summary['count']} episodes): "
            f"mae={split_summary['mean_mae']:.4f} rmse={split_summary['mean_rmse']:.4f}"
        )


def plot_episode_joints(
    replay: EpisodeReplay,
    metrics: EpisodeMetrics,
    joint_names: list[str],
    state_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot recorded, predicted and observed trajectories per joint."""
    plt = _import_pyplot()
    num_joints = len(joint_names)
    num_columns = min(4, num_joints)
    num_rows = -(-num_joints // num_columns)
    observed = align_observed_state_to_actions(replay.observed_state, joint_names, state_names)
    frames = np.arange(replay.predicted.shape[0])

    figure, axes = plt.subplots(num_rows, num_columns, figsize=(4 * num_columns, 3 * num_rows), squeeze=False)
    figure.suptitle(
        f"Episode {replay.episode_index} [{replay.split_label}] MAE: {metrics.mae:.4f}", fontsize=14
    )
    for joint_index, name in enumerate(joint_names):
        axis = axes[joint_index // num_columns][joint_index % num_columns]
        axis.plot(frames, replay.ground_truth[:, joint_index], "b-", linewidth=1.0, label="recorded")
        axis.plot(frames, replay.predicted[:, joint_index], "r--", linewidth=1.0, label="predicted")
        if observed is not None and np.isfinite(observed[:, joint_index]).any():
            axis.plot(
                frames, observed[:, joint_index], color="purple", linewidth=0.9, alpha=0.7, label="observed"
            )
        axis.set_title(f"{name} (MAE: {metrics.per_joint_mae[name]:.4f})", fontsize=9)
        axis.set_xlabel("frame", fontsize=8)
        axis.tick_params(labelsize=7)
        if joint_index == 0:
            axis.legend(fontsize=7)
    for unused_index in range(num_joints, num_rows * num_columns):
        axes[unused_index // num_columns][unused_index % num_columns].set_visible(False)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def align_observed_state_to_actions(
    observed_state: np.ndarray | None, joint_names: list[str], state_names: list[str]
) -> np.ndarray | None:
    """Align observed state columns to the action joints by name."""
    if observed_state is None:
        return None
    if observed_state.shape[1] == len(joint_names) and (not state_names or state_names == joint_names):
        return observed_state
    aligned = np.full((observed_state.shape[0], len(joint_names)), np.nan, dtype=np.float64)
    state_index_by_name = {name: index for index, name in enumerate(state_names)}
    for joint_index, name in enumerate(joint_names):
        state_index = state_index_by_name.get(name)
        if state_index is not None and state_index < observed_state.shape[1]:
            aligned[:, joint_index] = observed_state[:, state_index]
    return aligned


def plot_per_joint_mae_by_split(
    all_metrics: list[EpisodeMetrics], joint_names: list[str], output_path: Path, dpi: int
) -> None:
    """Plot per-joint MAE across episodes, grouped by split."""
    plt = _import_pyplot()
    split_labels = sorted({metrics.split_label for metrics in all_metrics})
    if not split_labels or not joint_names:
        return
    figure, axis = plt.subplots(figsize=(max(10, len(joint_names) * 1.5), 6))
    colors = plt.cm.Set2.colors
    box_width = 0.8 / len(split_labels)
    positions = np.arange(len(joint_names))
    for split_position, split_label in enumerate(split_labels):
        split_metrics = [metrics for metrics in all_metrics if metrics.split_label == split_label]
        values = [[metrics.per_joint_mae[name] for metrics in split_metrics] for name in joint_names]
        offset = (split_position - len(split_labels) / 2 + 0.5) * box_width
        boxes = axis.boxplot(
            values,
            positions=positions + offset,
            widths=box_width * 0.8,
            patch_artist=True,
            manage_ticks=False,
        )
        color = colors[split_position % len(colors)]
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in boxes["medians"]:
            median.set_color("black")
        axis.plot([], [], color=color, label=split_label, linewidth=10, alpha=0.6)
    axis.set_xlabel("joint")
    axis.set_ylabel("MAE")
    axis.set_title("Per-joint MAE across episodes, by split")
    axis.set_xticks(positions)
    axis.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=9)
    axis.grid(axis="y", linestyle="--", alpha=0.7)
    axis.legend(title="split")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


if __name__ == "__main__":
    main()
