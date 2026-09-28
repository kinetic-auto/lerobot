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
import logging
import math
from dataclasses import dataclass
from pprint import pformat
from typing import Literal, cast

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.processor import RelativeActionsProcessorStep, validate_relative_action_names
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import (
    ACTION,
    IMAGENET_STATS,
    OBS_IMAGE,
    OBS_PREFIX,
    OBS_STATE,
    PRETRAINED_MODEL_DIR,
    REWARD,
    SPLIT_INFO,
)
from lerobot.utils.io_utils import load_json

from .compute_stats import compute_relative_action_stats, compute_sequential_action_stats
from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .storage import DEFAULT_STORAGE_FORMAT, load_dataset_metadata
from .streaming_dataset import StreamingLeRobotDataset
from .utils import (
    EpisodeSplit,
    flatten_feature_names,
    resolve_episode_indices,
    resolve_excluded_features,
    split_episodes_randomly,
)


@dataclass
class TrainEvalDatasets:
    train: LeRobotDataset | MultiLeRobotDataset | StreamingLeRobotDataset
    eval: LeRobotDataset | None
    test: LeRobotDataset | None
    episode_split: EpisodeSplit | None


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig,
    ds_meta: LeRobotDatasetMetadata,
    rename_map: dict[str, str] | None = None,
) -> dict[str, list[float]] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list[float]] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    # Only policies that opt into modality-specific history (currently Pi05 with MEM)
    # define these; everything else falls back to the shared observation indices.
    explicit_image_indices = getattr(cfg, "image_observation_delta_indices", None)
    image_indices = (
        explicit_image_indices if explicit_image_indices is not None else cfg.observation_delta_indices
    )
    explicit_state_indices = getattr(cfg, "state_observation_delta_indices", None)
    state_indices = (
        explicit_state_indices if explicit_state_indices is not None else cfg.observation_delta_indices
    )

    delta_timestamps: dict[str, list[float]] = {}
    matched_image_keys: list[str] = []
    for key in ds_meta.features:
        policy_key = (rename_map or {}).get(key, key)
        if policy_key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if policy_key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        # `OBS_IMAGE` matches both the `observation.image` and `observation.images.<cam>`
        # conventions; matching `OBS_IMAGES` alone would silently give singular-key
        # datasets no image history at all.
        if policy_key.startswith(OBS_IMAGE):
            indices = image_indices
            matched_image_keys.append(key)
        elif policy_key == OBS_STATE:
            indices = state_indices
        else:
            indices = cfg.observation_delta_indices if policy_key.startswith(OBS_PREFIX) else None
        if indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in indices]

    # A policy asking for an image history that no dataset key can supply would train
    # on single frames without any error, so fail instead of degrading silently.
    if explicit_image_indices is not None and len(explicit_image_indices) > 1 and not matched_image_keys:
        raise ValueError(
            f"{type(cfg).__name__} requests {len(explicit_image_indices)} history frames per camera, but no "
            f"dataset feature maps to an image key. Dataset features: {sorted(ds_meta.features)}. "
            "Image keys must be named `observation.image*` after applying `--rename_map`."
        )

    if len(delta_timestamps) == 0:
        return None

    return delta_timestamps


def _finalize_dataset_metadata(
    cfg: TrainPipelineConfig,
    dataset: LeRobotDataset | MultiLeRobotDataset | StreamingLeRobotDataset,
    excluded_features: list[str],
) -> None:
    dataset.meta.exclude_features(excluded_features)
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)


def make_dataset(
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | MultiLeRobotDataset | StreamingLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    excluded_features: list[str] = []

    if isinstance(cfg.dataset.repo_id, str):
        repo_type = cast(Literal["dataset", "bucket"], cfg.dataset.repo_type)
        # Storage-aware loader: same as LeRobotDatasetMetadata(...), plus support
        # for datasets whose root is an object-store URI (e.g. ``hf://``).
        ds_meta = load_dataset_metadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
            repo_type=repo_type,
        )
        excluded_features = resolve_excluded_features(cfg.dataset.exclude_features, ds_meta.features)
        ds_meta.exclude_features(excluded_features)
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta, cfg.rename_map)
        episodes = resolve_episode_indices(
            cfg.dataset.episodes, ds_meta.total_episodes, cfg.dataset.exclude_episodes
        )
        if cfg.dataset.streaming and ds_meta.storage_format != DEFAULT_STORAGE_FORMAT:
            raise ValueError(
                f"dataset.streaming=True is not supported for storage_format="
                f"{ds_meta.storage_format!r}: StreamingLeRobotDataset only reads the default "
                f"{DEFAULT_STORAGE_FORMAT!r} layout. Note that some formats (e.g. 'lance') "
                "support remote map-style access without streaming mode."
            )
        if not cfg.dataset.streaming:
            if repo_type == "bucket" and ds_meta.storage_format == DEFAULT_STORAGE_FORMAT:
                raise ValueError(
                    f"repo_type='bucket' is streaming-only for the default {DEFAULT_STORAGE_FORMAT!r} "
                    "storage format: set dataset.streaming=true to train from an HF Storage Bucket."
                )
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
                repo_type=repo_type,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
                repo_type=repo_type,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    _finalize_dataset_metadata(cfg, dataset, excluded_features)

    return dataset


def _make_episode_subset(
    cfg: TrainPipelineConfig,
    episodes: list[int],
    image_transforms: ImageTransforms | None,
    delta_timestamps: dict[str, list[float]] | None,
    excluded_features: list[str],
) -> LeRobotDataset:
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cast(Literal["dataset", "bucket"], cfg.dataset.repo_type),
    )
    _finalize_dataset_metadata(cfg, dataset, excluded_features)
    return dataset


def _resolve_episode_split(cfg: TrainPipelineConfig, base_episodes: list[int]) -> EpisodeSplit:
    split_ratio = cfg.dataset.split_ratio
    if split_ratio is None:
        raise ValueError("dataset.split_ratio is required.")
    seed = cfg.seed or 0
    if cfg.resume and cfg.checkpoint_path is not None:
        split_path = cfg.checkpoint_path / PRETRAINED_MODEL_DIR / SPLIT_INFO
        if split_path.is_file():
            episode_split = EpisodeSplit.from_dict(load_json(split_path))
            saved_episodes = (
                episode_split.train_episodes + episode_split.val_episodes + episode_split.test_episodes
            )
            if len(saved_episodes) != len(set(saved_episodes)) or sorted(saved_episodes) != sorted(
                base_episodes
            ):
                raise ValueError(
                    "Checkpoint split episodes do not match the current dataset episode selection."
                )
            if (
                episode_split.split_ratio != split_ratio
                or episode_split.seed != seed
                or episode_split.total_episodes != len(base_episodes)
            ):
                logging.warning(
                    "Resume split metadata differs from the current split settings; "
                    "reusing the checkpoint split."
                )
            return episode_split
    return split_episodes_randomly(base_episodes, split_ratio, seed)


def _uses_recomputed_relative_action_stats(policy_cfg: PreTrainedConfig | RewardModelConfig) -> bool:
    return bool(getattr(policy_cfg, "use_relative_actions", False)) and hasattr(
        policy_cfg, "recompute_relative_action_stats"
    )


def apply_relative_action_stats(
    dataset: LeRobotDataset | MultiLeRobotDataset | StreamingLeRobotDataset,
    policy_cfg: PreTrainedConfig | RewardModelConfig,
) -> None:
    if not isinstance(dataset, LeRobotDataset):
        raise ValueError("Relative action statistics require a single map-style LeRobotDataset.")

    action_names = flatten_feature_names(dataset.meta.features[ACTION].get("names"))
    state_names = flatten_feature_names(dataset.meta.features[OBS_STATE].get("names"))
    if action_names is None or state_names is None:
        raise ValueError("Relative actions require named action and observation.state dimensions.")
    mask_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=getattr(policy_cfg, "relative_exclude_joints", []),
        action_names=action_names,
        mode=getattr(policy_cfg, "relative_action_mode", "obs_t"),
    )
    mask = mask_step._build_mask(dataset.meta.features[ACTION]["shape"][0])
    validate_relative_action_names(action_names, state_names, mask)
    if not getattr(policy_cfg, "recompute_relative_action_stats", True):
        return

    if mask_step.mode == "sequential":
        stats = compute_sequential_action_stats(
            dataset.hf_dataset,
            dataset.meta.features,
            getattr(policy_cfg, "relative_exclude_joints", []),
        )
    else:
        stats = compute_relative_action_stats(
            dataset.hf_dataset,
            dataset.meta.features,
            chunk_size=1,
            exclude_joints=getattr(policy_cfg, "relative_exclude_joints", []),
        )
    stats["std"] = stats["std"].clip(min=1e-6)
    dataset.meta.stats[ACTION] = {
        key: torch.as_tensor(value, dtype=torch.float32) for key, value in stats.items()
    }
    logging.info("Applied %s relative-action normalization statistics.", mask_step.mode)


def make_train_eval_datasets(cfg: TrainPipelineConfig) -> TrainEvalDatasets:
    full_dataset = make_dataset(cfg)
    policy_cfg = cfg.trainable_config
    recompute_relative_stats = _uses_recomputed_relative_action_stats(policy_cfg)
    if not isinstance(full_dataset, LeRobotDataset):
        if recompute_relative_stats:
            apply_relative_action_stats(full_dataset, policy_cfg)
        return TrainEvalDatasets(full_dataset, None, None, None)

    if cfg.dataset.eval_split == 0.0 and cfg.dataset.split_ratio is None:
        if recompute_relative_stats:
            apply_relative_action_stats(full_dataset, policy_cfg)
        return TrainEvalDatasets(full_dataset, None, None, None)

    base_episodes = (
        full_dataset.episodes
        if full_dataset.episodes is not None
        else list(range(full_dataset.meta.total_episodes))
    )
    episode_split = None
    if cfg.dataset.split_ratio is not None:
        episode_split = _resolve_episode_split(cfg, base_episodes)
        train_episodes = episode_split.train_episodes
        eval_episodes = episode_split.val_episodes
        test_episodes = episode_split.test_episodes
    else:
        episode_tasks = full_dataset.meta.episodes["tasks"]
        task_to_episodes: dict[str, list[int]] = {}
        for episode_index in base_episodes:
            task_key = episode_tasks[episode_index][0] if episode_tasks[episode_index] else ""
            task_to_episodes.setdefault(task_key, []).append(episode_index)
        train_episodes, eval_episodes = [], []
        for episodes in task_to_episodes.values():
            eval_count = math.ceil(len(episodes) * cfg.dataset.eval_split)
            train_episodes.extend(episodes[: len(episodes) - eval_count])
            eval_episodes.extend(episodes[len(episodes) - eval_count :])
        test_episodes = []

    if not train_episodes:
        raise ValueError(f"Dataset split leaves 0 training episodes from {len(base_episodes)} total.")

    excluded_features = list(cfg.dataset.exclude_features or [])
    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta, cfg.rename_map)
    train_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    train_dataset = _make_episode_subset(
        cfg, train_episodes, train_transforms, delta_timestamps, excluded_features
    )
    eval_dataset = (
        _make_episode_subset(cfg, eval_episodes, None, delta_timestamps, excluded_features)
        if eval_episodes
        else None
    )
    test_dataset = (
        _make_episode_subset(cfg, test_episodes, None, delta_timestamps, excluded_features)
        if test_episodes
        else None
    )
    if recompute_relative_stats:
        apply_relative_action_stats(train_dataset, policy_cfg)
    logging.info(
        "Dataset split: %d train, %d validation, %d test episodes.",
        len(train_episodes),
        len(eval_episodes),
        len(test_episodes),
    )
    return TrainEvalDatasets(train_dataset, eval_dataset, test_dataset, episode_split)
