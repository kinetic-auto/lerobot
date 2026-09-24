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
"""Save episode overviews with ACT attention rolled out onto camera keyframes.

Requires: pip install 'lerobot[dataset_viz]'

Example:
```
lerobot-attention-plot \
    --dataset-path /path/to/my_dataset \
    --policy-path /path/to/checkpoints/last/pretrained_model \
    --episodes 0
```
"""

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE
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


class AttentionRecorder:
    """Capture encoder self-attention, every decoder layer, and feature-map shapes."""

    def __init__(self, model: ACT) -> None:
        self.model = model
        self.encoder_self_attention: list[Tensor] = []
        self.decoder_self_attention: list[Tensor] = []
        self.decoder_cross_attention: list[Tensor] = []
        self.feature_map_shapes: list[tuple[int, int]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "AttentionRecorder":
        for layer in self.model.encoder.layers:
            self._handles.append(layer.self_attn.register_forward_hook(self._record_encoder_self_attention))
        for layer in self.model.decoder.layers:
            self._handles.append(layer.self_attn.register_forward_hook(self._record_decoder_self_attention))
            self._handles.append(layer.multihead_attn.register_forward_hook(self._record_decoder_cross_attention))
        self._handles.append(self.model.backbone.register_forward_hook(self._record_feature_map_shape))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _record_encoder_self_attention(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        del module, inputs
        self.encoder_self_attention.append(self._attention_weights(output))

    def _record_decoder_self_attention(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        del module, inputs
        self.decoder_self_attention.append(self._attention_weights(output))

    def _record_decoder_cross_attention(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        del module, inputs
        self.decoder_cross_attention.append(self._attention_weights(output))

    def _record_feature_map_shape(self, module: nn.Module, inputs: tuple, output: dict) -> None:
        del module, inputs
        feature_map = output["feature_map"]
        self.feature_map_shapes.append((int(feature_map.shape[-2]), int(feature_map.shape[-1])))

    @staticmethod
    def _attention_weights(output: tuple) -> Tensor:
        weights = output[1]
        if weights is None:
            raise RuntimeError(
                "MultiheadAttention returned no weights. The fast path was taken; attention cannot be read."
            )
        return weights.detach()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Local dataset folder path.")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Local ACT checkpoint directory or Hub id.",
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
        help="Policy camera keys to overlay. Default: every camera in the ACT checkpoint.",
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
        help="Directory for `episode_<i>.png` files. Default: `<dataset-path>/attention_plots/`.",
    )
    parser.add_argument("--dpi", type=int, default=130, help="Saved figure resolution.")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Timestamp tolerance forwarded to LeRobotDataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Default: the checkpoint device after auto-selection.",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.5,
        help="Blend weight of the attention heatmap over the image.",
    )
    args = parser.parse_args()

    init_logging()
    dataset = load_dataset(args.dataset_path, args.episodes, args.tolerance_s)
    feature_keys = resolve_feature_keys(dataset, args.feature_keys)
    obs_plot_groups = resolve_obs_plot_groups(args.obs_plot_groups)
    episode_indices = resolve_episode_indices_to_plot(dataset, args.episodes)
    policy, preprocessor = load_act_policy(args.policy_path, dataset.meta, args.device)
    image_observation_keys = resolve_policy_image_keys(dataset, policy, args.image_observations)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(str(dataset.root), "attention_plots")
    )

    for episode_index in episode_indices:
        output_path = plot_episode_attention(
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


def resolve_policy_image_keys(
    dataset: LeRobotDataset, policy: ACTPolicy, image_observations: list[str] | None
) -> list[str]:
    """Select policy cameras to overlay, in checkpoint order.

    Args:
        dataset (LeRobotDataset): Dataset.
        policy (ACTPolicy): Loaded ACT policy.
        image_observations (list[str] | None): Requested camera keys.

    Returns:
        list[str]: Camera keys.
    """
    policy_image_keys = list(policy.config.image_features)
    if image_observations is None:
        return policy_image_keys
    requested = resolve_image_observation_keys(dataset, image_observations)
    unknown = [key for key in requested if key not in policy_image_keys]
    if unknown:
        raise ValueError(
            f"Image observation keys are not policy cameras: {unknown}. Policy cameras: {policy_image_keys}"
        )
    return [key for key in policy_image_keys if key in requested]


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
            f"Feature keys are missing or not 1-D vector features: {invalid}. Plottable keys: {plottable}"
        )


def load_act_policy(
    policy_path: str, dataset_meta: LeRobotDatasetMetadata, device: str | None
) -> tuple[ACTPolicy, PolicyProcessorPipeline]:
    """Load an ACT checkpoint and the preprocessor saved with it.

    Args:
        policy_path (str): Checkpoint directory or Hub id.
        dataset_meta (LeRobotDatasetMetadata): Dataset metadata.
        device (str | None): Torch device override.

    Returns:
        tuple[ACTPolicy, PolicyProcessorPipeline]: Policy and preprocessor.
    """
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path
    if device is not None:
        policy_config.device = device
    policy = make_policy(policy_config, ds_meta=dataset_meta)
    if not isinstance(policy, ACTPolicy):
        raise TypeError(f"Policy at {policy_path} is {type(policy).__name__}, expected ACTPolicy.")
    if not policy.config.image_features:
        raise ValueError(f"ACT checkpoint at {policy_path} has no image features to plot.")
    missing_cameras = [key for key in policy.config.image_features if key not in dataset_meta.features]
    if missing_cameras:
        raise ValueError(
            f"Dataset is missing policy camera features: {missing_cameras}. "
            f"Dataset features: {list(dataset_meta.features)}"
        )
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return policy, preprocessor


def plot_episode_attention(
    dataset: LeRobotDataset,
    policy: ACTPolicy,
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
    """Save one episode overview with attention overlaid on camera keyframes.

    Args:
        dataset (LeRobotDataset): Dataset.
        policy (ACTPolicy): ACT policy.
        preprocessor (PolicyProcessorPipeline): Policy preprocessor.
        episode_index (int): Episode index.
        image_observation_keys (list[str]): Camera keys to overlay.
        feature_keys (list[str]): Feature keys.
        obs_plot_groups (list[str]): Dimension groups to plot.
        num_image_samples (int): Image count per strip.
        output_dir (str): Output directory.
        dpi (int): Figure resolution.
        heatmap_alpha (float): Heatmap blend weight.

    Returns:
        str: Saved figure path.
    """
    timestamps, signals = read_episode_signals(dataset, episode_index, feature_keys)
    split_timestamps = split_episode_timestamps(timestamps, num_image_samples)
    midpoint_timestamps = select_midpoint_timestamps(split_timestamps)
    keyframe_indices = snap_to_frame_indices(midpoint_timestamps, timestamps)
    keyframe_timestamps = timestamps[keyframe_indices]
    episode_frame_indices = episode_row_indices(dataset, episode_index)
    dataset_indices = episode_frame_indices[keyframe_indices]

    overlaid: dict[str, list[np.ndarray]] = {key: [] for key in image_observation_keys}
    for dataset_index in dataset_indices:
        frame = dataset[int(dataset_index)]
        attention_maps = compute_attention_rollout(policy, preprocessor, frame)
        for key in image_observation_keys:
            image_uint8 = frame_image_to_uint8(frame[key])
            overlaid[key].append(overlay_attention(image_uint8, attention_maps[key], heatmap_alpha))

    signal_names = {key: get_feature_names(dataset, key) for key in feature_keys}
    figure = draw_episode_overview(
        dataset.repo_id,
        policy.config.type,
        episode_index,
        timestamps,
        signals,
        signal_names,
        feature_keys,
        obs_plot_groups,
        split_timestamps,
        keyframe_timestamps,
        overlaid,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"episode_{episode_index}.png")
    figure.savefig(output_path, dpi=dpi)
    _import_pyplot().close(figure)
    return output_path


def episode_row_indices(dataset: LeRobotDataset, episode_index: int) -> np.ndarray:
    """Return dataset row positions for one episode.

    Args:
        dataset (LeRobotDataset): Dataset.
        episode_index (int): Episode index.

    Returns:
        np.ndarray: Row indices accepted by ``dataset[idx]``.
    """
    episode_column = np.asarray(dataset.hf_dataset["episode_index"]).reshape(-1)
    return np.flatnonzero(episode_column == episode_index)


def frame_image_to_uint8(image: Tensor) -> np.ndarray:
    """Convert a CHW float image in ``[0, 1]`` to HWC uint8.

    Args:
        image (Tensor): Image tensor.

    Returns:
        np.ndarray: HWC uint8 image.
    """
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(np.asarray(array, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def compute_attention_rollout(
    policy: ACTPolicy, preprocessor: PolicyProcessorPipeline, frame: dict
) -> dict[str, np.ndarray]:
    """Run one frame and return a spatial attention map per policy camera.

    Args:
        policy (ACTPolicy): ACT policy.
        preprocessor (PolicyProcessorPipeline): Policy preprocessor.
        frame (dict): One dataset frame.

    Returns:
        dict[str, np.ndarray]: ``(h, w)`` attention map per camera.
    """
    observation = {key: value for key, value in frame.items() if key.startswith(OBS_PREFIX)}
    observation["task"] = frame["task"]
    batch = preprocessor(observation)
    with AttentionRecorder(policy.model) as recorder:
        policy.predict_action_chunk(batch)
    return rollout_to_camera_maps(recorder, policy.config)


def rollout_to_camera_maps(recorder: AttentionRecorder, config: ACTConfig) -> dict[str, np.ndarray]:
    """Turn captured attention into one ``(h, w)`` map per camera.

    Args:
        recorder (AttentionRecorder): Weights from one forward pass.
        config (ACTConfig): ACT config.

    Returns:
        dict[str, np.ndarray]: Attention maps keyed by camera.
    """
    if len(recorder.decoder_cross_attention) != config.n_decoder_layers:
        raise RuntimeError(
            f"Expected {config.n_decoder_layers} decoder cross-attention matrices, "
            f"captured {len(recorder.decoder_cross_attention)}."
        )
    if len(recorder.decoder_self_attention) != config.n_decoder_layers:
        raise RuntimeError(
            f"Expected {config.n_decoder_layers} decoder self-attention matrices, "
            f"captured {len(recorder.decoder_self_attention)}."
        )
    if len(recorder.encoder_self_attention) != config.n_encoder_layers:
        raise RuntimeError(
            f"Expected {config.n_encoder_layers} encoder self-attention matrices, "
            f"captured {len(recorder.encoder_self_attention)}."
        )
    encoder_rollout = rollout_encoder_attention(recorder.encoder_self_attention)
    decoder_cross_attention = combine_decoder_cross_attention(
        recorder.decoder_self_attention, recorder.decoder_cross_attention
    )
    token_attention = attention_to_input_tokens(decoder_cross_attention, encoder_rollout)
    return split_camera_attention(
        token_attention,
        recorder.feature_map_shapes,
        count_prefix_tokens(config),
        list(config.image_features),
    )


def rollout_encoder_attention(encoder_self_attention: list[Tensor]) -> Tensor:
    """Compose residual-aware encoder self-attention into one input-token map.

    Args:
        encoder_self_attention (list[Tensor]): Per-layer ``(1, S, S)`` weights, layer 0 first.

    Returns:
        Tensor: Rollout ``(1, S, S)``.
    """
    rollout: Tensor | None = None
    for attention in encoder_self_attention:
        identity = torch.eye(attention.shape[-1], device=attention.device, dtype=attention.dtype)
        mixed = 0.5 * attention + 0.5 * identity
        rollout = mixed if rollout is None else mixed @ rollout
    if rollout is None:
        raise RuntimeError("Encoder self-attention was not captured.")
    return rollout


def combine_decoder_cross_attention(
    decoder_self_attention: list[Tensor], decoder_cross_attention: list[Tensor]
) -> Tensor:
    """Residual-sum every decoder layer's cross-attention onto encoder tokens.

    Layer ``k`` injects ``C_k``. Each later layer keeps half of that residual and mixes the
    action queries with ``S' = 0.5 * S + 0.5 * I``. Raw layer weights are ``(1/2)^{L-k}``
    and are normalized to sum to 1, so a single layer returns ``C`` unchanged.

    Args:
        decoder_self_attention (list[Tensor]): Per-layer ``(1, Q, Q)`` weights, layer 0 first.
        decoder_cross_attention (list[Tensor]): Per-layer ``(1, Q, S)`` weights, layer 0 first.

    Returns:
        Tensor: Combined cross-attention ``(1, Q, S)``.
    """
    n_layers = len(decoder_cross_attention)
    if n_layers == 0 or len(decoder_self_attention) != n_layers:
        raise RuntimeError(
            f"Expected one decoder self-attention and cross-attention per layer, "
            f"got {len(decoder_self_attention)} and {n_layers}."
        )
    cross_attention = decoder_cross_attention[0]
    batch_size, query_count, _ = cross_attention.shape
    carried = torch.eye(query_count, device=cross_attention.device, dtype=cross_attention.dtype)
    carried = carried.unsqueeze(0).expand(batch_size, query_count, query_count)
    terms: list[Tensor] = []
    raw_weights: list[float] = []
    for layer_index in range(n_layers - 1, -1, -1):
        terms.append(carried @ decoder_cross_attention[layer_index])
        raw_weights.append(0.5 ** (n_layers - layer_index))
        self_attention = decoder_self_attention[layer_index]
        identity = torch.eye(query_count, device=self_attention.device, dtype=self_attention.dtype)
        mixed_self_attention = 0.5 * self_attention + 0.5 * identity
        carried = carried @ mixed_self_attention
    weight_scale = sum(raw_weights)
    combined = torch.zeros_like(cross_attention)
    for weight, term in zip(raw_weights, terms, strict=True):
        combined = combined + (weight / weight_scale) * term
    return combined


def attention_to_input_tokens(decoder_cross_attention: Tensor, encoder_rollout: Tensor) -> Tensor:
    """Average action-query attention after mapping it onto encoder input tokens.

    Args:
        decoder_cross_attention (Tensor): ``(1, Q, S)`` cross-attention.
        encoder_rollout (Tensor): ``(1, S, S)`` encoder rollout.

    Returns:
        Tensor: ``(S,)`` input-token attention.
    """
    query_attention = decoder_cross_attention @ encoder_rollout
    return query_attention.mean(dim=1).squeeze(0)


def count_prefix_tokens(config: ACTConfig) -> int:
    """Count leading non-image encoder tokens.

    Args:
        config (ACTConfig): ACT config.

    Returns:
        int: Latent token, plus robot state and env state when the checkpoint has them.
    """
    prefix_count = 1
    if config.robot_state_feature is not None:
        prefix_count += 1
    if config.env_state_feature is not None:
        prefix_count += 1
    return prefix_count


def split_camera_attention(
    token_attention: Tensor,
    feature_map_shapes: list[tuple[int, int]],
    prefix_count: int,
    image_keys: list[str],
) -> dict[str, np.ndarray]:
    """Drop prefix tokens and reshape the rest into per-camera feature maps.

    Args:
        token_attention (Tensor): ``(S,)`` input-token attention.
        feature_map_shapes (list[tuple[int, int]]): ``(h, w)`` per camera, backbone order.
        prefix_count (int): Leading non-image tokens.
        image_keys (list[str]): Camera keys in checkpoint order.

    Returns:
        dict[str, np.ndarray]: ``(h, w)`` maps.
    """
    if len(feature_map_shapes) != len(image_keys):
        raise RuntimeError(
            f"Captured {len(feature_map_shapes)} feature maps for {len(image_keys)} policy cameras."
        )
    camera_attention = token_attention[prefix_count:]
    expected = sum(height * width for height, width in feature_map_shapes)
    if camera_attention.numel() != expected:
        raise RuntimeError(
            f"Camera tokens ({camera_attention.numel()}) do not match feature-map pixels ({expected})."
        )
    maps: dict[str, np.ndarray] = {}
    offset = 0
    for key, (height, width) in zip(image_keys, feature_map_shapes, strict=True):
        count = height * width
        patch = camera_attention[offset : offset + count]
        maps[key] = patch.reshape(height, width).detach().cpu().numpy()
        offset += count
    return maps


def overlay_attention(image_uint8: np.ndarray, attention_map: np.ndarray, alpha: float) -> np.ndarray:
    """Min-max normalize a feature map, upsample it, and blend a jet heatmap onto the image.

    Args:
        image_uint8 (np.ndarray): HWC uint8 image.
        attention_map (np.ndarray): ``(h, w)`` attention.
        alpha (float): Heatmap blend weight.

    Returns:
        np.ndarray: HWC uint8 overlay.
    """
    flat = np.asarray(attention_map, dtype=np.float32)
    low = float(flat.min())
    high = float(flat.max())
    if high > low:
        normalized = (flat - low) / (high - low)
    else:
        normalized = np.zeros_like(flat)
    height, width = image_uint8.shape[:2]
    upsampled = torch.nn.functional.interpolate(
        torch.from_numpy(normalized)[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    heatmap = _import_pyplot().get_cmap("jet")(upsampled)[..., :3]
    blended = (1.0 - alpha) * image_uint8.astype(np.float32) + alpha * (heatmap * 255.0)
    return np.clip(blended, 0, 255).astype(np.uint8)


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


def snap_to_frame_indices(target_timestamps: np.ndarray, frame_timestamps: np.ndarray) -> np.ndarray:
    """Snap targets to the nearest frame index.

    Args:
        target_timestamps (np.ndarray): Target timestamps.
        frame_timestamps (np.ndarray): Frame timestamps.

    Returns:
        np.ndarray: Frame indices.
    """
    if frame_timestamps.size == 0:
        raise ValueError("Cannot snap timestamps: episode has no frames")
    distances = np.abs(frame_timestamps[:, None] - target_timestamps[None, :])
    return distances.argmin(axis=0)


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
    policy_name: str,
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
        policy_name (str): Policy name.
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
        (
            f"{repo_id}   policy {policy_name}   episode {episode_index}   "
            f"{len(timestamps)} frames, {timestamps[-1]:.1f}s"
        ),
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
        for group_name, dim_indices, series_names in group_feature_dims(signal_names[key], obs_plot_groups):
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
