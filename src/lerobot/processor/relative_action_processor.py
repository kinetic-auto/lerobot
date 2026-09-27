# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry

if TYPE_CHECKING:  # a runtime import would be circular: ``pretrained`` imports from this package
    from lerobot.policies.pretrained import PreTrainedPolicy

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "bind_relative_anchor",
    "RELATIVE_ACTION_MODES",
    "to_relative_actions",
    "to_absolute_actions",
    "to_sequential_actions",
    "from_sequential_actions",
    "validate_relative_action_names",
]

RELATIVE_ACTION_MODES = ("obs_t", "sequential")


def _relative_state_offset(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> tuple[Tensor, int]:
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    # A temporally stacked observation (a policy whose ``observation_delta_indices`` spans several
    # frames, e.g. VLA-JEPA or LingBot-VA) hands over a (B, T_obs, state_dim) state. The reference
    # is the CURRENT frame -- delta 0, i.e. index 0 -- so collapse to it and let the offset
    # broadcast over the action horizon. pi0/pi05 pass a 2D (B, state_dim) state and are unaffected.
    if state.ndim == 3:
        state = state[:, 0]
    dims = min(actions.shape[-1], state.shape[-1], len(mask))
    if any(mask[dims : min(actions.shape[-1], len(mask))]):
        raise ValueError("Relative action dimensions cannot exceed the observation.state width.")
    mask_t = torch.tensor(mask[:dims], dtype=actions.dtype, device=actions.device)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    return state_offset, dims


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim), or (B, T_obs, state_dim) for a temporally stacked
            observation (collapsed to the current frame). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    state_offset, dims = _relative_state_offset(actions, state, mask)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim), or (B, T_obs, state_dim) for a temporally stacked
            observation (collapsed to the current frame). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    state_offset, dims = _relative_state_offset(actions, state, mask)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


def to_sequential_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    sequential = to_relative_actions(actions, state, mask)
    if actions.ndim != 3 or actions.shape[1] < 2:
        return sequential
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    sequential[:, 1:, :dims] = (actions[:, 1:, :dims] - actions[:, :-1, :dims]) * mask_t + actions[
        :, 1:, :dims
    ] * (~mask_t.bool())
    return sequential


def from_sequential_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    if actions.ndim != 3:
        return to_absolute_actions(actions, state, mask)
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    accumulated = actions.clone()
    accumulated[..., :dims] = torch.cumsum(actions[..., :dims], dim=1) * mask_t + actions[..., :dims] * (
        ~mask_t.bool()
    )
    return to_absolute_actions(accumulated, state, mask)


def validate_relative_action_names(
    action_names: Sequence[str], state_names: Sequence[str], mask: Sequence[bool]
) -> None:
    mismatches = []
    for index, is_relative in enumerate(mask):
        if not is_relative:
            continue
        action_name = action_names[index] if index < len(action_names) else "<missing>"
        state_name = state_names[index] if index < len(state_names) else "<missing>"
        if action_name != state_name:
            mismatches.append(f"{index}: action={action_name!r}, state={state_name!r}")
    if mismatches:
        raise ValueError(
            "Relative action dimensions must align by name with observation.state: " + "; ".join(mismatches)
        )


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    mode: str = "obs_t"
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _count_queued_actions: Callable[[], int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in RELATIVE_ACTION_MODES:
            raise ValueError(f"Unknown relative action mode {self.mode!r}; expected {RELATIVE_ACTION_MODES}.")

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # Cache state for the paired AbsoluteActionsProcessorStep -- but hold it for as long
        # as the policy is still serving the chunk that was generated against it. A fresh
        # chunk re-anchors on the tick the queue runs dry.
        if state is not None and not self._chunk_in_flight():
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        if self.mode == "sequential":
            new_transition[TransitionKey.ACTION] = to_sequential_actions(action, state, mask)
        else:
            new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def reset(self) -> None:
        self._last_state = None

    def _chunk_in_flight(self) -> bool:
        """Whether the policy still holds actions generated against the cached anchor."""
        return self._count_queued_actions is not None and self._count_queued_actions() > 0

    def bind_action_queue(self, count_queued_actions: Callable[[], int] | None) -> None:
        self._count_queued_actions = count_queued_actions

    def get_cached_state(self) -> torch.Tensor | None:
        """Return the cached ``observation.state`` used as the reference point for relative/absolute action conversions."""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "mode": self.mode,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)
    _anchor: torch.Tensor | None = field(default=None, init=False, repr=False)
    _previous_absolute_action: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        if self.relative_step.mode == "sequential":
            if action.ndim == 3:
                absolute_action = from_sequential_actions(action, cached_state, mask)
            else:
                absolute_action = self._integrate_streamed_action(action, cached_state, mask)
        else:
            absolute_action = to_absolute_actions(action, cached_state, mask)
        new_transition[TransitionKey.ACTION] = absolute_action
        return new_transition

    def _integrate_streamed_action(
        self, action: Tensor, cached_state: Tensor, mask: Sequence[bool]
    ) -> Tensor:
        if self._anchor is not cached_state:
            self._anchor = cached_state
            self._previous_absolute_action = None
        reference = cached_state if self._previous_absolute_action is None else self._previous_absolute_action
        absolute_action = to_absolute_actions(action, reference, mask)
        self._previous_absolute_action = absolute_action
        return absolute_action

    def reset(self) -> None:
        self._anchor = None
        self._previous_absolute_action = None

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def bind_relative_anchor(
    policy: "PreTrainedPolicy", pipeline: PolicyProcessorPipeline[Any, Any]
) -> RelativeActionsProcessorStep | None:
    """Let ``pipeline``'s relative-action step hold a chunk's anchor until the chunk drains.

    Call once wherever a policy and its preprocessor are built together; a disabled step counts
    as absent. Returns the step that was bound, or ``None`` if the pipeline has no enabled one.
    """
    step = next(
        (
            s
            for s in getattr(pipeline, "steps", ())
            if isinstance(s, RelativeActionsProcessorStep) and s.enabled
        ),
        None,
    )
    if step is not None:
        step.bind_action_queue(policy.count_queued_actions)
    return step
