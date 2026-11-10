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
from typing import Any

import torch

from lerobot.processor import (
    DepthRangeProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    make_default_pre_post_processors,
)

from .configuration_kinetic_act import KineticACTConfig


def make_kinetic_act_pre_post_processors(
    config: KineticACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the KineticACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (KineticACTConfig): The KineticACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """
    preprocessor, postprocessor = make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
    if config.valid_depth_range_meters is not None:
        minimum_depth_meters, maximum_depth_meters = config.valid_depth_range_meters
        _insert_before_normalizer(
            preprocessor, DepthRangeProcessorStep(minimum_depth_meters, maximum_depth_meters)
        )
    return preprocessor, postprocessor


def _insert_before_normalizer(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], step: ProcessorStep
) -> None:
    steps = list(preprocessor.steps)
    normalizer_index = next(
        (index for index, existing in enumerate(steps) if isinstance(existing, NormalizerProcessorStep)),
        None,
    )
    if normalizer_index is None:
        raise ValueError("The preprocessor has no normalizer step to insert the depth range step before.")
    preprocessor.steps = [*steps[:normalizer_index], step, *steps[normalizer_index:]]
