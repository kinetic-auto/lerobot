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
from typing import TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

MILLIMETERS_PER_METER = 1000.0
DepthMillimeters = TypeVar("DepthMillimeters", NDArray[np.generic], torch.Tensor)


def zero_depth_outside_valid_range(
    depth_millimeters: DepthMillimeters, minimum_depth_meters: float, maximum_depth_meters: float
) -> DepthMillimeters:
    """Set depth outside a meter range to zero millimeters.

    Args:
        depth_millimeters (DepthMillimeters): Depth in millimeters.
        minimum_depth_meters (float): Minimum depth in meters.
        maximum_depth_meters (float): Maximum depth in meters.

    Returns:
        DepthMillimeters: Depth with out-of-range values set to zero.
    """
    if not 0 <= minimum_depth_meters < maximum_depth_meters:
        raise ValueError(
            "Depth range must satisfy 0 <= minimum < maximum meters. "
            f"Got {(minimum_depth_meters, maximum_depth_meters)}."
        )
    minimum_depth_millimeters = round(minimum_depth_meters * MILLIMETERS_PER_METER)
    maximum_depth_millimeters = round(maximum_depth_meters * MILLIMETERS_PER_METER)

    # Filter out invalid depth values (tensor)
    if isinstance(depth_millimeters, torch.Tensor):
        is_outside_valid_range = (
            ~torch.isfinite(depth_millimeters)
            | (depth_millimeters < minimum_depth_millimeters)
            | (depth_millimeters > maximum_depth_millimeters)
        )
        return depth_millimeters.masked_fill(is_outside_valid_range, 0)

    # Filter out invalid depth values (numpy array)
    depth_copy = np.array(depth_millimeters, copy=True)
    is_outside_valid_range = (
        ~np.isfinite(depth_copy)
        | (depth_copy < minimum_depth_millimeters)
        | (depth_copy > maximum_depth_millimeters)
    )
    depth_copy[is_outside_valid_range] = 0
    return depth_copy
