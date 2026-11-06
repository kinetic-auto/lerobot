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

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

COMPRESSED_DEPTH_HEADER_BYTES = 12
RAW_DEPTH_FORMAT_PREFIX = "16UC1"
MILLIMETERS_PER_METER = 1000.0


@dataclass(frozen=True)
class CameraParameters:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    width: int
    height: int


def decode_compressed_depth_mm(format_string: str, payload: bytes) -> np.ndarray:
    # Check if the encoded depth is in 16UC1
    if not format_string.startswith(RAW_DEPTH_FORMAT_PREFIX):
        raise ValueError(
            f"Unsupported compressedDepth format {format_string!r}: only raw {RAW_DEPTH_FORMAT_PREFIX} "
            "depth stored as PNG is supported."
        )

    # Decode the compressed depth image
    png_bytes = payload[COMPRESSED_DEPTH_HEADER_BYTES:]
    depth_mm = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth_mm is None or depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
        raise ValueError("Compressed depth payload did not decode to a single-channel uint16 image.")

    return depth_mm


def load_camera_parameters_from_intrinsics_yaml(yaml_path: Path) -> CameraParameters:
    # Load the camera parameters from the YAML file
    with open(yaml_path) as yaml_file:
        intrinsics = yaml.safe_load(yaml_file)
    width, height = intrinsics["img_size"]

    # Populate the camera parameters
    return CameraParameters(
        camera_matrix=np.array(intrinsics["K"], dtype=np.float64).reshape(3, 3),
        distortion_coefficients=np.array(intrinsics["D"], dtype=np.float64),
        width=int(width),
        height=int(height),
    )


def load_depth_to_rgb_transform(depth_intrinsics_path: Path, color_intrinsics_path: Path) -> np.ndarray:
    depth_wrt_camera_link = load_optical_frame_wrt_camera_link(depth_intrinsics_path)
    rgb_wrt_camera_link = load_optical_frame_wrt_camera_link(color_intrinsics_path)
    return np.linalg.inv(rgb_wrt_camera_link) @ depth_wrt_camera_link


def load_optical_frame_wrt_camera_link(intrinsics_path: Path) -> np.ndarray:
    with open(intrinsics_path) as yaml_file:
        pose = yaml.safe_load(yaml_file).get("camera_link_to_optical")
    if pose is None:
        raise ValueError(f"{intrinsics_path} has no camera_link_to_optical entry.")

    # static_transforms convention: angles in degrees, translations in meters, R = Rz(yaw) Ry(pitch) Rx(roll)
    roll, pitch, yaw = np.radians([pose["roll"], pose["pitch"], pose["yaw"]])
    rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    optical_frame_wrt_camera_link = np.eye(4)
    optical_frame_wrt_camera_link[:3, :3] = rz @ ry @ rx
    optical_frame_wrt_camera_link[:3, 3] = [pose["x_T"], pose["y_T"], pose["z_T"]]

    return optical_frame_wrt_camera_link


def validate_depth_image_dims(depth_mm: np.ndarray, depth_camera: CameraParameters) -> None:
    if depth_mm.shape != (depth_camera.height, depth_camera.width):
        raise ValueError(
            f"Depth frame is {depth_mm.shape[1]}x{depth_mm.shape[0]} but the depth "
            f"intrinsics are for {depth_camera.width}x{depth_camera.height}; the stream profile changed."
        )


def build_transform_matrix(
    translation_xyz: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    unit_quaternion = np.asarray(rotation_xyzw, dtype=np.float64) / np.linalg.norm(rotation_xyzw)
    quaternion_x, quaternion_y, quaternion_z, quaternion_w = unit_quaternion
    rotation = np.array(
        [
            [
                1 - 2 * (quaternion_y * quaternion_y + quaternion_z * quaternion_z),
                2 * (quaternion_x * quaternion_y - quaternion_z * quaternion_w),
                2 * (quaternion_x * quaternion_z + quaternion_y * quaternion_w),
            ],
            [
                2 * (quaternion_x * quaternion_y + quaternion_z * quaternion_w),
                1 - 2 * (quaternion_x * quaternion_x + quaternion_z * quaternion_z),
                2 * (quaternion_y * quaternion_z - quaternion_x * quaternion_w),
            ],
            [
                2 * (quaternion_x * quaternion_z - quaternion_y * quaternion_w),
                2 * (quaternion_y * quaternion_z + quaternion_x * quaternion_w),
                1 - 2 * (quaternion_x * quaternion_x + quaternion_y * quaternion_y),
            ],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_xyz
    return transform


def align_depth_to_rgb(
    depth_mm: np.ndarray,
    depth_camera: CameraParameters,
    rgb_camera: CameraParameters,
    depth_frame_wrt_rgb_frame: np.ndarray,
) -> np.ndarray:
    rows, columns = np.nonzero(depth_mm)
    depth_values = depth_mm[rows, columns].astype(np.float64)
    focal_x, focal_y = depth_camera.camera_matrix[0, 0], depth_camera.camera_matrix[1, 1]
    center_x, center_y = depth_camera.camera_matrix[0, 2], depth_camera.camera_matrix[1, 2]
    points_in_depth_frame = np.stack(
        [
            (columns - center_x) / focal_x * depth_values,
            (rows - center_y) / focal_y * depth_values,
            depth_values,
        ],
        axis=1,
    )

    rotation = depth_frame_wrt_rgb_frame[:3, :3]
    translation_mm = depth_frame_wrt_rgb_frame[:3, 3] * MILLIMETERS_PER_METER
    points_in_color_frame = points_in_depth_frame @ rotation.T + translation_mm
    in_front_of_camera = points_in_color_frame[:, 2] > 0
    points_in_color_frame = points_in_color_frame[in_front_of_camera]

    projected_pixels, _ = cv2.projectPoints(
        points_in_color_frame.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        rgb_camera.camera_matrix,
        rgb_camera.distortion_coefficients,
    )
    pixel_columns = np.rint(projected_pixels[:, 0, 0]).astype(np.int64)
    pixel_rows = np.rint(projected_pixels[:, 0, 1]).astype(np.int64)
    inside_image = (
        (pixel_columns >= 0)
        & (pixel_columns < rgb_camera.width)
        & (pixel_rows >= 0)
        & (pixel_rows < rgb_camera.height)
    )

    # Keep the nearest surface where several depth pixels land on one color pixel
    nearest_depth = np.full(rgb_camera.height * rgb_camera.width, np.inf)
    flat_pixel_index = pixel_rows[inside_image] * rgb_camera.width + pixel_columns[inside_image]
    np.minimum.at(nearest_depth, flat_pixel_index, points_in_color_frame[inside_image, 2])

    aligned_depth = np.where(np.isfinite(nearest_depth), np.rint(nearest_depth), 0)
    aligned_depth = np.clip(aligned_depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return aligned_depth.reshape(rgb_camera.height, rgb_camera.width)
