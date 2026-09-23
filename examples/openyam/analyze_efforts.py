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

"""Compare recorded OpenYAM joint efforts with URDF gravity and a wrist wrench.

Gravity torques come from Pinocchio RNEA on the right arm. The residual
(measured effort minus gravity, and measured effort minus full inverse
dynamics) is mapped to a 6-D wrench at ``follower_r_link_6``. The wrench uses
Pinocchio ``LOCAL_WORLD_ALIGNED``: origin at the wrist, axes aligned with the
world, so ``fz`` stays vertical.

The cameras and state/action plots are the same ones ``lerobot-dataset-viz``
logs. Wrench and gravity series are added on that timeline.
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import (
    DEFAULT_RERUN_PORT,
    _as_2d_float,
    build_blueprint_from_dataset,
    get_feature_names,
    infer_repo_id,
    log_episode_scalars,
    log_episode_videos,
    resolve_dataset_root,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "/home/fuheng/test_data/car-door-opening-20260910"
DEFAULT_URDF = "/home/fuheng/anvil-loader/config/robot_description.urdf"
ARM_JOINTS = tuple(f"follower_r_joint{i}" for i in range(1, 7))
WRIST_FRAME = "follower_r_link_6"
JOINT_LABELS = tuple(f"right_joint{i}" for i in range(1, 7))
FORCE_LABELS = ("fx", "fy", "fz")
TORQUE_LABELS = ("tx", "ty", "tz")


class RightArmModel:
    """Right-arm gravity, inverse dynamics, and wrist Jacobian from a URDF."""

    def __init__(self, urdf_path: Path) -> None:
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.q_idx = np.array([self._joint_q(name) for name in ARM_JOINTS], dtype=int)
        self.v_idx = np.array([self._joint_v(name) for name in ARM_JOINTS], dtype=int)
        self.wrist_id = self.model.getFrameId(WRIST_FRAME)
        if self.wrist_id >= len(self.model.frames):
            raise ValueError(f"URDF has no frame {WRIST_FRAME}")

    def _joint_id(self, name: str) -> int:
        joint_id = self.model.getJointId(name)
        if joint_id >= self.model.njoints or self.model.names[joint_id] != name:
            raise ValueError(f"URDF has no joint {name}")
        return joint_id

    def _joint_q(self, name: str) -> int:
        return int(self.model.idx_qs[self._joint_id(name)])

    def _joint_v(self, name: str) -> int:
        return int(self.model.idx_vs[self._joint_id(name)])

    def evaluate(
        self, q_arm: np.ndarray, v_arm: np.ndarray, a_arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return gravity torque, inverse-dynamics torque, and the wrist Jacobian.

        ``q_arm``, ``v_arm``, and ``a_arm`` each have shape ``(6,)``. Torques have
        shape ``(6,)``. The Jacobian has shape ``(6, 6)`` and maps arm joint
        velocity to a world-aligned spatial velocity at ``follower_r_link_6``.
        """
        q = pin.neutral(self.model)
        v = np.zeros(self.model.nv)
        a = np.zeros(self.model.nv)
        q[self.q_idx] = q_arm
        v[self.v_idx] = v_arm
        a[self.v_idx] = a_arm

        zeros = np.zeros(self.model.nv)
        tau_g = np.asarray(pin.rnea(self.model, self.data, q, zeros, zeros), dtype=np.float64)[self.v_idx]
        tau_dyn = np.asarray(pin.rnea(self.model, self.data, q, v, a), dtype=np.float64)[self.v_idx]
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian = pin.getFrameJacobian(
            self.model, self.data, self.wrist_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        jacobian_arm = np.asarray(jacobian, dtype=np.float64)[:, self.v_idx]
        return tau_g, tau_dyn, jacobian_arm


def state_arm_indices(names: list[str], suffix: str) -> list[int]:
    """Column indices of ``right_joint1..6.{suffix}`` inside ``observation.state``."""
    indices = []
    for label in JOINT_LABELS:
        key = f"{label}.{suffix}"
        if key not in names:
            raise ValueError(f"observation.state is missing {key}")
        indices.append(names.index(key))
    return indices


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_center = a - a.mean()
    b_center = b - b.mean()
    denom = float(np.linalg.norm(a_center) * np.linalg.norm(b_center))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_center, b_center) / denom)


def choose_effort_sign(tau_meas: np.ndarray, tau_g: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the sign that makes measured effort track gravity, and the raw correlations.

    Motor torque that holds the arm against gravity has the same sign as RNEA.
    A negative mean correlation flips the recorded effort.
    """
    raw = np.array([pearson(tau_meas[:, i], tau_g[:, i]) for i in range(tau_meas.shape[1])])
    sign = -1.0 if float(raw.mean()) < 0.0 else 1.0
    return sign, raw


def wrench_from_torque(tau: np.ndarray, jacobians: np.ndarray) -> np.ndarray:
    """Least-squares wrench at the wrist, ``tau ≈ J.T @ wrench``."""
    wrench = np.zeros_like(tau)
    for i in range(tau.shape[0]):
        solved, _, _, _ = np.linalg.lstsq(jacobians[i].T, tau[i], rcond=None)
        wrench[i] = solved
    return wrench


def analyze_episode(dataset: LeRobotDataset, urdf_path: Path) -> dict[str, np.ndarray | float]:
    """Compute gravity, residuals, and wrist wrenches for the loaded episode."""
    hf = dataset.hf_dataset.with_format("numpy")
    state = _as_2d_float(hf[OBS_STATE])
    names = get_feature_names(dataset, OBS_STATE)
    q = state[:, state_arm_indices(names, "position")]
    v = state[:, state_arm_indices(names, "velocity")]
    tau_meas = state[:, state_arm_indices(names, "effort")]
    fps = float(dataset.fps)
    if fps <= 0.0:
        raise ValueError(f"Dataset fps must be positive, got {dataset.fps}")
    acc = np.gradient(v, 1.0 / fps, axis=0)

    arm = RightArmModel(urdf_path)
    n = q.shape[0]
    tau_g = np.zeros((n, 6))
    tau_dyn = np.zeros((n, 6))
    jacobians = np.zeros((n, 6, 6))
    for i in range(n):
        tau_g[i], tau_dyn[i], jacobians[i] = arm.evaluate(q[i], v[i], acc[i])

    sign, raw_corr = choose_effort_sign(tau_meas, tau_g)
    tau_signed = sign * tau_meas
    residual_g = tau_signed - tau_g
    residual_d = tau_signed - tau_dyn
    signed_corr = np.array([pearson(tau_signed[:, i], tau_g[:, i]) for i in range(6)])
    wrench_g = wrench_from_torque(residual_g, jacobians)
    wrench_d = wrench_from_torque(residual_d, jacobians)

    timestamps = np.asarray(hf["timestamp"], dtype=np.float64).reshape(-1)
    frame_index = np.asarray(hf["frame_index"], dtype=np.int64).reshape(-1)
    return {
        "sign": sign,
        "raw_corr": raw_corr,
        "signed_corr": signed_corr,
        "tau_signed": tau_signed,
        "tau_g": tau_g,
        "residual_g": residual_g,
        "residual_d": residual_d,
        "wrench_g": wrench_g,
        "wrench_d": wrench_d,
        "speed": np.linalg.norm(v, axis=1),
        "timestamps": timestamps,
        "frame_index": frame_index,
    }


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values**2, axis=0))


def log_report(result: dict[str, np.ndarray | float]) -> None:
    """Print per-joint correlation, residual RMS, and low- vs high-speed force."""
    sign = float(result["sign"])
    signed_corr = np.asarray(result["signed_corr"])
    tau_g = np.asarray(result["tau_g"])
    residual_g = np.asarray(result["residual_g"])
    residual_d = np.asarray(result["residual_d"])
    force_g = np.linalg.norm(np.asarray(result["wrench_g"])[:, :3], axis=1)
    force_d = np.linalg.norm(np.asarray(result["wrench_d"])[:, :3], axis=1)
    speed = np.asarray(result["speed"])
    split = float(np.median(speed))
    slow = speed <= split
    fast = ~slow

    logger.info("Wrist wrench frame: %s, LOCAL_WORLD_ALIGNED (world axes, fz up)", WRIST_FRAME)
    logger.info("Effort sign applied to the recording: %+.0f", sign)
    corr_bits = []
    for label, corr, gravity in zip(JOINT_LABELS, signed_corr, tau_g.T, strict=True):
        if float(np.std(gravity)) < 1e-6:
            corr_bits.append(f"{label}=n/a (no gravity torque)")
        else:
            corr_bits.append(f"{label}={corr:+.3f}")
    logger.info("Per-joint correlation of signed effort with gravity: %s", ", ".join(corr_bits))
    logger.info(
        "RMS residual after gravity (N·m): %s  overall=%.3f",
        ", ".join(
            f"{label}={value:.3f}" for label, value in zip(JOINT_LABELS, _rms(residual_g), strict=True)
        ),
        float(np.sqrt(np.mean(residual_g**2))),
    )
    logger.info(
        "RMS residual after inverse dynamics (N·m): %s  overall=%.3f",
        ", ".join(
            f"{label}={value:.3f}" for label, value in zip(JOINT_LABELS, _rms(residual_d), strict=True)
        ),
        float(np.sqrt(np.mean(residual_d**2))),
    )
    logger.info(
        "Mean |F| split at joint-speed median %.3f rad/s (%d slow / %d fast frames)",
        split,
        int(slow.sum()),
        int(fast.sum()),
    )
    logger.info(
        "Gravity residual |F|: slow=%.2f N  fast=%.2f N",
        float(force_g[slow].mean()) if slow.any() else float("nan"),
        float(force_g[fast].mean()) if fast.any() else float("nan"),
    )
    logger.info(
        "Dynamics residual |F|: slow=%.2f N  fast=%.2f N",
        float(force_d[slow].mean()) if slow.any() else float("nan"),
        float(force_d[fast].mean()) if fast.any() else float("nan"),
    )


def _log_matrix(entity: str, values: np.ndarray, names: tuple[str, ...], indexes: list) -> None:
    import rerun as rr

    grouped = np.ascontiguousarray(values, dtype=np.float64)
    n, dim = grouped.shape
    columns = rr.Scalars.columns(scalars=grouped.reshape(-1))
    if dim > 1:
        columns = columns.partition([dim] * n)
    rr.log(entity, rr.SeriesLines(names=list(names)), static=True)
    rr.send_columns(entity, indexes=indexes, columns=columns)


def _timeseries(entity: str, names: tuple[str, ...]):
    import rerun as rr
    import rerun.blueprint as rrb

    styling = rr.SeriesLines(names=list(names))
    return rrb.TimeSeriesView(origin=entity, name=entity, overrides={entity: styling})


def visualize_analysis(
    dataset: LeRobotDataset,
    result: dict[str, np.ndarray | float],
    episode_index: int,
    mode: str,
    web_port: int | None,
    grpc_port: int,
) -> None:
    """Log the dataset plus gravity and wrench series, then serve the Rerun viewer."""
    from lerobot.utils.import_utils import require_package

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    if mode not in ("local", "distant"):
        raise ValueError(mode)

    import rerun.blueprint as rrb

    series = (
        ("gravity/measured", np.asarray(result["tau_signed"]), JOINT_LABELS),
        ("gravity/model", np.asarray(result["tau_g"]), JOINT_LABELS),
        ("residual/gravity", np.asarray(result["residual_g"]), JOINT_LABELS),
        ("residual/dynamics", np.asarray(result["residual_d"]), JOINT_LABELS),
        ("wrench/force", np.asarray(result["wrench_g"])[:, :3], FORCE_LABELS),
        ("wrench/torque", np.asarray(result["wrench_g"])[:, 3:], TORQUE_LABELS),
        ("wrench/force_dynamics", np.asarray(result["wrench_d"])[:, :3], FORCE_LABELS),
    )
    by_entity = {entity: _timeseries(entity, names) for entity, _, names in series}
    dataset_blueprint = build_blueprint_from_dataset(dataset)
    dataset_views = list(dataset_blueprint.root_container.contents)
    cameras = [view for view in dataset_views if isinstance(view, rrb.Spatial2DView)]
    scalars = [view for view in dataset_views if not isinstance(view, rrb.Spatial2DView)]
    # The dataset grid is wide enough that appending torque puts it below the fold.
    # Keep force and torque on their own row under the cameras.
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*cameras),
            rrb.Horizontal(
                by_entity["wrench/force"],
                by_entity["wrench/torque"],
                by_entity["wrench/force_dynamics"],
            ),
            rrb.Grid(
                by_entity["gravity/measured"],
                by_entity["gravity/model"],
                by_entity["residual/gravity"],
                by_entity["residual/dynamics"],
                *scalars,
            ),
            row_shares=[2, 2, 3],
        )
    )

    logger.info("Starting Rerun (%d camera views, torque row is under the cameras)", len(cameras))
    # A new application id so the viewer does not reuse the blueprint saved for the previous session.
    app_id = f"{dataset.repo_id}/episode_{episode_index}/wrist"
    rr.init(app_id, default_blueprint=blueprint)
    gc.collect()

    local_web_port = web_port if web_port is not None else DEFAULT_RERUN_PORT
    server_uri = rr.serve_grpc(grpc_port=grpc_port, server_memory_limit="2GiB", default_blueprint=blueprint)
    if mode == "distant":
        logger.info("Connect with: rerun rerun+http://IP:%s/proxy", grpc_port)
        rr.serve_web_viewer(open_browser=False, web_port=local_web_port, connect_to=server_uri)

    logger.info("Logging to Rerun")
    log_episode_videos(dataset, episode_index)
    log_episode_scalars(dataset)
    indexes = [
        rr.TimeColumn("timestamp", duration=np.asarray(result["timestamps"])),
        rr.TimeColumn("frame_index", sequence=np.asarray(result["frame_index"])),
    ]
    for entity, values, names in series:
        _log_matrix(entity, values, names, indexes)

    recording = rr.get_data_recording()
    if recording is not None:
        recording.flush()
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    logger.info("Active blueprint includes wrench/torque. Recording: %s", app_id)

    if mode == "local":
        logger.info("Opening Rerun in the browser at http://127.0.0.1:%s", local_web_port)
        rr.serve_web_viewer(open_browser=True, web_port=local_web_port, connect_to=server_uri)

    logger.info("Logged episode %s. Viewer is open — press play in Rerun, Ctrl-C to exit.", episode_index)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Ctrl-C received. Exiting.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Local LeRobot dataset directory.")
    parser.add_argument(
        "--urdf", type=Path, default=Path(DEFAULT_URDF), help="OpenYAM URDF with inertial parameters."
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Episode to analyze and visualize.")
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="'local' opens the Rerun web viewer. 'distant' serves a gRPC proxy for a remote viewer.",
    )
    parser.add_argument("--web-port", type=int, default=None, help="Rerun web viewer port (default 9090).")
    parser.add_argument("--grpc-port", type=int, default=9876, help="Rerun gRPC port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logging()
    root = resolve_dataset_root(args.root)
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"No URDF at {urdf_path}")

    repo_id = infer_repo_id(None, root)
    logger.info("Loading dataset %s episode %s", root, args.episode_index)
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=1e-4)
    logger.info("Computing gravity and wrist wrench from %s", urdf_path)
    result = analyze_episode(dataset, urdf_path)
    log_report(result)
    visualize_analysis(
        dataset,
        result,
        episode_index=args.episode_index,
        mode=args.mode,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
    )


if __name__ == "__main__":
    main()
