"""Minimal GIGA grasp inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mujoco_app.pose_types import Pose

from vgn.detection_implicit import VGNImplicit
from vgn.perception import CameraIntrinsic, TSDFVolume
from vgn.utils.transform import Rotation, Transform


@dataclass
class GraspPrediction:
    pose: Pose
    width: float


def get_wrist_camera_observation(sim: Any) -> dict[str, Any]:
    """Render an observation from the configured wrist camera.

    The returned dict is directly compatible with
    `estimate_grasp_from_hand_camera`.
    """

    mujoco_cfg = dict(getattr(sim, "cfg", {}).get("mujoco", {}))
    base_camera_cfg = dict(mujoco_cfg.get("camera", {}))
    wrist_camera_cfg = dict(mujoco_cfg.get("wrist_camera", {}))

    if not wrist_camera_cfg.get("enable", True):
        raise ValueError("MuJoCo wrist camera is disabled in the config")

    camera_name = str(wrist_camera_cfg.get("name", "wrist_cam"))
    width = int(base_camera_cfg.get("width", 640))
    height = int(base_camera_cfg.get("height", 480))
    near = float(base_camera_cfg.get("near", 0.01))
    far = float(base_camera_cfg.get("far", 5.0))
    fovy = float(wrist_camera_cfg.get("fovy", base_camera_cfg.get("fovy", 58.0)))

    rgb, depth, intrinsic, extrinsic = sim.render_camera(
        camera_name,
        width=width,
        height=height,
        near=near,
        far=far,
        fovy=fovy,
    )
    return {
        "camera_name": camera_name,
        "camera_far": far,
        "rgb": rgb,
        "depth": depth,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
    }


def estimate_grasp_from_hand_camera(
    observation: dict[str, Any],
    object_pose: Pose,
    model_path: Path = Path(__file__).resolve().parents[2] / "giga/giga_pile.pt",
    tsdf_size: float = 0.30,
    tsdf_resolution: int = 40,
) -> GraspPrediction:
    """Use GIGA to calculate grasp

    Expected observation keys:
    - `depth`: float depth image in meters, shape (H, W)
    - `intrinsic`: 3x3 camera intrinsic matrix
    - `extrinsic`: 4x4 world-to-camera matrix
    """

    depth = np.asarray(observation["depth"], dtype=np.float32)
    intrinsic = np.asarray(observation["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(observation["extrinsic"], dtype=np.float64)

    if depth.ndim != 2:
        raise ValueError("Expected observation['depth'] to have shape (H, W)")
    if intrinsic.shape != (3, 3):
        raise ValueError("Expected observation['intrinsic'] to have shape (3, 3)")
    if extrinsic.shape != (4, 4):
        raise ValueError("Expected observation['extrinsic'] to have shape (4, 4)")

    planner = VGNImplicit(
        model_path,
        "giga",
        best=True,
        force_detection=True,
        qual_th=0.9,
        resolution=tsdf_resolution,
    )

    camera_intrinsic = CameraIntrinsic(
        width=depth.shape[1],
        height=depth.shape[0],
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2],
    )
    tsdf = TSDFVolume(size=tsdf_size, resolution=tsdf_resolution)

    cube_origin = np.asarray(object_pose.position, dtype=np.float64) - (
        tsdf_size / 2.0
    )
    world_from_tsdf = Transform(Rotation.identity(), cube_origin)
    camera_from_world = Transform.from_matrix(extrinsic)
    camera_from_tsdf = camera_from_world * world_from_tsdf
    tsdf.integrate(depth, camera_intrinsic, camera_from_tsdf)

    grasps, scores, _ = planner(SimpleNamespace(tsdf=tsdf))
    if len(grasps) == 0:
        raise ValueError("GIGA did not return any valid grasp")

    best_grasp = grasps[0]
    world_rotation = world_from_tsdf.rotation * best_grasp.pose.rotation
    world_position = world_from_tsdf.transform_point(best_grasp.pose.translation)

    grasp_pose = Pose(
        body_name=object_pose.body_name,
        position=np.asarray(world_position, dtype=float),
        quaternion_xyzw=world_rotation.as_quat(),
    )
    return GraspPrediction(
        pose=grasp_pose,
        width=float(best_grasp.width),
    )
