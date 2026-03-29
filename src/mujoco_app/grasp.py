"""Minimal GIGA grasp inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mujoco_app.pose_types import Pose

from giga.detection_implicit import gigaImplicit
from giga.perception import CameraIntrinsic, TSDFVolume
from giga.utils.transform import Rotation, Transform


@dataclass
class GraspPrediction:
    pose: Pose
    width: float


def estimate_grasp_from_hand_camera(
    observation: dict[str, Any],
    object_pose: Pose,
    model_path: Path = Path(__file__).resolve().parents[2] / "giga_models/giga_pile.pt",
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

    planner = gigaImplicit(
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
