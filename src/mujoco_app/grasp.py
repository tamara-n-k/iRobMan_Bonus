"""Top-down mesh-based grasp estimation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from mujoco_app.pose_types import Pose
from mujoco_app.transformations import quat_xyzw_to_matrix

if TYPE_CHECKING:
    from mujoco_app.mj_simulation import MjSim


DEFAULT_ANGLE_STEP_DEG = 1.0
MIN_GRASP_CLEARANCE_FROM_BOTTOM = 0.001
MAX_GRASP_DEPTH_FROM_TOP = 0.1
DEFAULT_GRIPPER_SITE_NAME = "gripper"


def estimate_top_down_grasp(
    sim: "MjSim",
    object_pose: Pose,
) -> Pose:
    """Estimate a top-down grasp from the configured object mesh.

    The mesh is transformed into world coordinates using the provided object
    pose. The top view is then scanned over yaw angles until the minimum projected grasp width is
    found.
    """
    grasp_body_name = object_pose.body_name

    mesh_path = _resolve_object_mesh_path(sim)
    mesh_vertices = _load_mesh_vertices(mesh_path)
    world_vertices = _transform_vertices(
        mesh_vertices,
        object_pose.position,
        object_pose.quaternion_xyzw,
    )

    center_xy = _top_view_center(world_vertices[:, :2])
    z_position = _grasp_z_position(world_vertices[:, 2])
    yaw_rad = _find_min_width_yaw(
        world_vertices[:, :2],
        center_xy,
        DEFAULT_ANGLE_STEP_DEG,
    )

    grasp_center_position = np.array(
        [center_xy[0], center_xy[1], z_position],
        dtype=float,
    )
    grasp_pose = Pose(
        body_name=grasp_body_name,
        position=_controller_target_position(
            sim,
            grasp_center_position,
            _top_down_quaternion_xyzw(yaw_rad),
        ),
        quaternion_xyzw=_top_down_quaternion_xyzw(yaw_rad),
    )
    return grasp_pose


def _resolve_object_mesh_path(sim: "MjSim") -> Path:
    grasp_cfg = dict(sim.cfg.get("mujoco", {}).get("grasp_object", {}))
    xml_path = Path(grasp_cfg["xml"])
    if not xml_path.is_absolute():
        xml_path = Path.cwd() / xml_path
    xml_path = xml_path.resolve()

    default_mesh = xml_path.parent / "textured.obj"
    if default_mesh.exists():
        return default_mesh

    root = ET.parse(xml_path).getroot()
    for mesh_node in root.findall(".//mesh"):
        mesh_file = mesh_node.get("file")
        if not mesh_file:
            continue
        candidate = (xml_path.parent / mesh_file).resolve()
        if candidate.suffix.lower() == ".obj" and candidate.exists():
            return candidate

    raise FileNotFoundError(f"No object mesh could be resolved from {xml_path}")


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Mesh at {mesh_path} does not contain valid vertices")
    return vertices


def _transform_vertices(
    vertices: np.ndarray,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    rotation = quat_xyzw_to_matrix(quaternion_xyzw)
    return (vertices @ rotation.T) + np.asarray(position, dtype=float)


def _top_view_center(projected_vertices: np.ndarray) -> np.ndarray:
    mins = np.min(projected_vertices, axis=0)
    maxs = np.max(projected_vertices, axis=0)
    return 0.5 * (mins + maxs)


def _grasp_z_position(z_values: np.ndarray) -> float:
    """Choose the lowest mesh-only grasp height that keeps simple clearances."""
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    return max(
        z_min + MIN_GRASP_CLEARANCE_FROM_BOTTOM,
        z_max - MAX_GRASP_DEPTH_FROM_TOP,
    )


def _controller_target_position(
    sim: "MjSim",
    grasp_center_position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    ee_offset_local = _end_effector_frame_offset(sim)
    if ee_offset_local is None:
        return grasp_center_position.copy()
    rotation = quat_xyzw_to_matrix(quaternion_xyzw)
    return grasp_center_position - rotation @ ee_offset_local


def _end_effector_frame_offset(sim: "MjSim") -> np.ndarray | None:
    ee_body_name = sim.robot_settings.get("ee_body_name", "hand")
    ee_body_id = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_BODY,
        ee_body_name,
    )
    if ee_body_id < 0:
        return None

    gripper_site_id = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_SITE,
        DEFAULT_GRIPPER_SITE_NAME,
    )
    if (
        gripper_site_id >= 0
        and int(sim.model.site_bodyid[gripper_site_id]) == ee_body_id
    ):
        return np.asarray(sim.model.site_pos[gripper_site_id], dtype=float)

    finger_offsets = []
    for finger_body_name in ("left_finger", "right_finger"):
        finger_body_id = mujoco.mj_name2id(
            sim.model,
            mujoco.mjtObj.mjOBJ_BODY,
            finger_body_name,
        )
        if finger_body_id < 0:
            continue
        if int(sim.model.body_parentid[finger_body_id]) != ee_body_id:
            continue
        finger_offsets.append(
            np.asarray(sim.model.body_pos[finger_body_id], dtype=float)
        )
    if finger_offsets:
        return np.mean(np.asarray(finger_offsets, dtype=float), axis=0)

    return None


def _find_min_width_yaw(
    projected_vertices: np.ndarray,
    center_xy: np.ndarray,
    angle_step_deg: float,
) -> float:
    centered = projected_vertices - center_xy
    angles = np.deg2rad(np.arange(0.0, 180.0, angle_step_deg, dtype=float))

    best_yaw = None
    best_width = np.inf
    best_span = -np.inf
    for yaw_rad in angles:
        finger_axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=float)
        closing_axis = np.array([-finger_axis[1], finger_axis[0]], dtype=float)
        width = _axis_extent(centered, closing_axis)
        span = _axis_extent(centered, finger_axis)
        if width < best_width - 1e-9 or (
            np.isclose(width, best_width) and span > best_span
        ):
            best_yaw = float(yaw_rad)
            best_width = float(width)
            best_span = float(span)

    if best_yaw is None:
        raise ValueError("Failed to determine a top-down grasp yaw")

    return best_yaw


def _axis_extent(points: np.ndarray, axis: np.ndarray) -> float:
    projections = points @ axis
    return float(np.max(projections) - np.min(projections))


def _top_down_quaternion_xyzw(yaw_rad: float) -> np.ndarray:
    rotation = Rotation.from_euler("z", yaw_rad) * Rotation.from_euler(
        "x", np.pi
    )
    return rotation.as_quat().astype(float)
