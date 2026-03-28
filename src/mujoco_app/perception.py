"""Open3D-based pose-estimation skeleton."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from mujoco_app.pose_types import Pose

if TYPE_CHECKING:
    from mujoco_app.mj_simulation import MjSim


def estimate_grasp_object_pose(
    sim: "MjSim",
    observation: Mapping[str, Any],
) -> Pose:
    """Estimate the grasp object's 6D pose from a camera observation.

    Expected observation keys:
    - `rgb`: uint8 RGB image, shape (H, W, 3)
    - `depth`: float depth image in meters, shape (H, W)
    - `intrinsic`: 3x3 intrinsic matrix
    - `extrinsic`: 4x4 world-to-camera matrix
    """
    body_name = sim.ids["grasp_object"]["body_name"]
    model_cloud = _load_object_model_point_cloud(sim)
    scene_cloud = _build_scene_point_cloud(observation)
    scene_cloud = _filter_scene_point_cloud(sim, scene_cloud, model_cloud)
    _save_debug_point_cloud(scene_cloud)
    transform = _estimate_transform_open3d(model_cloud, scene_cloud)
    return _pose_from_transform(body_name, transform)


def _load_object_model_point_cloud(sim: "MjSim") -> o3d.geometry.PointCloud:
    """Load point cloud for object of which we want to estimate the pose."""
    mesh_path = _resolve_object_mesh_path(sim)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    return mesh.sample_points_uniformly(number_of_points=5000)


def _resolve_object_mesh_path(sim: "MjSim") -> Path:
    """Create path to mesh of grasp object."""
    grasp_cfg = dict(sim.cfg.get("mujoco", {}).get("grasp_object", {}))
    xml_path = Path(grasp_cfg["xml"])
    if not xml_path.is_absolute():
        xml_path = Path.cwd() / xml_path
    return xml_path.resolve().parent / "textured.obj"


def _save_debug_point_cloud(point_cloud: o3d.geometry.PointCloud) -> Path:
    """Persist the filtered scene cloud for offline inspection."""
    output_path = Path.cwd() / "debug" / "scene_cloud_filtered.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), point_cloud)
    return output_path


def _build_scene_point_cloud(
    observation: Mapping[str, Any],
) -> o3d.geometry.PointCloud:
    """Build a world-frame point cloud from the RGB-D observation."""
    rgb = observation["rgb"]
    depth = observation["depth"]
    intrinsic = observation["intrinsic"]
    extrinsic = observation["extrinsic"]

    valid = np.isfinite(depth) & (depth > 0.0)

    rows, cols = np.nonzero(valid)
    depth_values = depth[rows, cols].astype(np.float64)

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])

    # MuJoCo cameras look along -Z, so positive depth values map to negative
    # camera-space Z before transforming into world coordinates.
    z = -depth_values
    x = (cols.astype(np.float64) - float(intrinsic[0, 2])) * z / fx
    y = (rows.astype(np.float64) - float(intrinsic[1, 2])) * z / fy
    camera_points = np.column_stack((x, y, z))

    homogeneous = np.concatenate(
        [camera_points, np.ones((len(camera_points), 1), dtype=np.float64)],
        axis=1,
    )
    world_points = (np.linalg.inv(extrinsic) @ homogeneous.T).T[:, :3]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(world_points)
    point_cloud.colors = o3d.utility.Vector3dVector(
        rgb[rows, cols].astype(np.float64) / 255.0
    )
    return point_cloud


def _filter_scene_point_cloud(
    sim: "MjSim",
    scene_cloud: o3d.geometry.PointCloud,
    model_cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """Reduce point cloud to workspace, remove table plane, select cluster which size best fits grasp object."""
    lower, upper = _workspace_bounds(sim, model_cloud)
    filtered = scene_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))

    table_top = _table_top(sim)
    z_clearance = max(0.006, 0.04 * float(_point_cloud_extent(model_cloud)[2]))
    points = np.asarray(filtered.points)
    keep = points[:, 2] >= (table_top + z_clearance)
    if np.count_nonzero(keep) >= 10:
        filtered = filtered.select_by_index(np.flatnonzero(keep).tolist())

    if len(filtered.points) >= 50:
        filtered = _remove_horizontal_plane(filtered, model_cloud)

    # TODO this is kind of brittle maybe remove later, sometimes selects the wrong cluster
    clustered = _select_best_cluster(filtered, model_cloud)
    if clustered is not None:
        filtered = clustered
    return filtered


def _workspace_bounds(
    sim: "MjSim",
    model_cloud: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative world-space bounds for the tabletop workspace."""
    table_cfg = dict(sim.cfg.get("table", {}))
    table_pos = np.asarray(table_cfg.get("pos", (0.6, 0.0, 0.7)), dtype=float)
    table_size = np.asarray(table_cfg.get("size", (0.65, 0.95, 0.025)), dtype=float)

    model_extent = _point_cloud_extent(model_cloud)
    xy_margin = max(0.08, 0.5 * float(np.max(model_extent[:2])) + 0.05)
    z_margin_above = max(0.25, 4.0 * float(model_extent[2]) + 0.05)

    lower = np.array(
        [
            table_pos[0] - table_size[0] - xy_margin,
            table_pos[1] - table_size[1] - xy_margin,
            _table_top(sim) - 0.03,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            table_pos[0] + table_size[0] + xy_margin,
            table_pos[1] + table_size[1] + xy_margin,
            _table_top(sim) + z_margin_above,
        ],
        dtype=float,
    )
    return lower, upper


def _table_top(sim: "MjSim") -> float:
    table_cfg = dict(sim.cfg.get("table", {}))
    table_pos = np.asarray(table_cfg.get("pos", (0.6, 0.0, 0.7)), dtype=float)
    table_size = np.asarray(table_cfg.get("size", (0.65, 0.95, 0.025)), dtype=float)
    return float(table_pos[2] + table_size[2])


def _remove_horizontal_plane(
    point_cloud: o3d.geometry.PointCloud,
    model_cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """Remove a dominant horizontal plane if Open3D finds one."""
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=max(0.004, 0.5 * _choose_voxel_size(model_cloud)),
        ransac_n=3,
        num_iterations=1000,
    )

    normal = np.asarray(plane_model[:3], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return point_cloud

    normal /= normal_norm
    if abs(float(np.dot(normal, np.array([0.0, 0.0, 1.0])))) < np.cos(np.deg2rad(25.0)):
        return point_cloud

    filtered = point_cloud.select_by_index(
        np.asarray(inliers, dtype=int).tolist(),
        invert=True,
    )
    if len(filtered.points) == 0:
        return point_cloud
    return filtered


def _select_best_cluster(
    point_cloud: o3d.geometry.PointCloud,
    model_cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud | None:
    """Pick the cluster whose size best matches the object model."""
    if len(point_cloud.points) < 10:
        return None

    labels = np.asarray(
        point_cloud.cluster_dbscan(
            eps=max(0.015, 2.5 * _choose_voxel_size(model_cloud)),
            min_points=8,
            print_progress=False,
        ),
        dtype=int,
    )
    model_max_extent = float(np.max(_point_cloud_extent(model_cloud)))

    best_cluster = None
    best_extent_error = np.inf
    for label in np.unique(labels):
        if label < 0:
            continue
        indices = np.flatnonzero(labels == label)
        if len(indices) < 10:
            continue

        cluster = point_cloud.select_by_index(indices.tolist())
        cluster_max_extent = float(np.max(_point_cloud_extent(cluster)))
        extent_error = abs(cluster_max_extent - model_max_extent)
        if extent_error < best_extent_error:
            best_extent_error = extent_error
            best_cluster = cluster
    return best_cluster


def _estimate_transform_open3d(
    model_cloud: o3d.geometry.PointCloud,
    scene_cloud: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Estimate the object pose using Open3D global registration followed by ICP."""
    voxel_size = _choose_voxel_size(model_cloud)
    model_down, model_fpfh = _preprocess_point_cloud(model_cloud, voxel_size)
    scene_down, scene_fpfh = _preprocess_point_cloud(scene_cloud, voxel_size)

    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh,
        mutual_filter=False,
        max_correspondence_distance=voxel_size * 2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                voxel_size * 2.0
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    # Refine the coarse pose with icp.
    refined_result = o3d.pipelines.registration.registration_icp(
        model_cloud,
        scene_cloud,
        voxel_size * 1.5,
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    return np.asarray(refined_result.transformation, dtype=float)


def _preprocess_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
):
    """preprocess point cloud according to https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#:~:text=return%20pcd%5Fdown%2C%20pcd%5Ffpfh"""
    point_cloud_down = point_cloud.voxel_down_sample(voxel_size)
    point_cloud_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    point_cloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return point_cloud_down, point_cloud_fpfh


def _choose_voxel_size(point_cloud: o3d.geometry.PointCloud) -> float:
    bounds_extent = _point_cloud_extent(point_cloud)
    max_extent = float(np.max(bounds_extent))
    return max(0.0025, max_extent / 30.0)


def _run_icp(
    model_cloud: o3d.geometry.PointCloud,
    scene_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    initial_transform: np.ndarray,
):
    return o3d.pipelines.registration.registration_icp(
        model_cloud,
        scene_cloud,
        voxel_size * 1.5,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

def _point_cloud_extent(point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(point_cloud.get_max_bound()) - np.asarray(
        point_cloud.get_min_bound()
    )


def _pose_from_transform(body_name: str, transform: np.ndarray) -> Pose:
    rotation = np.asarray(transform[:3, :3], dtype=float)
    position = np.asarray(transform[:3, 3], dtype=float)
    return Pose(
        body_name=body_name,
        position=position,
        quaternion_xyzw=Rotation.from_matrix(rotation).as_quat(),
    )
