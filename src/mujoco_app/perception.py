"""Open3D-based pose-estimation skeleton."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mujoco_app.pose_types import Pose
import open3d as o3d
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from mujoco_app.mj_simulation import MjSim


def estimate_grasp_object_pose(
    sim: "MjSim",
    observation: Mapping[str, Any],
) -> Pose:
    """Estimates the grasp object's 6D pose from an observation.

    Expected observation keys:
    - `depth`: float depth image in meters, shape (H, W)
    - `intrinsic`: 3x3 intrinsic matrix
    - `extrinsic`: 4x4 world-to-camera matrix
    - `mask`: bool array for the grasp object
    """
    body_name = sim.ids["grasp_object"]["body_name"]
    model_cloud = _load_object_model_point_cloud(sim)
    scene_cloud = _build_scene_point_cloud(observation)
    transform = _estimate_transform_open3d(model_cloud, scene_cloud)
    return _pose_from_transform(body_name, transform)


def _load_object_model_point_cloud(sim: "MjSim"):
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


def _build_scene_point_cloud(observation: Mapping[str, Any]):
    """Build point cloud of current scene."""
    rgb = observation["rgb"]
    depth = observation["depth"]
    intrinsic = observation["intrinsic"]
    extrinsic = observation["extrinsic"]
    mask = observation["mask"]

    # Keep only valid depth values that belong to the target region.
    valid = np.isfinite(depth) & (depth > 0.0) & mask
    masked_depth = np.zeros_like(depth, dtype=np.float32)
    masked_depth[valid] = depth[valid]
    masked_rgb = np.zeros_like(rgb, dtype=np.uint8)
    masked_rgb[valid] = rgb[valid]

    # Read the pinhole-camera parameters from the intrinsic matrix.
    focal_length_x = float(abs(intrinsic[0, 0]))
    focal_length_y = float(abs(intrinsic[1, 1]))
    principal_point_x = float(intrinsic[0, 2])
    principal_point_y = float(intrinsic[1, 2])
    image_height, image_width = depth.shape

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width,
        image_height,
        focal_length_x,
        focal_length_y,
        principal_point_x,
        principal_point_y,
    )

    # Back-project the masked RGB-D image into a 3D point cloud.
    color_image = o3d.geometry.Image(masked_rgb)
    depth_image = o3d.geometry.Image(masked_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1.0,
        depth_trunc=float(np.nanmax(masked_depth[valid])) + 1e-6,
        convert_rgb_to_intensity=False,
    )
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d_intrinsic,
    )

    # Convert from camera coordinates to world coordinates.
    point_cloud.transform(np.linalg.inv(extrinsic))
    return point_cloud


def _estimate_transform_open3d(model_cloud, scene_cloud) -> np.ndarray:
    """Pose estimation using RANSAC and ICP, following Open3D tutorial."""
    voxel_size = _choose_voxel_size(model_cloud)

    # preprocess point clouds
    model_down, model_fpfh = _preprocess_point_cloud(model_cloud, voxel_size)
    scene_down, scene_fpfh = _preprocess_point_cloud(scene_cloud, voxel_size)

    # Find a coarse alignment from feature correspondences.
    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh,
        mutual_filter=True,
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

    # Refine the coarse pose on the full-resolution clouds.
    refined_result = o3d.pipelines.registration.registration_icp(
        model_cloud,
        scene_cloud,
        voxel_size * 1.5,
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return np.asarray(refined_result.transformation, dtype=float)


def _preprocess_point_cloud(point_cloud, voxel_size):
    """preprocess point cloud according to https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#:~:text=return%20pcd%5Fdown%2C%20pcd%5Ffpfh"""
    point_cloud_down = point_cloud.voxel_down_sample(voxel_size)
    point_cloud_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    point_cloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    return point_cloud_down, point_cloud_fpfh


def _choose_voxel_size(point_cloud) -> float:
    bounds_extent = np.asarray(point_cloud.get_max_bound()) - np.asarray(
        point_cloud.get_min_bound()
    )
    max_extent = float(np.max(bounds_extent))
    return max(0.005, max_extent / 30.0)


def _pose_from_transform(body_name: str, transform: np.ndarray) -> Pose:
    rotation = np.asarray(transform[:3, :3], dtype=float)
    position = np.asarray(transform[:3, 3], dtype=float)
    return Pose(
        body_name=body_name,
        position=position,
        quaternion_xyzw=Rotation.from_matrix(rotation).as_quat(),
    )
