import argparse
import time
from typing import Any, Dict

import numpy as np
import yaml

from mujoco_app.ground_truth import get_body_pose_ground_truth
from mujoco_app.mj_simulation import MjSim
from mujoco_app.perception import estimate_grasp_object_pose
from mujoco_app.pose_types import Pose


def normalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return np.asarray(quaternion_xyzw, dtype=float) / norm


def pose_error(
    estimate: Pose, ground_truth: Pose
) -> dict[str, float]:
    position_error = float(
        np.linalg.norm(estimate.position - ground_truth.position)
    )
    estimate_quat = normalize_quaternion(estimate.quaternion_xyzw)
    ground_truth_quat = normalize_quaternion(ground_truth.quaternion_xyzw)
    dot = float(np.clip(np.abs(np.dot(estimate_quat, ground_truth_quat)), 0.0, 1.0))
    orientation_error_deg = float(np.degrees(2.0 * np.arccos(dot)))
    return {
        "position_error_m": position_error,
        "orientation_error_deg": orientation_error_deg,
    }


def select_camera_name(sim: MjSim) -> str:
    if "side_cam" in sim.extra_cameras:
        return "side_cam"
    return sim.ids.get("cam_name", "static")


def build_observation(
    sim: MjSim,
    config: Dict[str, Any],
    camera_name: str,
) -> Dict[str, Any]:
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    near = float(cam_cfg.get("near", 0.01))
    far = float(cam_cfg.get("far", 5.0))
    fovy = float(cam_cfg.get("fovy", 58.0))
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
        "rgb": rgb,
        "depth": depth,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "mask": depth > 0.0,
    }


def grasp_body_name(sim: MjSim) -> str:
    return sim.ids["grasp_object"]["body_name"]


def run_realtime_loop(sim: MjSim) -> None:
    timestep = float(sim.model.opt.timestep)
    start_time = time.perf_counter()
    step_idx = 0

    print("Running in real time. Close the viewer or press Ctrl+C to stop.")
    while sim._viewer is not None and sim._viewer.is_running():
        sim.step()
        step_idx += 1
        target_elapsed = step_idx * timestep
        wall_elapsed = time.perf_counter() - start_time
        remaining = target_elapsed - wall_elapsed
        if remaining > 0.0:
            time.sleep(remaining)


def compare_estimates(
    config: Dict[str, Any],
    stabilization_steps: int,
    realtime: bool,
) -> None:
    sim = MjSim(config)
    try:
        sim.reset()
        for _ in range(stabilization_steps):
            sim.step()

        camera_name = select_camera_name(sim)
        observation = build_observation(sim, config, camera_name)
        body_name = grasp_body_name(sim)

        print(f"Perception comparison using camera '{camera_name}':")
        estimate = estimate_grasp_object_pose(sim, observation)
        ground_truth = get_body_pose_ground_truth(sim, body_name)
        error = pose_error(estimate, ground_truth)

        print(f"  {body_name}:")
        print(
            "    estimate pos={} quat={}".format(
                np.round(estimate.position, 4).tolist(),
                np.round(estimate.quaternion_xyzw, 4).tolist(),
            )
        )
        print(
            "    ground truth pos={} quat={}".format(
                np.round(ground_truth.position, 4).tolist(),
                np.round(ground_truth.quaternion_xyzw, 4).tolist(),
            )
        )
        print(
            "    errors: position={:.4f} m, orientation={:.2f} deg".format(
                error["position_error_m"],
                error["orientation_error_deg"],
            )
        )

        if realtime:
            if not bool(config.get("mujoco", {}).get("gui", False)):
                print("Realtime requested, but `mujoco.gui` is false in the config.")
                return
            run_realtime_loop(sim)
        elif bool(config.get("mujoco", {}).get("gui", False)):
            print("Realtime mode disabled. Viewer will close after reporting.")
    finally:
        sim.close()


def main(config_path: str, stabilization_steps: int, realtime: bool) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    compare_estimates(config, stabilization_steps, realtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_config_mj.yaml",
    )
    parser.add_argument(
        "--stabilization-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Keep stepping the simulation in wall-clock real time after the comparison.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        stabilization_steps=args.stabilization_steps,
        realtime=args.realtime,
    )
