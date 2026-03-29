"""
Enhanced viewer that logs sensor data during simulation.
Usage: python view_object_with_sensors.py YcbBanana [--save-sensors]
"""
from math import dist
import math
import time
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from fsspec import config
import mujoco
from mujoco import viewer
import numpy as np
import yaml

from mujoco_app.mj_simulation import MjSim
from mujoco_app.ik_solver import IKSolver
from mujoco_app.robot_controller import RobotController


def check_object_in_basket(sim: MjSim) -> dict:
    """Check if the grasp object is inside the basket.

    Returns a dict with keys:
        - in_basket: bool - True if object is in basket
        - in_x, in_y, in_z: bool - Individual bound checks
        - object_pos: list - [x, y, z] position of object
    """
    # Get basket parameters
    basket_center = sim.ids.get("basket_center")
    basket_dims = sim.ids.get("basket_dims")
    basket_height = sim.ids.get("basket_height")

    # Default if basket not configured
    if basket_center is None or basket_dims is None or basket_height is None:
        return {
            "in_basket": False,
            "in_x": False,
            "in_y": False,
            "in_z": False,
            "object_pos": None,
        }

    # Get grasp object body ID
    grasp_obj_info = sim.ids.get("grasp_object", {})
    obj_body_name = grasp_obj_info.get("body_name", "sample_object")

    obj_body_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name
    )

    if obj_body_id < 0:
        return {
            "in_basket": False,
            "in_x": False,
            "in_y": False,
            "in_z": False,
            "object_pos": None,
        }

    # Get object position
    obj_pos = sim.data.xpos[obj_body_id].copy()

    # Check bounds
    in_x = abs(obj_pos[0] - basket_center[0]) < basket_dims[0] / 2.0
    in_y = abs(obj_pos[1] - basket_center[1]) < basket_dims[1] / 2.0
    basket_bottom = basket_center[2] - basket_height / 2.0
    in_z = obj_pos[2] > basket_bottom

    return {
        "in_basket": in_x and in_y and in_z,
        "in_x": in_x,
        "in_y": in_y,
        "in_z": in_z,
        "object_pos": obj_pos.tolist(),
    }


class SensorLogger:
    """Logs sensor data during simulation."""

    def __init__(self, sim: MjSim, log_dir: Path):
        self.sim = sim
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.camera_log = []
        self.joint_log = []
        self.ee_log = []
        self.contact_log = []
        self.basket_log = []

    def log_frame(self, save_images=False):
        """Log sensor data for current frame."""
        import mujoco

        # Camera data (RGB, depth from primary camera)
        cam_cfg = self.sim.cfg.get("mujoco", {}).get("camera", {})
        width = cam_cfg.get("width", 640)
        height = cam_cfg.get("height", 480)
        near = cam_cfg.get("near", 0.01)
        far = cam_cfg.get("far", 5.0)
        fovy = cam_cfg.get("fovy", 58.0)

        rgb, depth, intrinsic, extrinsic = self.sim.get_static_renders(
            width=width, height=height, near=near, far=far, fovy=fovy
        )

        # Joint states
        arm_positions = [
            self.sim.data.qpos[qidx] for qidx, _ in self.sim.robot.arm_pairs
        ]
        arm_velocities = [
            self.sim.data.qvel[vidx] for _, vidx in self.sim.robot.arm_pairs
        ]

        # End-effector pose
        ee_body_name = self.sim.robot_settings.get("ee_body_name", "hand")
        ee_body_id = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )
        ee_pos = self.sim.data.xpos[ee_body_id].copy()
        ee_rot = self.sim.data.xmat[ee_body_id].reshape(3, 3).copy()

        # Contacts
        contacts = self.sim.contacts()

        # Basket detection
        in_basket = self._check_object_in_basket()

        # Log data
        self.camera_log.append(
            {
                "frame": self.frame_count,
                "time": self.sim.data.time,
                "rgb_shape": rgb.shape,
                "depth_mean": float(np.mean(depth)),
                "depth_std": float(np.std(depth)),
            }
        )

        self.joint_log.append(
            {
                "frame": self.frame_count,
                "time": self.sim.data.time,
                "positions": np.array(arm_positions).tolist(),
                "velocities": np.array(arm_velocities).tolist(),
            }
        )

        self.ee_log.append(
            {
                "frame": self.frame_count,
                "time": self.sim.data.time,
                "position": ee_pos.tolist(),
                "rotation_matrix": ee_rot.tolist(),
            }
        )

        self.contact_log.append(
            {
                "frame": self.frame_count,
                "time": self.sim.data.time,
                "num_contacts": len(contacts),
            }
        )

        self.basket_log.append(
            {
                "frame": self.frame_count,
                "time": self.sim.data.time,
                "in_basket": in_basket["in_basket"],
                "in_x_bounds": in_basket["in_x"],
                "in_y_bounds": in_basket["in_y"],
                "in_z_bounds": in_basket["in_z"],
                "object_position": in_basket["object_pos"],
            }
        )

        # Optionally save images
        if save_images and self.frame_count % 30 == 0:  # Save every 30 frames
            np.save(self.log_dir / f"rgb_frame_{self.frame_count:06d}.npy", rgb)
            np.save(
                self.log_dir / f"depth_frame_{self.frame_count:06d}.npy", depth
            )

        self.frame_count += 1

    def _check_object_in_basket(self):
        """Check if the grasp object is inside the basket."""

        # Get basket parameters
        basket_center = self.sim.ids.get("basket_center")
        basket_dims = self.sim.ids.get("basket_dims")
        basket_height = self.sim.ids.get("basket_height")

        # Default values if basket not configured
        if (
            basket_center is None
            or basket_dims is None
            or basket_height is None
        ):
            return {
                "in_basket": False,
                "in_x": False,
                "in_y": False,
                "in_z": False,
                "object_pos": None,
            }

        # Get grasp object body ID
        grasp_obj_info = self.sim.ids.get("grasp_object", {})
        obj_body_name = grasp_obj_info.get("body_name", "sample_object")

        obj_body_id = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name
        )

        if obj_body_id < 0:
            return {
                "in_basket": False,
                "in_x": False,
                "in_y": False,
                "in_z": False,
                "object_pos": None,
            }

        # Get object position
        obj_pos = self.sim.data.xpos[obj_body_id].copy()

        # Check if object is within basket bounds
        in_x = abs(obj_pos[0] - basket_center[0]) < basket_dims[0] / 2.0
        in_y = abs(obj_pos[1] - basket_center[1]) < basket_dims[1] / 2.0

        # Object should be above basket bottom
        basket_bottom = basket_center[2] - basket_height / 2.0
        in_z = obj_pos[2] > basket_bottom

        in_basket = in_x and in_y and in_z

        return {
            "in_basket": in_basket,
            "in_x": in_x,
            "in_y": in_y,
            "in_z": in_z,
            "object_pos": obj_pos.tolist(),
        }

    def save_logs(self):
        """Save all logs to files."""

        with open(self.log_dir / "camera_log.json", "w") as f:
            json.dump(self.camera_log, f, indent=2)

        with open(self.log_dir / "joint_log.json", "w") as f:
            json.dump(self.joint_log, f, indent=2)

        with open(self.log_dir / "ee_log.json", "w") as f:
            json.dump(self.ee_log, f, indent=2)

        with open(self.log_dir / "contact_log.json", "w") as f:
            json.dump(self.contact_log, f, indent=2)

        with open(self.log_dir / "basket_log.json", "w") as f:
            json.dump(self.basket_log, f, indent=2)

        # Print summary statistics
        in_basket_count = sum(
            1 for entry in self.basket_log if entry["in_basket"]
        )
        in_basket_percentage = (
            (in_basket_count / len(self.basket_log) * 100)
            if self.basket_log
            else 0
        )

        print(f"\n[OK] Saved sensor logs to: {self.log_dir}")
        print(f"  - Logged {self.frame_count} frames")
        print(f"  - Object in basket: {in_basket_percentage:.1f}% of frames")


def view_object_with_sensors(
    object_name,
    save_sensors=False,
    save_images=False,
    log_interval=10,
    enable_gui=True,
):
    """Open GUI and optionally log sensor data."""
    print(f"Loading {object_name} in MuJoCo viewer...")
    if save_sensors:
        print("[INFO] Sensor logging ENABLED")
    print("Press Ctrl+C to exit\n")

    # Load configuration
    project_root = Path(__file__).parent.resolve()
    config_path = project_root / "configs" / "test_config_mj.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update to use the requested object
    object_xml = (
        project_root
        / "assets"
        / "mujoco_objects"
        / object_name
        / "textured.xml"
    )

    config["mujoco"]["grasp_object"]["xml"] = str(object_xml)
    config["mujoco"]["gui"] = enable_gui

    # Initialize simulation
    sim = MjSim(config)
    sim.reset()
    # Define home position
    q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    
    # Setup sensor logger if requested
    logger = None
    if save_sensors:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = project_root / "sensor_logs" / f"{object_name}_{timestamp}"
        logger = SensorLogger(sim, log_dir)
        print(f"  Logging to: {log_dir}\n")
    
    # Create robot controller
    ik = IKSolver(model=sim.model, data=sim.data, ee_body="hand", joint_dofs=7)
    controller = RobotController(
        model=sim.model,
        data=sim.data,
        sim=sim,
        ik_solver=ik,
        logger=logger,
        log_interval=log_interval
    )

    print(f"[OK] {object_name} loaded!")
    print("  - Textures should be visible")
    print("  - Use mouse to rotate view")
    print("  - Press Ctrl+C to exit")
    print("\n[WARNING] SAFETY FEATURES ACTIVE:")
    print("   [X] Collision Detection - exits if robot hits obstacles")
    # print("   [!] Slip Detection - warns if object slips from gripper")  # Disabled
    if save_sensors:
        print(f"   [LOG] Sensor Logging - saving every {log_interval} frames")
    print("\nViewer is running...")
    time.sleep(2)  # Give user time to adjust viewer before simulation starts

    # Track object slip detection
    slip_warning_count = 0
    last_slip_warning_time = 0.0
    last_basket_check_time = 0.0
    basket_check_interval = 2.0  # Check every 2 seconds

    # === PHASE 1: MOVE TO HOME ===
    print("\n" + "="*80)
    print("[PHASE 1] MOVING TO HOME POSITION")
    print("="*80)
    controller.move_to_home(q_home, verbose=True)
    time.sleep(1)  # Pause briefly at home position

    # === PHASE 2: MOVE TO BANANA ===
    print("\n" + "="*80)
    print("[PHASE 2] MOVING TO BANANA TARGET")
    print("="*80)

    banana_mat = sim.data.body("sample_object").xmat.reshape(3, 3)
    banana_yaw = math.atan2(banana_mat[1, 0], banana_mat[0, 0])

    q_flip = np.array([0.0, 1.0, 0.0, 0.0])
    q_yaw = np.array([math.cos(banana_yaw/2), 0, 0, math.sin(banana_yaw/2)])

    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = q_flip
    target_quat = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    banana_pos_static = sim.data.body("sample_object").xpos.copy()  
    world_target = banana_pos_static + np.array([0.0, 0.0, 0.2])  # Move 10 cm above the banana
    
    print(f"Banana World Pose: {banana_pos_static}")
    print(f"Target Position (10cm above): {world_target}")
    print(f"Target Quaternion (WXYZ): {target_quat}")
    
    success = controller.move_to_target(world_target, target_quat, verbose=True)
    
    if success:
        print("\n[SUCCESS] Robot reached target!")
    else:
        print("\n[WARNING] Failed to reach target")
    
    # Get step count from controller
    step_count = controller.get_step_count()

    try: 
        while True:
            # Hold current position (robot stays where it is)
            sim.step()
            time.sleep(0.01)
            step_count += 1

            # Log sensors at specified interval
            if logger and step_count % log_interval == 0:
                logger.log_frame(save_images=save_images)
                if step_count % 1000 == 0:
                    print(
                        f"  [LOG] Logged {logger.frame_count} frames (step {step_count})"
                    )

            # Slip detection disabled
            # grasp_object_name = config.get('mujoco', {}).get('grasp_object', {}).get('body', 'sample_object')
            # if sim.check_object_slip(grasp_object_name, gripper_body_name="hand", velocity_threshold=0.05):
            #     current_time = sim.data.time
            #     if current_time - last_slip_warning_time > 2.0:
            #         slip_warning_count += 1
            #         print(f"\nSLIP DETECTED (#{slip_warning_count}): Object '{grasp_object_name}' is slipping!")
            #         print(f"   Time: {current_time:.2f}s")
            #         last_slip_warning_time = current_time

            # Check for collisions with obstacles
            if sim.check_robot_obstacle_collision(robot_body_prefix="link"):
                mobjs = sim.ids.get("moving_obstacles", {})
                obstacle_geoms = set()
                for obstacle_name in mobjs.keys():
                    body_id = mujoco.mj_name2id(
                        sim.model, mujoco.mjtObj.mjOBJ_BODY, obstacle_name
                    )
                    if body_id >= 0:
                        geoms = np.where(sim.model.geom_bodyid == body_id)[0]
                        obstacle_geoms.update(geoms.tolist())

                robot_keywords = ["link", "hand", "finger", "gripper"]
                collision_body = "unknown"
                for contact in sim.contacts():
                    g1, g2 = contact["geom1"], contact["geom2"]
                    if g1 in obstacle_geoms or g2 in obstacle_geoms:
                        robot_geom = g1 if g2 in obstacle_geoms else g2
                        body_id = sim.model.geom_bodyid[robot_geom]
                        body_name = mujoco.mj_id2name(
                            sim.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                        )
                        if body_name and any(
                            kw in body_name.lower() for kw in robot_keywords
                        ):
                            collision_body = body_name
                            break

                print("\n" + "=" * 80)
                print("[COLLISION] COLLISION DETECTED - STOPPING SIMULATION")
                print("=" * 80)
                print(
                    f"Robot part '{collision_body}' collided with moving obstacle"
                )
                print("=" * 80)

                if logger:
                    logger.save_logs()

                sim.close()
                return 1

            # Periodic basket goal check
            current_time = sim.data.time
            if current_time - last_basket_check_time > basket_check_interval:
                basket_status = check_object_in_basket(sim)
                if basket_status["in_basket"]:
                    print(
                        f"\nNice! Object is in the basket! (t={current_time:.1f}s)"
                    )
                    print(
                        f"   Position: [{basket_status['object_pos'][0]:.3f}, "
                        f"{basket_status['object_pos'][1]:.3f}, "
                        f"{basket_status['object_pos'][2]:.3f}]"
                    )
                last_basket_check_time = current_time

    except KeyboardInterrupt:
        print("\nClosing viewer...")

        if logger:
            logger.save_logs()

        sim.close()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View object with optional sensor logging"
    )
    parser.add_argument(
        "object_name", help="Name of the object (e.g., YcbBanana)"
    )
    parser.add_argument(
        "--save-sensors",
        action="store_true",
        help="Save sensor data during simulation",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save RGB/depth images (requires --save-sensors)",
    )
    parser.add_argument(
        "--enable_gui",
        action="store_true",
        help="Enable GUI for viewing scene",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log sensor data every N steps (default: 10)",
    )

    args = parser.parse_args()

    if args.save_images and not args.save_sensors:
        print(
            (
                "Warning: --save-images requires"
                " --save-sensors, enabling sensor saving"
            )
        )
        args.save_sensors = True

    if not args.enable_gui:
        os.environ["MUJOCO_GL"] = "egl"

    sys.exit(
        view_object_with_sensors(
            args.object_name,
            save_sensors=args.save_sensors,
            save_images=args.save_images,
            log_interval=args.log_interval,
            enable_gui=args.enable_gui,
        )
    )
