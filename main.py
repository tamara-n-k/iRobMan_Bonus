import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from mujoco_app.evaluation import Evaluation, ExpData
from mujoco_app.exceptions import CollisionDetectedError
from mujoco_app.ik_solver import IKSolver
from mujoco_app.mj_simulation import MjSim
from mujoco_app.robot_controller import RobotController
from mujoco_app.task_manager import PickPlaceTask


# Metric for success
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


# point projection from 3D to 2D
def project_points(X_world, K, R_cw, t_cw):
    X_cam = (R_cw @ X_world.T + t_cw[:, None]).T
    x = (K @ X_cam.T).T
    return x[:, :2] / x[:, 2:3]


# Helper method to visualize RGB and DEPTH image
def show_rgb_depth(
    rgb,
    depth,
    cam_name,
    near=None,
    far=None,
    cmap="viridis",
    figsize=(10, 4),
    title_rgb="RGB",
    title_depth="Depth",
):
    """
    Display RGB and depth images side-by-side.

    Args:
        rgb (H, W, 3): uint8 RGB image
        depth (H, W): float depth image (meters)
        cam_name (str): name of the camera
        near (float, optional): near clip for visualization
        far (float, optional): far clip for visualization
        cmap (str): colormap for depth
        figsize (tuple): matplotlib figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title(cam_name + "_" + title_rgb)
    axes[0].axis("off")

    # Depth processing
    depth_vis = depth.astype(np.float32)

    if near is not None and far is not None:
        depth_vis = np.clip(depth_vis, near, far)

    # Hide invalid depth
    depth_vis[~np.isfinite(depth_vis)] = np.nan

    axes[1].set_title(cam_name + "_" + title_depth)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# Experiment runner example
def runner(
    config: Dict[str, Any],
    num_experiments: int,
    asset_name: str
) -> List[ExpData]:  # TODO collision detection und slip detection
    # Configure in YAML
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    near = cam_cfg.get("near", 0.01)
    far = cam_cfg.get("far", 5.0)
    fovy = cam_cfg.get("fovy", 58.0)

    sim = MjSim(config)
    experiment_runs: List[ExpData] = []
    for run_index in range(1, num_experiments + 1):
        sim.reset()

        # For sim stabilization
        for _ in range(1000):
            sim.step()
        # Define home position
        q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        # Create robot controller
        ik = IKSolver(model=sim.model, data=sim.data, ee_body="hand", joint_dofs=7)
        controller = RobotController(
            model=sim.model,
            data=sim.data,
            sim=sim,
            ik_solver=ik
        )
        
        # lower iterations per step for reaching the target pose
        print("Moving to target pose...")
        sim.step()
        task = PickPlaceTask(sim, controller)
        exp_data = ExpData(
            sim=sim,
            body_name="sample_object",
            asset_name=asset_name,
            run_index=run_index,
        )
        task_success = False
        try:
            print("\n[MISSION CONTROL] Starting Pick-and-Place sequence...")
            task_success = task.run(
                q_home=q_home,
                exp_data=exp_data,
                object_name="sample_object",
                basket_body_name="basket",
            )
            if task_success:
                print("\n[MISSION CONTROL] Task completed successfully.")
            else:
                print("\n[MISSION CONTROL] Task did not complete.")
        except CollisionDetectedError as ce:
            exp_data.save_termination_reason("collision")
            print(f"\n[ERROR] Task interrupted: {ce}")
        except Exception as e:
            print(f"\n[ERROR] Task interrupted: {e}")
            exp_data.save_termination_reason("error")

        basket_status = check_object_in_basket(sim)
        exp_data.save_final_basket_status(basket_status)
        experiment_runs.append(exp_data)
        
    sim.close()
    print("Simulation completed.")
    return experiment_runs

def _get_objects() -> List[str]:
    return [
        "YcbBanana", "YcbCrackerBox", "YcbFoamBrick", "YcbMasterChefCan", "YcbMustardBottle",
        "YcbPear", "YcbPowerDrill", "YcbStrawberry", "YcbTennisBall", "YcbTomatoSoupCan"
    ]


def main(config_path: str):
    experiment_runs: List[ExpData] = []
    for object_name in _get_objects():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            project_root = Path(__file__).parent.resolve()
            object_xml = project_root / "assets" / "mujoco_objects" / object_name / "textured.xml"

            if not object_xml.exists():
                print(f"[ERROR] Object XML not found: {object_xml}")
                return
            config["mujoco"]["grasp_object"]["xml"] = str(object_xml)
        # You can make runner for one experiment
        experiment_runs.extend(
            runner(
                config=config,
                num_experiments=1,
                asset_name=object_name
            )
        )
    evaluation = Evaluation(experiment_runs)
    evaluation_report = evaluation.format_overview()
    output_path = Path("evaluation_results") / "evaluation_overview.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(evaluation_report + "\n", encoding="utf-8")
    print(evaluation_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/test_config_mj.yaml"
    )
    args = parser.parse_args()
    main(config_path=args.config)
