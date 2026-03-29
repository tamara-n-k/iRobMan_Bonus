import argparse
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import yaml

from mujoco_app.grasp import (
    estimate_grasp_from_hand_camera,
    get_wrist_camera_observation,
)


def run_grasp_smoke_test(
    config_path: Path,
    model_path: Path,
    stabilization_steps: int,
    gui: bool,
    mujoco_gl: str | None,
) -> None:
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl

    from mujoco_app.ground_truth import get_body_pose_ground_truth
    from mujoco_app.mj_simulation import MjSim

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.setdefault("mujoco", {})["gui"] = gui

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    sim = MjSim(config)
    try:
        sim.reset()
        for _ in range(stabilization_steps):
            sim.step()

        body_name = sim.ids["grasp_object"]["body_name"]
        object_pose = get_body_pose_ground_truth(sim, body_name)
        observation = get_wrist_camera_observation(sim)
        depth = np.asarray(observation["depth"], dtype=float)
        camera_far = float(observation["camera_far"])
        if float(np.min(depth)) >= camera_far - 1e-3:
            raise ValueError(
                "Camera observation is empty or saturated at the far plane. "
                "Try a smaller --stabilization-steps value."
            )
        prediction = estimate_grasp_from_hand_camera(
            observation=observation,
            object_pose=object_pose,
            model_path=model_path,
        )

        print("Grasp inference succeeded.")
        print(f"camera={observation['camera_name']}")
        print(
            "depth_range_m=({:.4f}, {:.4f})".format(
                float(np.min(depth)),
                float(np.max(depth)),
            )
        )
        print(
            "object_position_m={}".format(
                np.round(object_pose.position, 4).tolist(),
            )
        )
        print(
            "grasp_position_m={}".format(
                np.round(prediction.pose.position, 4).tolist(),
            )
        )
        print(
            "grasp_quaternion_xyzw={}".format(
                np.round(prediction.pose.quaternion_xyzw, 4).tolist(),
            )
        )
        print(f"grasp_width_m={prediction.width:.4f}")
    finally:
        sim.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test the GIGA grasp pipeline inside the MuJoCo scene."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/test_config_mj.yaml"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("giga/giga_pile.pt"),
    )
    parser.add_argument(
        "--stabilization-steps",
        type=int,
        default=10,
        help=(
            "Number of simulation steps before capturing the grasp observation. "
            "Keep this small for the default wrist camera scene."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the MuJoCo viewer while running the smoke test.",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default=None,
        help="Optional MuJoCo GL backend such as egl or osmesa for headless runs.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print the full Python traceback on failure.",
    )
    args = parser.parse_args()

    try:
        run_grasp_smoke_test(
            config_path=args.config,
            model_path=args.model,
            stabilization_steps=args.stabilization_steps,
            gui=args.gui,
            mujoco_gl=args.mujoco_gl,
        )
    except Exception as exc:
        print(f"Grasp inference failed: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
