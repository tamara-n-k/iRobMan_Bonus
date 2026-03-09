import argparse
import math
import time
from typing import Any

import matplotlib.pyplot as plt
import yaml

from mujoco_app.mj_simulation import MjSim


def configured_camera_names(config: dict[str, Any]) -> list[str]:
    mujoco_cfg = dict(config.get("mujoco", {}))
    names: list[str] = []

    primary_cfg = dict(mujoco_cfg.get("camera", {}))
    primary_name = primary_cfg.get("name")
    if primary_name:
        names.append(primary_name)

    user_cfg = dict(mujoco_cfg.get("user_camera", {}))
    if user_cfg.get("enable", True):
        user_name = user_cfg.get("name")
        if user_name:
            names.append(user_name)

    wrist_cfg = dict(mujoco_cfg.get("wrist_camera", {}))
    if wrist_cfg.get("enable", False):
        wrist_name = wrist_cfg.get("name")
        if wrist_name:
            names.append(wrist_name)

    for extra_cfg in mujoco_cfg.get("extra_cameras", []):
        extra_name = extra_cfg.get("name")
        if extra_name:
            names.append(extra_name)

    deduped_names: list[str] = []
    for name in names:
        if name not in deduped_names:
            deduped_names.append(name)
    return deduped_names


def render_camera_frame(
    sim: MjSim,
    config: dict[str, Any],
    camera_name: str,
):
    cam_cfg = dict(config.get("mujoco", {}).get("camera", {}))
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    near = float(cam_cfg.get("near", 0.01))
    far = float(cam_cfg.get("far", 5.0))
    fovy = float(cam_cfg.get("fovy", 58.0))
    rgb, _, _, _ = sim.render_camera(
        camera_name,
        width=width,
        height=height,
        near=near,
        far=far,
        fovy=fovy,
    )
    return rgb


def create_figure(camera_names: list[str]):
    num_cameras = len(camera_names)
    num_cols = min(2, num_cameras)
    num_rows = int(math.ceil(num_cameras / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]
    else:
        axes = list(axes)
    return fig, list(axes)


def view_cameras(
    config: dict[str, Any],
    stabilization_steps: int,
    realtime: bool,
) -> None:
    sim = MjSim(config)
    try:
        sim.reset()
        for _ in range(stabilization_steps):
            sim.step()

        camera_names = configured_camera_names(config)
        if not camera_names:
            raise ValueError("No cameras configured to display.")

        fig, axes = create_figure(camera_names)
        image_artists = []

        for axis, camera_name in zip(axes, camera_names):
            frame = render_camera_frame(sim, config, camera_name)
            artist = axis.imshow(frame)
            axis.set_title(camera_name)
            axis.axis("off")
            image_artists.append((camera_name, artist))

        for axis in axes[len(camera_names) :]:
            axis.axis("off")

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        timestep = float(sim.model.opt.timestep)
        start_time = time.perf_counter()
        step_idx = 0

        while plt.fignum_exists(fig.number):
            sim.step()
            step_idx += 1

            for camera_name, artist in image_artists:
                frame = render_camera_frame(sim, config, camera_name)
                artist.set_data(frame)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            if realtime:
                target_elapsed = step_idx * timestep
                wall_elapsed = time.perf_counter() - start_time
                remaining = target_elapsed - wall_elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        sim.close()


def main(config_path: str, stabilization_steps: int, realtime: bool) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    view_cameras(config, stabilization_steps, realtime)


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
        help="Pace the simulation to wall-clock time while updating the camera window.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        stabilization_steps=args.stabilization_steps,
        realtime=args.realtime,
    )
