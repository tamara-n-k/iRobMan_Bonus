"""Ground-truth pose accessors for evaluation-only code."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_app.pose_types import Pose

if TYPE_CHECKING:
    from mujoco_app.mj_simulation import MjSim


def get_body_pose_ground_truth(sim: "MjSim", body_name: str) -> Pose:
    """Returns the current MuJoCo body pose for evaluation only."""

    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model")

    position = sim.data.xpos[body_id].copy()
    rotation = sim.data.xmat[body_id].reshape(3, 3)
    quat_wxyz = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat_wxyz, rotation.reshape(-1))
    quaternion_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=float,
    )
    return Pose(
        body_name=body_name,
        position=position,
        quaternion_xyzw=quaternion_xyzw,
    )
