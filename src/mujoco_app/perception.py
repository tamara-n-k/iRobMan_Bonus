"""Perception placeholders for pose estimation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from mujoco_app.pose_types import Pose

if TYPE_CHECKING:
    from mujoco_app.mj_simulation import MjSim


def estimate_body_pose(
    sim: "MjSim",
    body_name: str,
    observation: Mapping[str, Any] | None = None,
) -> Pose:
    """Placeholder pose estimator.

    This function is where the actual RGB-D based detection / pose estimation
    pipeline should go. For now it returns a nominal pose prior from the scene
    configuration so the rest of the task pipeline has a stable interface.
    """

    return Pose(
        body_name=body_name,
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    )
