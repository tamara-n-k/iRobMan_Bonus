"""Shared pose containers used by perception and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    """Container for an object pose."""

    body_name: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray
