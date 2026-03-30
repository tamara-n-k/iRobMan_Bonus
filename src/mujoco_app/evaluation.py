"""Experiment evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional

import numpy as np

from mujoco_app.ground_truth import get_body_pose_ground_truth
from mujoco_app.pose_types import Pose


class ExpData:
    """Per-run evaluation data collector."""

    def __init__(
        self,
        sim: Any,
        body_name: str,
        asset_name: str,
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        self.asset_name = asset_name
        self.perception_estimated_pose: Optional[Pose] = None
        self.perception_ground_truth_pose: Optional[Pose] = None
        self._height_before_grasp: Optional[float] = None
        self._height_after_grasp: Optional[float] = None

    def save_perception(self, estimated_pose: Pose) -> None:
        """Store the estimated pose and current ground truth for this run."""

        self.perception_estimated_pose = estimated_pose
        self.perception_ground_truth_pose = get_body_pose_ground_truth(
            self.sim, self.body_name
        )

    def save_height_before_grasp(self) -> float:
        """Sample and store the object height before grasp."""

        self._height_before_grasp = self._object_height()
        return self._height_before_grasp

    def save_height_after_grasp(self) -> float:
        """Sample and store the post-grasp object height."""

        if self._height_before_grasp is None:
            raise ValueError(
                "Height before grasp was not recorded. "
                "Call save_height_before_grasp() first."
            )

        self._height_after_grasp = self._object_height()
        return self._height_after_grasp

    @property
    def height_before_grasp(self) -> Optional[float]:
        return self._height_before_grasp

    @property
    def height_after_grasp(self) -> Optional[float]:
        return self._height_after_grasp

    def _object_height(self) -> float:
        return float(self.sim.data.body(self.body_name).xpos[2])


class Evaluation:
    """Aggregates stage-wise experiment metrics over recorded runs."""

    def __init__(
        self,
        experiment_runs: list[ExpData],
        perception_position_tolerance: float = 0.02,
        perception_orientation_tolerance_deg: float = 10.0,
        grasp_height_threshold: float = 0.02,
    ) -> None:
        self.experiment_runs = experiment_runs
        self.perception_position_tolerance = perception_position_tolerance
        self.perception_orientation_tolerance_deg = (
            perception_orientation_tolerance_deg
        )
        self.grasp_height_threshold = grasp_height_threshold

    def overview(self) -> dict[str, dict[str, dict[str, float | int | None]]]:
        """Return aggregated metrics for every evaluated object."""

        results: DefaultDict[str, Dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: {"perception": [], "grasp": []}
        )
        for exp_data in self.experiment_runs:
            perception_result = self._evaluate_perception(exp_data)
            if perception_result is not None:
                results[exp_data.asset_name]["perception"].append(
                    perception_result
                )

            grasp_result = self._evaluate_grasp(exp_data)
            if grasp_result is not None:
                results[exp_data.asset_name]["grasp"].append(grasp_result)

        summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
        for object_name, stage_results in results.items():
            perception_results = stage_results["perception"]
            grasp_results = stage_results["grasp"]
            summary[object_name] = {
                "perception": self._summarize_perception(perception_results),
                "grasp": self._summarize_grasp(grasp_results),
            }
        return summary

    def format_overview(self) -> str:
        """Render a compact text report for console output."""

        summary = self.overview()
        if not summary:
            return "No evaluation data collected."

        lines = ["\n=== Evaluation Overview ==="]
        for object_name in sorted(summary):
            perception = summary[object_name]["perception"]
            grasp = summary[object_name]["grasp"]
            lines.append(f"{object_name}:")
            lines.append(
                "  Perception -> "
                f"runs={perception['num_trials']}, "
                f"success_rate={perception['success_rate']:.2%}, "
                "avg_position_error="
                f"{self._format_metric(perception['avg_position_error'], '.4f')} m, "
                "avg_orientation_error="
                f"{self._format_metric(perception['avg_orientation_error_deg'], '.2f')} deg"
            )
            lines.append(
                "  Grasp      -> "
                f"runs={grasp['num_trials']}, "
                f"success_rate={grasp['success_rate']:.2%}, "
                "avg_height_increase="
                f"{self._format_metric(grasp['avg_height_increase'], '.4f')} m, "
                "avg_height_error="
                f"{self._format_metric(grasp['avg_height_error'], '.4f')} m"
            )
        return "\n".join(lines)

    def _evaluate_perception(
        self,
        exp_data: ExpData,
    ) -> Optional[dict[str, Any]]:
        estimated_pose = exp_data.perception_estimated_pose
        ground_truth = exp_data.perception_ground_truth_pose
        if estimated_pose is None or ground_truth is None:
            return None

        position_error = float(
            np.linalg.norm(estimated_pose.position - ground_truth.position)
        )
        orientation_error_deg = float(
            self._quaternion_angle_deg(
                estimated_pose.quaternion_xyzw,
                ground_truth.quaternion_xyzw,
            )
        )
        success = (
            position_error <= self.perception_position_tolerance
            and orientation_error_deg
            <= self.perception_orientation_tolerance_deg
        )
        return {
            "success": success,
            "position_error": position_error,
            "orientation_error_deg": orientation_error_deg,
        }

    def _evaluate_grasp(
        self,
        exp_data: ExpData,
    ) -> Optional[dict[str, Any]]:
        initial_height = exp_data.height_before_grasp
        lifted_height = exp_data.height_after_grasp
        if initial_height is None or lifted_height is None:
            return None

        height_increase = float(lifted_height - initial_height)
        height_error = float(
            max(0.0, self.grasp_height_threshold - height_increase)
        )
        success = height_increase >= self.grasp_height_threshold
        return {
            "success": success,
            "initial_height": float(initial_height),
            "lifted_height": float(lifted_height),
            "height_increase": height_increase,
            "height_error": height_error,
        }

    def _summarize_perception(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, float | int | None]:
        if not results:
            return {
                "num_trials": 0,
                "success_rate": 0.0,
                "avg_position_error": None,
                "avg_orientation_error_deg": None,
            }

        return {
            "num_trials": len(results),
            "success_rate": float(np.mean([item["success"] for item in results])),
            "avg_position_error": float(
                np.mean([item["position_error"] for item in results])
            ),
            "avg_orientation_error_deg": float(
                np.mean([item["orientation_error_deg"] for item in results])
            ),
        }

    def _summarize_grasp(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, float | int | None]:
        if not results:
            return {
                "num_trials": 0,
                "success_rate": 0.0,
                "avg_height_increase": None,
                "avg_height_error": None,
            }

        return {
            "num_trials": len(results),
            "success_rate": float(np.mean([item["success"] for item in results])),
            "avg_height_increase": float(
                np.mean([item["height_increase"] for item in results])
            ),
            "avg_height_error": float(
                np.mean([item["height_error"] for item in results])
            ),
        }

    @staticmethod
    def _format_metric(value: float | int | None, fmt: str) -> str:
        if value is None:
            return "n/a"
        return format(value, fmt)

    @staticmethod
    def _quaternion_angle_deg(
        quaternion_a_xyzw: np.ndarray,
        quaternion_b_xyzw: np.ndarray,
    ) -> float:
        quat_a = np.asarray(quaternion_a_xyzw, dtype=float)
        quat_b = np.asarray(quaternion_b_xyzw, dtype=float)

        norm_a = np.linalg.norm(quat_a)
        norm_b = np.linalg.norm(quat_b)
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            raise ValueError("Quaternion norm is zero.")

        quat_a /= norm_a
        quat_b /= norm_b
        dot_product = float(np.clip(abs(np.dot(quat_a, quat_b)), -1.0, 1.0))
        angle_rad = 2.0 * np.arccos(dot_product)
        return float(np.degrees(angle_rad))
