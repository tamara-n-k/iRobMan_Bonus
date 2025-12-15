import mujoco
import numpy as np


class MjRobot:
    """Minimal robot wrapper in MuJoCo Menagerie.

    Provides kinematic control utilities including forward kinematics,
    Jacobian-based inverse kinematics for position and full 6D pose,
    and joint position management with automatic limit enforcement.

    Attributes:
        model: MuJoCo model instance.
        data: MuJoCo data instance.
        ee_id: Integer ID of the end-effector body.
        nq: Number of generalized positions in the model.
        nv: Number of generalized velocities in the model.
        arm_pairs: List of (qpos_index, dof_index) tuples for controllable arm joints.
    """

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, ee_body_name: str
    ):
        """Initializes the robot wrapper.

        Args:
            model: MuJoCo model containing the robot.
            data: MuJoCo data instance for state storage.
            ee_body_name: Name of the end-effector body in the model.

        Raises:
            ValueError: If the end-effector body is not found in the model.
        """
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )
        if self.ee_id < 0:
            raise ValueError(
                f"End-effector body '{ee_body_name}' not found in model"
            )

        # Select only robot arm joints (exclude free bodies like the ball)
        self.nq = model.nq
        self.nv = model.nv
        self.arm_pairs = []  # list of (qpos_index, dof_index) for controllable joints
        self._joint_lims = {}  # map qpos_index -> (min,max) if limited
        for i in range(1, 8):
            jname = f"joint{i}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            jtype = int(model.jnt_type[jid])
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            # Only include hinge/slide (1 DoF) joints for IK updates
            if jtype in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                self.arm_pairs.append((qadr, dadr))
                # Cache limits if available (autolimits enabled in menagerie)
                rng = self.model.jnt_range[jid]
                if (
                    np.isfinite(rng[0])
                    and np.isfinite(rng[1])
                    and rng[0] < rng[1]
                ):
                    self._joint_lims[qadr] = (float(rng[0]), float(rng[1]))

    def get_qpos(self):
        """Returns a copy of the current generalized position vector.

        Returns:
            A numpy array containing the current qpos.
        """
        return self.data.qpos.copy()

    def set_qpos(self, q):
        """Sets the generalized position vector and updates kinematics.

        Args:
            q: New qpos vector to apply.
        """
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def set_arm_joint_positions(
        self, joint_positions, clamp: bool = True, sync: bool = True
    ):
        """Sets the controllable arm joints (joint1..joint7) to the provided positions.

        Args:
            joint_positions: Iterable of joint angle values.
            clamp: Whether to enforce joint limits.
            sync: Whether to run forward kinematics after setting positions.

        Raises:
            ValueError: If the number of positions doesn't match the arm's DoF.
        """
        joint_positions = list(joint_positions)
        if len(joint_positions) != len(self.arm_pairs):
            raise ValueError(
                f"Expected {len(self.arm_pairs)} joint values, got {len(joint_positions)}"
            )
        for value, (qidx, _) in zip(joint_positions, self.arm_pairs):
            new_q = float(value)
            if clamp and qidx in self._joint_lims:
                lo, hi = self._joint_lims[qidx]
                new_q = min(max(new_q, lo), hi)
            self.data.qpos[qidx] = new_q
        if sync:
            mujoco.mj_forward(self.model, self.data)

    def get_ee_pose(self):
        """Computes the current end-effector pose.

        Returns:
            A tuple (position, quaternion_xyzw) where position is a 3D numpy array
            and quaternion_xyzw is a 4D numpy array in [x, y, z, w] order.
        """
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self.ee_id].copy()
        xmat = self.data.xmat[self.ee_id].reshape(3, 3)
        # convert 3x3 to quaternion (wxyz) using MuJoCo helper
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.flatten())
        # MuJoCo returns (w,x,y,z)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        return pos, quat_xyzw
