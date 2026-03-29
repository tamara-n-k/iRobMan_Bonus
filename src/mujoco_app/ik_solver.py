import mujoco
import numpy as np


class IKSolver:

    def __init__(self, model, data, ee_body="hand", joint_dofs=7):

        self.model = model
        self.data = data
        self.joint_dofs = joint_dofs

        self.body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            ee_body
        )

        # joint limits
        self.joint_min = model.jnt_range[:joint_dofs,0]
        self.joint_max = model.jnt_range[:joint_dofs,1]

    def quat_to_mat(self, q):

        w,x,y,z = q

        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])

    def orientation_error(self, R_target, R_current):

        R_err = R_target @ R_current.T

        return 0.5 * np.array([
            R_err[2,1] - R_err[1,2],
            R_err[0,2] - R_err[2,0],
            R_err[1,0] - R_err[0,1]
        ])

    def compute_error(self, target_pos, target_quat):

        current_pos = self.data.xpos[self.body_id].copy()

        current_rot = self.data.xmat[self.body_id].reshape(3,3)

        pos_error = target_pos - current_pos

        R_target = self.quat_to_mat(target_quat)

        rot_error = self.orientation_error(
            R_target,
            current_rot
        )

        # weight orientation less for grasping
        error = np.concatenate([
            1.0 * pos_error,
            0.4 * rot_error
        ])

        return error

    def compute_jacobian(self):

        J_pos = np.zeros((3,self.model.nv))
        J_rot = np.zeros((3,self.model.nv))

        mujoco.mj_jacBody(
            self.model,
            self.data,
            J_pos,
            J_rot,
            self.body_id
        )

        J = np.vstack([
            J_pos[:,:self.joint_dofs],
            J_rot[:,:self.joint_dofs]
        ])

        return J


    def solve(
            self,
            target_pos,
            target_quat,
            q_init=None,
            max_iters=300,
            tol=1e-4,
            damping=0.03,
            step_size=0.4):

        # Save original joint positions
        original_q = self.data.qpos.copy()

        if q_init is None:
            q = self.data.qpos[:self.joint_dofs].copy()
        else:
            q = q_init.copy()

        success = False
        error_history = []

        for i in range(max_iters):

            self.data.qpos[:self.joint_dofs] = q

            mujoco.mj_forward(
                self.model,
                self.data
            )

            error = self.compute_error(
                target_pos,
                target_quat
            )
            
            error_norm = np.linalg.norm(error)
            error_history.append(error_norm)

            if error_norm < tol:
                success = True
                break

            J = self.compute_jacobian()

            JT = J.T

            JJ = J @ JT

            # Damping matrix - use standard Levenberg-Marquardt
            damp_matrix = damping*damping*np.eye(6)

            delta_q = JT @ np.linalg.solve(
                JJ + damp_matrix,
                error
            )

            q += step_size * delta_q

            q = np.clip(
                q,
                self.joint_min,
                self.joint_max
            )

        # Restore original joint positions
        self.data.qpos = original_q
        mujoco.mj_forward(self.model, self.data)

        if not success:
            print(
                f"\nIK solver did not converge after {max_iters} iterations")
            

        return q, success