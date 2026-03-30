"""
Robot motion controller using IK and RRT motion planning.
Provides high-level control methods for smooth, collision-aware robot movement.
"""
import numpy as np
import time
import mujoco
from mujoco_app.ik_solver import IKSolver
from mujoco_app.motion_planner import RRTPlanner
from mujoco_app.exceptions import CollisionDetectedError


class RobotController:
    """Controls robot movement using IK solving and RRT path planning."""
    
    def __init__(self, model, data, sim, ik_solver=None, logger=None, log_interval=10):
        self.model = model
        self.data = data
        self.sim = sim
        self.logger = logger
        self.log_interval = log_interval
        self.step_count = 0
        
        # Create IK solver if not provided
        if ik_solver is None:
            self.ik_solver = IKSolver(model=model, data=data, ee_body="hand", joint_dofs=7)
        else:
            self.ik_solver = ik_solver
        
        # Create RRT planner
        self.planner = RRTPlanner(model, data, range(7), sim.ids)
    
    def move_to_home(self, home_q, verbose=True):
        current_q = self.data.qpos[:7].copy()
        
        if verbose:
            print(f"\n[CONTROLLER] Moving to home position...")
            dist = np.linalg.norm(home_q - current_q)
            print(f"  Distance to home (joint space): {dist:.4f} rad")
        
        self._execute_path([home_q], num_steps=300, verbose=verbose)
        
        if verbose:
            print("[CONTROLLER] Reached home position!")
        return True
    
    def move_to_target(self, target_pos, target_quat, num_steps=200, verbose=True):
        current_q = self.data.qpos[:7].copy()
        
        if verbose:
            print(f"\n[CONTROLLER] Planning path to target...")
            current_hand_pos = self.data.body("hand").xpos.copy()
            dist = np.linalg.norm(target_pos - current_hand_pos)
            print(f"  Distance from Hand to Target: {dist:.4f} meters")
            print(f"  Current Joint Configuration: {current_q}")
        
        # Solve IK
        if verbose:
            print(f"  Solving inverse kinematics...")
        q_target, ik_success = self.ik_solver.solve(
            target_pos,
            target_quat,
            q_init=current_q.copy(),
        )
        
        if not ik_success:
            if verbose:
                print("[CONTROLLER] IK solver failed to find solution!")
            return False
        
        if verbose:
            print(f"  IK Solution: {q_target[:7]}")
        
        self.data.qpos[:7] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        # Plan RRT path
        if verbose:
            print(f"  Planning collision-free path with RRT...")
        path = self.planner.plan(current_q, q_target[:7], max_iter=5000, step_size=0.2, goal_bias=0.1)
        
        if path:
            if verbose:
                print(f"[CONTROLLER] Path found with {len(path)} waypoints! Moving to target...")
            self._execute_path(path, num_steps=num_steps, verbose=verbose)
            if verbose:
                print("[CONTROLLER] Target reached!")
            return True
        else:
            if verbose:
                print("[CONTROLLER] Failed to find collision-free path!")
            return False
    
    def _execute_path(self, path, num_steps=200, verbose=False):
        last_waypoint = self.data.qpos[:7].copy()
        
        for waypoint_idx, next_waypoint in enumerate(path):
            for s in range(num_steps):
                alpha = s / num_steps
                q_interp = (1 - alpha) * last_waypoint + alpha * next_waypoint
                
                self.data.ctrl[:7] = q_interp
                for _ in range(5): 
                    self.sim.step()
                    if self.sim.check_robot_obstacle_collision():
                        raise CollisionDetectedError()
                #time.sleep(0.005)
                self.step_count += 1
                
            last_waypoint = next_waypoint.copy()
            
            if verbose and waypoint_idx % 5 == 0:
                print(f"    Progress: {waypoint_idx}/{len(path)} waypoints")


    def move_cartesian_linear(self, start_pos, target_pos, target_quat, num_steps=50):
        print(f"[IK] Executing Cartesian Linear move over {num_steps} steps...")
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            current_target_pos = (1 - alpha) * start_pos + alpha * target_pos

            q_step, success = self.ik_solver.solve(
                current_target_pos, 
                target_quat, 
                q_init=self.data.qpos[:7]
            )
            
            if success:
                self.data.ctrl[:7] = q_step[:7]
                for _ in range(5): 
                    self.sim.step()
                    if self.sim.check_robot_obstacle_collision():
                        raise CollisionDetectedError()
                #time.sleep(0.005)
                self.step_count += 1

            else:
                print(f"Warning: IK failed at step {i} of linear path")

    def reset_step_count(self):
        self.step_count = 0
    
    def get_step_count(self):
        return self.step_count
