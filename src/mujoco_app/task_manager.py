import numpy as np
import time
from mujoco_app.grasp import estimate_top_down_grasp  # Correct function name
from mujoco_app.transformations import quat_xyzw_to_wxyz

class PickPlaceTask:
    def __init__(self, sim, controller):
        self.sim = sim
        self.controller = controller
        
    def run(self, q_home, object_name="sample_object", basket_body_name="basket"):

        print("\n[GEOMETRY] Calculating grasp pose from mesh...")
        self.controller.move_to_home(q_home)
        self._settle(50)

        grasp_pose = estimate_top_down_grasp(self.sim, body_name=object_name)

        target_pos = grasp_pose.position.copy()
        print(f"[GEOMETRY] Estimated grasp position: {target_pos.round(3)}")
        # Convert XYZW to WXYZ for MuJoCo
        target_quat = quat_xyzw_to_wxyz(grasp_pose.quaternion_xyzw)
        
        target_width = 0.04 

        # === PHASE 2: MOVE TO TARGET (PRE-GRASP) ===
        print(f"[STATUS] Target Acquired at: {target_pos.round(3)}")
        self.sim.data.ctrl[7:] = target_width
        
        pre_grasp_pos = target_pos + np.array([0.0, 0.0, 0.15])
        success = self.controller.move_to_target(pre_grasp_pos, target_quat)

        if success:
            self._settle(100)
    
            # === PHASE 3: DESCENT ===
            print("[PICK] Descending to mesh-calculated pose...")
            start_xyz = self.sim.data.body("hand").xpos.copy()
            self.controller.move_cartesian_linear(start_xyz, target_pos, target_quat, num_steps=150)
            self._settle(50)

            # === PHASE 4: GRASP ===
            print("[GRASP] Closing gripper...")
            self.sim.data.ctrl[7:] = 0.00
            self._settle(100)

            # === PHASE 5: LIFT ===
            print("[STATUS] Lifting...")
            current_pose = self.sim.data.body("hand").xpos.copy()
            post_grasp_lift = current_pose + np.array([0.0, 0.0, 0.3]) 
            self.controller.move_cartesian_linear(current_pose, post_grasp_lift, target_quat, num_steps=150)

            # === PHASE 6: DEPOSIT ===
            basket_pos = self.sim.data.body(basket_body_name).xpos.copy()
            basket_hover = basket_pos + np.array([0.0, 0.0, 0.25]) 
            basket_drop = basket_pos + np.array([0.0, 0.0, 0.12]) 

            if self.controller.move_to_target(basket_hover, target_quat):
                curr_pose = self.sim.data.body("hand").xpos.copy()
                self.controller.move_cartesian_linear(curr_pose, basket_drop, target_quat, num_steps=100)
                self._settle(50)
                
                print("[PLACE] Releasing...")
                self.sim.data.ctrl[7:] = 0.04 
                self._settle(100)

                # Retreat
                retreat_pose = basket_drop + np.array([0.0, 0.0, 0.2])
                self.controller.move_cartesian_linear(basket_drop, retreat_pose, target_quat, num_steps=80)
                
                self.controller.move_to_home(q_home)
                return True
        return False

    def _settle(self, steps):
        for _ in range(steps):
            self.sim.step()
            time.sleep(0.002)