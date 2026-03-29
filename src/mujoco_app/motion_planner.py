import numpy as np
import mujoco
import time

class RRTPlanner:
    def __init__(self, model, data, joint_ids, ids):
        self.model = model
        self.data = data
        self.joint_ids = joint_ids # e.g., [0, 1, 2, 3, 4, 5, 6]
        self.ids = ids
        
    def is_collision_free(self, q):
        # Save current qpos
        current_q = self.data.qpos.copy()
        
        # Set new q
        self.data.qpos[self.joint_ids] = q
        mujoco.mj_forward(self.model, self.data)
        
        # Check for collisions with obstacles
        mobjs = self.ids.get("moving_obstacles", {})
        obstacle_geoms = set()
        for obstacle_name in mobjs.keys():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obstacle_name)
            if body_id >= 0:
                geoms = np.where(self.model.geom_bodyid == body_id)[0]
                obstacle_geoms.update(geoms.tolist())
        
        robot_keywords = ["link", "hand", "finger", "gripper"]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 in obstacle_geoms or g2 in obstacle_geoms:
                robot_geom = g1 if g2 in obstacle_geoms else g2
                body_id = self.model.geom_bodyid[robot_geom]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name and any(kw in body_name.lower() for kw in robot_keywords):
                    # Collision detected
                    self.data.qpos = current_q
                    mujoco.mj_forward(self.model, self.data)
                    return False
        
        # No collision
        self.data.qpos = current_q
        mujoco.mj_forward(self.model, self.data)
        return True

    def plan(self, start_q, goal_q, max_iter=1000, step_size=0.05, goal_bias=0.1):
        tree = [start_q]
        parents = {0: None}
        
        for _ in range(max_iter):
            # Goal-biased sampling: with probability goal_bias, sample the goal
            if np.random.rand() < goal_bias:
                q_rand = goal_q.copy()
            else:
                # sample random configuration within joint limits
                q_rand = np.array([np.random.uniform(self.model.jnt_range[i, 0], 
                                                   self.model.jnt_range[i, 1]) 
                                  for i in self.joint_ids])
            
            # find nearest node in tree
            distances = [np.linalg.norm(q - q_rand) for q in tree]
            nearest_idx = np.argmin(distances)
            q_near = tree[nearest_idx]
            
            # step from near to rand
            diff = q_rand - q_near
            dist = np.linalg.norm(diff)
            if dist > 0:
                q_new = q_near + (diff / dist) * min(step_size, dist)
            else:
                q_new = q_near
            
            # Clip to joint limits
            q_new = np.clip(q_new, self.model.jnt_range[self.joint_ids, 0], self.model.jnt_range[self.joint_ids, 1])
            
            # collision Check & Add to Tree
            if self.is_collision_free(q_new):
                tree.append(q_new)
                new_idx = len(tree) - 1
                parents[new_idx] = nearest_idx
                
                # check if we can reach the goal
                if np.linalg.norm(q_new - goal_q) < step_size:
                    # Trace path back
                    path = [goal_q]
                    curr = new_idx
                    while curr is not None:
                        path.append(tree[curr])
                        curr = parents[curr]
                    path = path[::-1]  # Start to Goal
                    
                    # Smooth the path
                    path = self.smooth_path(path)
                    return path
        return None
    
    def smooth_path(self, path, iterations=20):
        """
        Smooth the path using shortcutting - try to remove intermediate waypoints.
        """
        smoothed = list(path)
        
        for _ in range(iterations):
            improved = False
            # Try removing waypoints
            i = 0
            while i < len(smoothed) - 2:
                # Try to connect smoothed[i] directly to smoothed[i+2], skipping smoothed[i+1]
                if self.is_collision_free_path(smoothed[i], smoothed[i + 2]):
                    smoothed.pop(i + 1)
                    improved = True
                else:
                    i += 1
            
            if not improved:
                break
        
        return smoothed
    
    def is_collision_free_path(self, q_start, q_end, num_checks=10):
        """
        Check if path between two configs is collision-free using linear interpolation.
        """
        for t in np.linspace(0, 1, num_checks):
            q_interp = (1 - t) * q_start + t * q_end
            if not self.is_collision_free(q_interp):
                return False
        return True