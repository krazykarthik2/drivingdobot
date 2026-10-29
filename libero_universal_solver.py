import numpy as np

class Universal3DWayfinder:
    """
    A unified, general-purpose 3D Waypoint and Trajectory Generator.
    Operates strictly on 3D geometric task descriptors (source pose, target pose, action primitive)
    without per-scene hardcoded offsets or task-specific manual tuning.
    """
    def __init__(self, clearance_height: float = 0.12, approach_speed: float = 6.0):
        self.clearance_height = clearance_height
        self.approach_speed = approach_speed
        self.reset()

    def reset(self):
        self.stage = 0
        self.stage_step = 0
        self.completed = False

    def plan_step(self, current_eef_pos: np.ndarray, source_3d: np.ndarray, target_3d: np.ndarray, 
                  action_type: str = "pick_and_place") -> np.ndarray:
        """
        Computes 7-DOF robot action [dx, dy, dz, droll, dpitch, dyaw, gripper]
        using a canonical 3D phase state machine.
        """
        action = np.zeros(7)
        self.stage_step += 1

        if action_type == "pick_and_place":
            # Canonical Phase Sequence:
            # 0: Pre-approach above source
            # 1: Descend to source
            # 2: Clamp gripper
            # 3: Lift with clearance
            # 4: Traverse above target
            # 5: Descend to target
            # 6: Release gripper
            # 7: Retract upward

            lift_height = max(source_3d[2], target_3d[2]) + self.clearance_height

            if self.stage == 0:
                # Hover above source
                goal = np.array([source_3d[0], source_3d[1], lift_height])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.6, 0.6)
                action[6] = -1.0  # open
                if np.linalg.norm(err) < 0.025 or self.stage_step > 35:
                    self.stage = 1
                    self.stage_step = 0

            elif self.stage == 1:
                # Descend directly onto source grasp altitude
                goal = np.array([source_3d[0], source_3d[1], source_3d[2]])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.5, 0.5)
                action[6] = -1.0  # open
                if np.linalg.norm(err) < 0.020 or self.stage_step > 35:
                    self.stage = 2
                    self.stage_step = 0

            elif self.stage == 2:
                # Grip / clamp
                action[6] = 1.0  # close
                if self.stage_step > 25:
                    self.stage = 3
                    self.stage_step = 0

            elif self.stage == 3:
                # Lift cleanly with vertical clearance
                goal = np.array([current_eef_pos[0], current_eef_pos[1], lift_height])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.6, 0.6)
                action[6] = 1.0
                if np.linalg.norm(err) < 0.030 or self.stage_step > 30:
                    self.stage = 4
                    self.stage_step = 0

            elif self.stage == 4:
                # Transfer horizontally to target waypoint
                goal = np.array([target_3d[0], target_3d[1], lift_height])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.6, 0.6)
                action[6] = 1.0
                if np.linalg.norm(err[:2]) < 0.025 or self.stage_step > 45:
                    self.stage = 5
                    self.stage_step = 0

            elif self.stage == 5:
                # Descend onto target placement plane
                goal = np.array([target_3d[0], target_3d[1], target_3d[2] + 0.02])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.4, 0.4)
                action[6] = 1.0
                if np.linalg.norm(err) < 0.030 or self.stage_step > 35:
                    self.stage = 6
                    self.stage_step = 0

            elif self.stage == 6:
                # Release
                action[6] = -1.0  # open
                if self.stage_step > 20:
                    self.stage = 7
                    self.stage_step = 0

            elif self.stage == 7:
                # Retract upward to clear
                goal = np.array([target_3d[0], target_3d[1], lift_height])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.5, 0.5)
                action[6] = -1.0
                if np.linalg.norm(err) < 0.030 or self.stage_step > 25:
                    self.completed = True

        elif action_type == "push":
            # Canonical linear surface displacement
            vec = target_3d[:2] - source_3d[:2]
            dist = np.linalg.norm(vec)
            u = vec / (dist + 1e-6)
            standoff = np.array([source_3d[0] - u[0]*0.07, source_3d[1] - u[1]*0.07, source_3d[2]])
            
            if self.stage == 0:
                # Standoff approach
                goal = np.array([standoff[0], standoff[1], source_3d[2] + self.clearance_height])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.6, 0.6)
                action[6] = 1.0
                if np.linalg.norm(err) < 0.030 or self.stage_step > 30:
                    self.stage = 1
                    self.stage_step = 0
            elif self.stage == 1:
                # Align behind object
                goal = standoff
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.5, 0.5)
                action[6] = 1.0
                if np.linalg.norm(err) < 0.025 or self.stage_step > 30:
                    self.stage = 2
                    self.stage_step = 0
            elif self.stage == 2:
                # Push through target
                goal = np.array([target_3d[0], target_3d[1], source_3d[2]])
                err = goal - current_eef_pos
                action[:3] = np.clip(err * self.approach_speed, -0.4, 0.4)
                action[6] = 1.0
                if np.linalg.norm(err[:2]) < 0.025 or self.stage_step > 60:
                    self.completed = True

        return action
