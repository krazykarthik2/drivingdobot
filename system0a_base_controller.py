import numpy as np
from dobot_kinematics import DobotKinematics

class System0ABaseController:
    """
    System 0A: The Programmed Solver (Base Controller).
    A simple, hardcoded analytical solver using Inverse Kinematics.
    - Takes pick/place 2D pixel coordinates or physical coordinates.
    - Calculates the naive, straight-line trajectory and joint angles.
    - Completely ignores obstacles (moves straight horizontally through them).
    - Ignores wrist grip angles (assumes yaw=0.0).
    - For an empty table with perfectly round objects, this naive solver reaches the target with zero AI.
    """
    def __init__(self, table_z=-0.05, standby_z=0.12):
        self.kin = DobotKinematics()
        self.table_z = table_z
        self.standby_z = standby_z

    def generate_naive_trajectory(self, start_xyz, target_xyz, num_waypoints=20):
        """
        Generates naive straight-line waypoints in Cartesian space and joint angles.
        start_xyz: [x, y, z]
        target_xyz: [x, y, z]
        Returns:
            cartesian_path: np.ndarray of shape [num_waypoints, 4] (x, y, z, yaw)
            joint_path: np.ndarray of shape [num_waypoints, 4] (j1, j2, j3, j4)
        """
        cartesian_path = np.zeros((num_waypoints, 4), dtype=np.float32)
        joint_path = np.zeros((num_waypoints, 4), dtype=np.float32)

        for i in range(num_waypoints):
            alpha = float(i) / max(1, num_waypoints - 1)
            # Naive straight line in 3D:
            # Does not curve over obstacles, ignores orientation
            x = (1.0 - alpha) * start_xyz[0] + alpha * target_xyz[0]
            y = (1.0 - alpha) * start_xyz[1] + alpha * target_xyz[1]
            z = (1.0 - alpha) * start_xyz[2] + alpha * target_xyz[2]
            yaw = 0.0 # Naive controller completely ignores object orientation

            cartesian_path[i] = [x, y, z, yaw]
            joint_path[i] = self.kin.inverse(x, y, z, yaw)

        return cartesian_path, joint_path

    def get_naive_step(self, current_xyz, target_xyz, step_size=0.015):
        """
        Computes a single naive step towards the target.
        """
        delta = target_xyz - current_xyz
        dist = np.linalg.norm(delta)
        if dist < step_size:
            next_xyz = target_xyz.copy()
        else:
            next_xyz = current_xyz + (delta / dist) * step_size
            
        next_yaw = 0.0
        naive_joints = self.kin.inverse(next_xyz[0], next_xyz[1], next_xyz[2], next_yaw)
        return next_xyz, naive_joints

if __name__ == "__main__":
    solver = System0ABaseController()
    start = np.array([0.20, 0.0, 0.05])
    target = np.array([0.24, 0.10, -0.05])
    c_path, j_path = solver.generate_naive_trajectory(start, target, num_waypoints=5)
    print("Naive Cartesian path:\n", c_path)
    print("Naive Joint path:\n", j_path)
