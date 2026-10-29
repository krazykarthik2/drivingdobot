import numpy as np

class DobotKinematics:
    """Exact Analytical Forward and Inverse Kinematics for Dobot Magician 4-DOF."""
    def __init__(self):
        self.L1 = 0.138    # Base to Rear Arm joint height
        self.L2 = 0.135    # Rear arm length
        self.L3 = 0.147    # Forearm length
        self.L4 = 0.060    # Tool end-effector offset

    def forward(self, j1, j2, j3, j4):
        r = self.L2 * np.cos(j2) + self.L3 * np.cos(j2 + j3) + self.L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        z = self.L1 + self.L2 * np.sin(j2) + self.L3 * np.sin(j2 + j3)
        yaw = j1 + j4
        return np.array([x, y, z, yaw], dtype=np.float32)

    def inverse(self, x, y, z, yaw=0.0):
        j1 = np.arctan2(y, x)
        r_total = np.sqrt(x**2 + y**2)
        r = r_total - self.L4
        dz = z - self.L1

        D_sq = r**2 + dz**2
        D = np.sqrt(D_sq)

        cos_j3 = (D_sq - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_j3 = np.clip(cos_j3, -1.0, 1.0)
        j3 = -np.arccos(cos_j3)

        alpha = np.arctan2(dz, r)
        beta = np.arctan2(self.L3 * np.sin(-j3), self.L2 + self.L3 * np.cos(-j3))
        j2 = alpha + beta

        j4 = yaw - j1
        return np.array([j1, j2, j3, j4], dtype=np.float32)
