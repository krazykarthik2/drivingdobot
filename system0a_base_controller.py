"""
System 0A: The Programmed Solver (Base Controller)
RESTRICTION:
System 0A ONLY SEES PAINTED PIXELS (the synthetic silhouette image on black) AND NOTHING ELSE.
It does NOT have access to ground truth simulation states, simulator coordinates, or physics bodies.
It extracts the center pixel of the GREEN painted silhouette (Pick) and BLUE painted silhouette (Place),
converts those 2D pixels to (X, Y) via camera calibration homography, and computes the naive straight-line IK.
"""

import numpy as np
import cv2
from dobot_kinematics import DobotKinematics
from system1_perception import CameraProjector

class System0APaintedPixelSolver:
    """
    Base Controller operating PURELY on painted silhouette pixels on black.
    Zero knowledge of simulator states.
    """
    """
    Base Controller operating PURELY on painted silhouette pixels on black.
    Zero knowledge of simulator states.
    """
    def __init__(self, table_z=-0.04, standby_z=0.10):
        self.kin = DobotKinematics()
        self.projector = CameraProjector(width=640, height=480, table_z=table_z)
        self.table_z = table_z
        self.standby_z = standby_z

    def extract_silhouette_centroids_from_pixels(self, painted_image):
        """
        Inspects ONLY the painted pixel image:
          - Green pixels: (0, 255, 0) in BGR -> Pick target
          - Blue pixels: (255, 0, 0) in BGR -> Place target
        Computes the center pixel (u, v) using image moments.
        """
        if painted_image is None or painted_image.size == 0:
            return None, None

        # Ensure image is in 640x480 resolution for homography projection
        if painted_image.shape[:2] != (480, 640):
            painted_image = cv2.resize(painted_image, (640, 480), interpolation=cv2.INTER_NEAREST)

        # 1. Mask for Green pixels (Pick)
        # In BGR: B < 50, G > 150, R < 50
        green_mask = (painted_image[:, :, 1] > 150) & (painted_image[:, :, 0] < 80) & (painted_image[:, :, 2] < 80)
        green_mask = green_mask.astype(np.uint8) * 255
        
        pick_uv = None
        M_green = cv2.moments(green_mask)
        if M_green["m00"] > 0:
            pick_uv = (float(M_green["m10"] / M_green["m00"]), float(M_green["m01"] / M_green["m00"]))

        # 2. Mask for Blue pixels (Place)
        # In BGR: B > 150, G < 80, R < 80
        blue_mask = (painted_image[:, :, 0] > 150) & (painted_image[:, :, 1] < 80) & (painted_image[:, :, 2] < 80)
        blue_mask = blue_mask.astype(np.uint8) * 255

        place_uv = None
        M_blue = cv2.moments(blue_mask)
        if M_blue["m00"] > 0:
            place_uv = (float(M_blue["m10"] / M_blue["m00"]), float(M_blue["m01"] / M_blue["m00"]))

        return pick_uv, place_uv

    def generate_naive_plan_from_pixels(self, painted_image, current_joints=None, num_steps=30):
        """
        Takes ONLY the painted silhouette image on black.
        1. Finds green center pixel and blue center pixel.
        2. Converts pixels to table XY coordinates.
        3. Computes naive straight-line trajectory using analytical IK (ignoring obstacles and assuming flat yaw=0).
        """
        pick_uv, place_uv = self.extract_silhouette_centroids_from_pixels(painted_image)

        if pick_uv is None:
            pick_xy = (0.22, 0.0)
        else:
            pick_xy = self.projector.pixel_to_table_xy(pick_uv[0], pick_uv[1])

        if place_uv is None:
            place_xy = (0.22, -0.10)
        else:
            place_xy = self.projector.pixel_to_table_xy(place_uv[0], place_uv[1])

        # Current or standby start position
        if current_joints is not None:
            start_xyz = self.kin.forward(*current_joints)[:3]
        else:
            start_xyz = np.array([0.20, 0.0, self.standby_z], dtype=np.float32)

        pick_xyz = np.array([pick_xy[0], pick_xy[1], self.table_z], dtype=np.float32)
        place_xyz = np.array([place_xy[0], place_xy[1], self.table_z], dtype=np.float32)

        half = num_steps // 2
        naive_joints = []
        naive_cart = []

        # Waypoints: start -> pick -> place (straight-line in Cartesian, flat yaw=0.0)
        for i in range(half):
            a = i / float(half - 1)
            pt = (1.0 - a) * start_xyz + a * pick_xyz
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
            naive_joints.append(j)
            naive_cart.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))

        for i in range(half):
            a = i / float(half - 1)
            pt = (1.0 - a) * pick_xyz + a * place_xyz
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
            naive_joints.append(j)
            naive_cart.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))

        return {
            "naive_joints": np.array(naive_joints, dtype=np.float32),
            "naive_cart": np.array(naive_cart, dtype=np.float32),
            "pick_uv": pick_uv,
            "place_uv": place_uv,
            "pick_xy": pick_xy,
            "place_xy": place_xy
        }

# Alias for backward compatibility
System0ABaseController = System0APaintedPixelSolver

if __name__ == "__main__":
    from system1_perception import System1PerceptionEngine
    seg = System1PerceptionEngine()
    dummy_objects = [
        {"color": "green", "type": "wrench", "pos": np.array([0.24, 0.06, -0.04]), "angle": 30.0},
        {"color": "blue", "type": "platform", "pos": np.array([0.20, -0.08, -0.04]), "angle": 0.0}
    ]
    target = {"pick": {"color": "green"}, "place": {"color": "blue"}, "avoid": []}
    _, s_full, _ = seg.segment_scene_to_silhouettes(None, dummy_objects, target)
    
    solver = System0APaintedPixelSolver()
    plan = solver.generate_naive_plan_from_pixels(s_full)
    print("Detected Pick Pixel:", plan["pick_uv"], "-> Converted XY:", plan["pick_xy"])
    print("Detected Place Pixel:", plan["place_uv"], "-> Converted XY:", plan["place_xy"])
    print("Generated Naive Joint Plan shape:", plan["naive_joints"].shape)
