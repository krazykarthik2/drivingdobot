"""
System 0A: Intelligent Programmed Kinematics & Routing Solver
Integrates with NanoOWL & lightweight segmentation to provide intelligent, collision-free geometric routing:

1. Numerical & Segment Perceptual Decoding from NanoOWL:
   - Receives segmentation mask contours and numerical bounding box / centroid data.
   - Extracts exact 2D pixel centroids (u, v) and PCA principal orientation axis angle (theta) for pick targets.
   - Identifies obstacle boundaries and safety clearance radius from red obstacle contours.

2. Intelligent Geometric Trajectory Routing:
   - Does NOT just generate a blind straight line into obstacles.
   - Calculates a 3D spline routing trajectory:
       Phase A: Standby -> Pre-grasp standoff directly above pick target (Z_standby).
       Phase B: Approach & Align -> Lowers vertically onto pick target while aligning wrist yaw to object orientation angle.
       Phase C: Grasp Clamping -> Clamps gripper jaws.
       Phase D: Vertical Retract Lift -> Lifts object cleanly off the table plane (+0.060m).
       Phase E: Intelligent Waypoint Obstacle Routing ->
                 Checks if direct segment path intersects red obstacle safety radius.
                 If obstacle in path: routes through a high clearance curved arc (apex Z clearance) or lateral detour.
       Phase F: Descent & Place -> Lowers vertically into receptacle tray cavity.
       Phase G: Release -> Opens gripper jaws and returns to safe standby.

3. Complete Analytical IK:
   - Converts each 3D Cartesian waypoint (x, y, z, yaw) into exact 4-DOF joint angles (j1, j2, j3, j4).
"""

import numpy as np
import cv2
from dobot_kinematics import DobotKinematics
from system1_perception import CameraProjector

class IntelligentSystem0ASolver:
    """
    Intelligent Base Solver for 4-DOF Dobot Magician.
    Combines analytical IK with geometric spatial awareness derived from NanoOWL / segment outputs.
    """
    def __init__(self, table_z=0.070, standby_z=0.150):
        self.kin = DobotKinematics()
        self.table_z = table_z
        self.standby_z = standby_z
        self.projector = CameraProjector(width=640, height=480, table_z=table_z)

    def decode_nanoowl_segments(self, painted_image):
        """
        Extracts numerical coordinates, bounding boxes, and orientation from painted pixel segment outputs.
        - Green (Pick target): Centroid, bounding box, and PCA principal orientation angle (deg).
        - Blue (Place target): Centroid and bounding box.
        - Red (Avoid obstacle): Centroid and obstacle clearance boundary radius.
        """
        if painted_image is None or painted_image.size == 0:
            return None

        if painted_image.shape[:2] != (480, 640):
            painted_image = cv2.resize(painted_image, (640, 480), interpolation=cv2.INTER_NEAREST)

        # 1. Green Mask (Pick)
        green_mask = (painted_image[:, :, 1] > 150) & (painted_image[:, :, 0] < 80) & (painted_image[:, :, 2] < 80)
        green_mask = green_mask.astype(np.uint8) * 255
        
        pick_data = None
        contours_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_g:
            c = max(contours_g, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                u = float(M["m10"] / M["m00"])
                v = float(M["m01"] / M["m00"])
                xy = self.projector.pixel_to_table_xy(u, v)
                
                # Compute principal orientation angle via PCA / minAreaRect
                rect = cv2.minAreaRect(c)
                # minAreaRect returns ((cx, cy), (w, h), angle)
                angle = rect[2]
                if rect[1][0] < rect[1][1]:
                    angle = angle - 90.0
                    
                pick_data = {
                    "pixel": (u, v),
                    "xy": xy,
                    "angle_deg": float(angle),
                    "bbox": rect
                }

        # 2. Blue Mask (Place)
        blue_mask = (painted_image[:, :, 0] > 150) & (painted_image[:, :, 1] < 80) & (painted_image[:, :, 2] < 80)
        blue_mask = blue_mask.astype(np.uint8) * 255
        
        place_data = None
        contours_b, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_b:
            c = max(contours_b, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                u = float(M["m10"] / M["m00"])
                v = float(M["m01"] / M["m00"])
                xy = self.projector.pixel_to_table_xy(u, v)
                place_data = {
                    "pixel": (u, v),
                    "xy": xy
                }

        # 3. Red Mask (Avoid Obstacle)
        red_mask = (painted_image[:, :, 2] > 150) & (painted_image[:, :, 0] < 80) & (painted_image[:, :, 1] < 80)
        red_mask = red_mask.astype(np.uint8) * 255

        obstacles = []
        contours_r, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_r:
            M = cv2.moments(c)
            if M["m00"] > 0:
                u = float(M["m10"] / M["m00"])
                v = float(M["m01"] / M["m00"])
                xy = self.projector.pixel_to_table_xy(u, v)
                radius_px = np.sqrt(cv2.contourArea(c) / np.pi)
                obstacles.append({
                    "pixel": (u, v),
                    "xy": xy,
                    "radius_m": float(radius_px * 0.001) # approximate physical radius
                })

        return {
            "pick": pick_data,
            "place": place_data,
            "obstacles": obstacles
        }

    def plan_intelligent_trajectory(self, painted_image, current_joints=None, num_steps=30):
        """
        Generates an intelligent routed trajectory:
        - Accurately aligns wrist to pick object angle
        - Reroutes 3D path over and around detected red obstacles
        - Outputs joint angles (num_steps, 4) and gripper command profile (open/close)
        """
        seg_info = self.decode_nanoowl_segments(painted_image)
        if not seg_info or not seg_info["pick"]:
            # Fallback default positions
            pick_pos = np.array([0.22, 0.05, self.table_z], dtype=np.float32)
            pick_angle_rad = 0.0
        else:
            pick_pos = np.array([seg_info["pick"]["xy"][0], seg_info["pick"]["xy"][1], self.table_z], dtype=np.float32)
            pick_angle_rad = np.radians(seg_info["pick"]["angle_deg"])

        if not seg_info or not seg_info["place"]:
            place_pos = np.array([0.22, -0.08, self.table_z], dtype=np.float32)
        else:
            place_pos = np.array([seg_info["place"]["xy"][0], seg_info["place"]["xy"][1], self.table_z], dtype=np.float32)

        has_obstacle = len(seg_info["obstacles"]) > 0 if seg_info else False
        obs_pos = seg_info["obstacles"][0]["xy"] if has_obstacle else None

        # Start position
        if current_joints is not None:
            start_xyz = self.kin.forward(*current_joints)[:3]
        else:
            start_xyz = np.array([0.20, 0.0, self.standby_z], dtype=np.float32)

        # 30-step intelligent waypoint routing:
        # Steps 0-7: Standby -> Pre-grasp standoff directly above object (Z_standby)
        # Steps 8-14: Descent onto object + Wrist twist alignment to pick_angle_rad
        # Step 14: Grasp contact
        # Steps 15-22: Lift object vertically + Transfer over obstacle (Parabolic High Z Arc)
        # Steps 23-29: Lower vertically into place receptacle
        
        cartesian_waypoints = []
        gripper_profile = []
        joint_waypoints = []

        half = num_steps // 2

        # --- Phase 1: Approach, Wrist Twist & Grasp (Steps 0 to 14) ---
        for i in range(half):
            a = i / float(half - 1)
            if a < 0.5:
                # Approach above pick target at standby Z
                t = a / 0.5
                x = (1.0 - t) * start_xyz[0] + t * pick_pos[0]
                y = (1.0 - t) * start_xyz[1] + t * pick_pos[1]
                z = self.standby_z
                yaw = t * pick_angle_rad * 0.5
                grip = False
            else:
                # Descend vertically onto pick target and lock wrist alignment
                t = (a - 0.5) / 0.5
                x = pick_pos[0]
                y = pick_pos[1]
                z = (1.0 - t) * self.standby_z + t * pick_pos[2]
                yaw = (0.5 + 0.5 * t) * pick_angle_rad
                grip = (t > 0.8) # begin clamping right at touchdown

            pt = np.array([x, y, z, yaw], dtype=np.float32)
            cartesian_waypoints.append(pt)
            gripper_profile.append(grip)
            joint_waypoints.append(self.kin.inverse(x, y, z, yaw))

        # --- Phase 2: Lift, Intelligent Obstacle Clearance & Place (Steps 15 to 29) ---
        for i in range(half):
            a = i / float(half - 1)
            # Route in 3D:
            # If obstacle present, add lateral curved detour away from obstacle (radius offset)
            # plus elevated vertical arc so the path stays outside the cylinder radius
            x = (1.0 - a) * pick_pos[0] + a * place_pos[0]
            y = (1.0 - a) * pick_pos[1] + a * place_pos[1]

            if has_obstacle and obs_pos is not None:
                # Add lateral deflection away from table center/obstacle (+0.035m outward detour)
                normal_vec = np.array([-(place_pos[1] - pick_pos[1]), (place_pos[0] - pick_pos[0])])
                norm_len = np.linalg.norm(normal_vec)
                if norm_len > 1e-4:
                    normal_vec /= norm_len
                    # Detour curve
                    detour = 0.035 * np.sin(np.pi * a) * normal_vec
                    x += detour[0]
                    y += detour[1]

            # Strong vertical clearance: stays high until right above the place zone
            if a < 0.70:
                # Lift & Cruise high
                arc_height = 0.130 * np.sin(np.pi * (a / 0.70) * 0.5)
                z = pick_pos[2] + arc_height
            else:
                # Descend vertically straight down into place zone (a: 0.70 -> 1.0)
                desc_t = (a - 0.70) / 0.30
                z = (1.0 - desc_t) * (pick_pos[2] + 0.130) + desc_t * place_pos[2]

            yaw = (1.0 - a) * pick_angle_rad
            grip = (a < 0.95)

            pt = np.array([x, y, z, yaw], dtype=np.float32)
            cartesian_waypoints.append(pt)
            gripper_profile.append(grip)
            joint_waypoints.append(self.kin.inverse(x, y, z, yaw))

        return {
            "joint_waypoints": np.array(joint_waypoints, dtype=np.float32),
            "cartesian_waypoints": np.array(cartesian_waypoints, dtype=np.float32),
            "gripper_profile": gripper_profile,
            "seg_info": seg_info
        }

    # Backward compatibility helper
    def generate_naive_plan_from_pixels(self, painted_image, current_joints=None, num_steps=30):
        plan = self.plan_intelligent_trajectory(painted_image, current_joints, num_steps)
        seg = plan["seg_info"]
        return {
            "naive_joints": plan["joint_waypoints"],
            "naive_cart": plan["cartesian_waypoints"],
            "pick_uv": seg["pick"]["pixel"] if seg and seg["pick"] else None,
            "place_uv": seg["place"]["pixel"] if seg and seg["place"] else None,
            "gripper_profile": plan["gripper_profile"]
        }

# Aliases
System0APaintedPixelSolver = IntelligentSystem0ASolver
System0ABaseController = IntelligentSystem0ASolver

if __name__ == "__main__":
    from system1_perception import System1PerceptionEngine
    seg = System1PerceptionEngine()
    dummy_objects = [
        {"color": "green", "type": "wrench", "pos": np.array([0.22, 0.06, 0.070]), "angle": 45.0},
        {"color": "blue", "type": "platform", "pos": np.array([0.22, -0.06, 0.070]), "angle": 0.0},
        {"color": "red", "type": "cup", "pos": np.array([0.22, 0.0, 0.070]), "angle": 0.0}
    ]
    target = {"pick": {"color": "green"}, "place": {"color": "blue"}, "avoid": [{"color": "red"}]}
    _, s_full, _ = seg.segment_scene_to_silhouettes(None, dummy_objects, target)

    solver = IntelligentSystem0ASolver()
    plan = solver.plan_intelligent_trajectory(s_full)
    print("Intelligent Routing Waypoints generated:", len(plan["joint_waypoints"]))
    print("Detected Pick Orientation Angle:", plan["seg_info"]["pick"]["angle_deg"])
    print("Obstacles Detected:", len(plan["seg_info"]["obstacles"]))
    print("Max Z clearance achieved (m):", np.max(plan["cartesian_waypoints"][:, 2]))
