import cv2
import numpy as np

COLOR_RGB_MAP = {
    "red": [235, 45, 45],
    "green": [40, 215, 75],
    "blue": [40, 110, 235],
    "yellow": [240, 220, 40],
    "purple": [180, 50, 230],
    "cyan": [35, 220, 225],
    "orange": [245, 140, 30]
}

def get_40deg_camera_pose():
    """
    Returns the Extrinsic 4x4 matrix for a 40-degree front elevated camera.
    Positioned in front of the Dobot Magician workspace looking down at 40 degrees.
    """
    cam_eye = np.array([0.45, 0.00, 0.26], dtype=np.float32)
    cam_tgt = np.array([0.22, 0.00, -0.05], dtype=np.float32)
    cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    fwd = (cam_tgt - cam_eye) / np.linalg.norm(cam_tgt - cam_eye)
    rgt = np.cross(fwd, cam_up)
    rgt /= np.linalg.norm(rgt)
    up = np.cross(rgt, fwd)
    
    T_cam = np.eye(4, dtype=np.float32)
    T_cam[:3, :3] = np.column_stack([rgt, up, -fwd])
    T_cam[:3, 3] = cam_eye
    return T_cam, cam_eye, cam_tgt

class CameraProjector:
    """
    Handles Camera Intrinsic and Extrinsic Calibration & Bidirectional Mapping:
    - 3D World (Dobot base frame, meters) <-> 2D Pixel coordinates (u, v) on table surface.
    - Designed specifically for the 40-degree front camera setup.
    """
    def __init__(self, width=640, height=480, fov_deg=50.0, table_z=-0.05):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.table_z = table_z
        
        # Camera Intrinsics
        yfov_rad = np.radians(fov_deg)
        fy = (height / 2.0) / np.tan(yfov_rad / 2.0)
        fx = fy  # square pixels
        cx = width / 2.0
        cy = height / 2.0
        self.K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)
        
        self.T_cam, self.cam_eye, self.cam_tgt = get_40deg_camera_pose()
        # Camera coordinate system: x right, y up, z backwards (-fwd)
        # Standard OpenCV camera: x right, y down, z forward
        R_world_to_cam = self.T_cam[:3, :3].T
        t_world_to_cam = -R_world_to_cam @ self.T_cam[:3, 3]
        self.R_w2c = R_world_to_cam
        self.t_w2c = t_world_to_cam

        # Precompute table homography for ultra-fast bidirectional projection
        # Sample 4 reference points on table plane
        pts_3d = np.array([
            [0.16, -0.15, self.table_z],
            [0.16,  0.15, self.table_z],
            [0.32,  0.15, self.table_z],
            [0.32, -0.15, self.table_z]
        ], dtype=np.float32)
        
        src_pts = pts_3d[:, :2].astype(np.float32)
        dst_pts = np.array([self.project_3d_to_pixel(p) for p in pts_3d], dtype=np.float32)
        self.H_table_to_pixel = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_pixel_to_table = np.linalg.inv(self.H_table_to_pixel)

    def project_3d_to_pixel(self, pt_3d):
        """Projects a 3D point [x, y, z] in robot base frame to pixel coordinates [u, v]."""
        p_c = self.R_w2c @ (pt_3d - self.cam_eye)
        # In pyrender camera: x is right, y is up, -z is forward
        # In pinhole: x_cam = p_c[0], y_cam = -p_c[1], z_cam = -p_c[2]
        z_cam = -p_c[2]
        if z_cam <= 0.001:
            z_cam = 0.001
        u = self.K[0, 0] * (p_c[0] / z_cam) + self.K[0, 2]
        v = self.K[1, 1] * (-p_c[1] / z_cam) + self.K[1, 2]
        return np.array([u, v], dtype=np.float32)

    def pixel_to_table_xy(self, u, v):
        """Converts pixel coordinate (u, v) to physical Dobot (X, Y) meters on the table surface."""
        p_homo = np.array([u, v, 1.0], dtype=np.float32)
        res = self.H_pixel_to_table @ p_homo
        res /= res[2]
        return float(res[0]), float(res[1])

class System1PerceptionEngine:
    """
    System 1: The Perception Engine (Edge Segmentation).
    Translates raw camera input and System 2 semantic labels into pure zero-background
    solid colored silhouettes:
      - GREEN (0, 255, 0): Pick target shape (e.g. wrench, block)
      - BLUE (255, 0, 0): Place target shape (e.g. bin, platform)
      - RED (0, 0, 255): Obstacle shape (e.g. cup, barrier)
    Pitch-black background (0, 0, 0).
    """
    def __init__(self, width=128, height=128, table_z=0.070):
        self.out_w = width
        self.out_h = height
        self.projector = CameraProjector(width=640, height=480, table_z=table_z)

    def segment_scene_to_silhouettes(self, raw_img, world_objects, reasoning_target):
        """
        Generates synthetic silhouette image with:
          Green for pick target
          Blue for place target
          Red for avoid obstacles
        on a solid black background.
        
        Args:
           raw_img: [H, W, 3] camera image (optional or simulation ground truth)
           world_objects: list of dicts with object metadata (shape, color, pos, orientation, etc.)
           reasoning_target: dict from System 2 (pick, place, avoid)
        """
        # Pitch-black background
        silhouette_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        pick_color = reasoning_target.get("pick", {}).get("color", "green")
        place_color = reasoning_target.get("place", {}).get("color", "blue")
        avoid_colors = [a.get("color", "red") for a in reasoning_target.get("avoid", [])]

        extracted_metadata = {
            "pick": None,
            "place": None,
            "avoid": []
        }

        for obj in world_objects:
            color = obj["color"]
            obj_type = obj.get("type", "cube")
            pos = obj["pos"]
            angle = obj.get("angle", 0.0)
            
            # Map 3D pos to 2D image coordinates
            uv = self.projector.project_3d_to_pixel(pos)
            u, v = int(round(uv[0])), int(round(uv[1]))

            # Determine silhouette color role
            # Standard colors in BGR:
            # Green: (0, 255, 0)
            # Blue: (255, 0, 0)
            # Red: (0, 0, 255)
            if color == pick_color:
                role = "pick"
                render_color = (0, 255, 0) # Green for Pick
            elif color == place_color:
                role = "place"
                render_color = (255, 0, 0) # Blue for Place
            elif color in avoid_colors or color == "red":
                role = "avoid"
                render_color = (0, 0, 255) # Red for Avoid
            else:
                # Distractor not involved in task -> left black / unsegmented
                continue

            # Render exact geometric silhouette based on shape type
            if obj_type == "wrench" or obj.get("is_wrench", False):
                # Elongated shape with oriented orientation
                self._draw_wrench_silhouette(silhouette_canvas, u, v, angle, render_color)
                contour_pts = self._get_wrench_contour(u, v, angle)
            elif obj_type == "cup" or obj_type == "cylinder":
                # Circular obstacle silhouette
                cv2.circle(silhouette_canvas, (u, v), 18, render_color, -1)
                contour_pts = cv2.boxPoints(((u, v), (36, 36), 0))
            elif obj_type == "platform" or obj_type == "bin" or obj_type == "tray":
                # Larger rectangular silhouette
                self._draw_oriented_box(silhouette_canvas, u, v, 42, 42, angle, render_color)
                contour_pts = cv2.boxPoints(((u, v), (42, 42), angle))
            else:
                # Standard cube/block silhouette
                self._draw_oriented_box(silhouette_canvas, u, v, 24, 24, angle, render_color)
                contour_pts = cv2.boxPoints(((u, v), (24, 24), angle))

            item_info = {
                "pixel": (u, v),
                "phys_xy": (float(pos[0]), float(pos[1])),
                "angle": float(angle),
                "type": obj_type,
                "color": color
            }
            if role == "pick":
                extracted_metadata["pick"] = item_info
            elif role == "place":
                extracted_metadata["place"] = item_info
            elif role == "avoid":
                extracted_metadata["avoid"].append(item_info)

        # Resize to network input shape [128, 128]
        downsampled = cv2.resize(silhouette_canvas, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)
        return downsampled, silhouette_canvas, extracted_metadata

    def _draw_wrench_silhouette(self, img, u, v, angle_deg, color):
        """Draws realistic elongated wrench/tool silhouette with head and handle."""
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        # Handle (elongated rectangle)
        length = 50
        width = 12
        h_pts = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ], dtype=np.float32)
        
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_h = (h_pts @ R.T) + np.array([u, v])
        cv2.fillPoly(img, [rot_h.astype(np.int32)], color)

        # Wrench open-ended head on one side
        head_c = np.array([u + cos_a * (length/2), v + sin_a * (length/2)])
        cv2.circle(img, (int(head_c[0]), int(head_c[1])), 12, color, -1)
        # Inner slot cutout to make it distinctly a wrench
        slot_c = head_c + np.array([cos_a * 4, sin_a * 4])
        cv2.circle(img, (int(slot_c[0]), int(slot_c[1])), 5, (0, 0, 0), -1)

    def _get_wrench_contour(self, u, v, angle_deg):
        rad = np.radians(angle_deg)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pts = np.array([[-30, -12], [30, -12], [30, 12], [-30, 12]], dtype=np.float32)
        return (pts @ R.T) + np.array([u, v])

    def _draw_oriented_box(self, img, u, v, w, h, angle_deg, color):
        rect = ((u, v), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, color, -1)

if __name__ == "__main__":
    proj = CameraProjector()
    px = proj.project_3d_to_pixel(np.array([0.22, 0.0, -0.05]))
    print("Projected center (0.22, 0, -0.05) -> pixel:", px)
    xy = proj.pixel_to_table_xy(px[0], px[1])
    print("Back-projected pixel -> XY:", xy)
    
    seg = System1PerceptionEngine()
    dummy_objects = [
        {"color": "green", "type": "wrench", "pos": np.array([0.22, 0.05, -0.05]), "angle": 35.0},
        {"color": "blue", "type": "bin", "pos": np.array([0.24, -0.08, -0.05]), "angle": 0.0},
        {"color": "red", "type": "cup", "pos": np.array([0.23, -0.01, -0.05]), "angle": 0.0}
    ]
    target = {"pick": {"color": "green"}, "place": {"color": "blue"}, "avoid": [{"color": "red"}]}
    s128, s_full, meta = seg.segment_scene_to_silhouettes(None, dummy_objects, target)
    print("Silhouette rendered shape:", s128.shape)
    cv2.imwrite("test_silhouette_demo.png", s_full)
    print("Saved test_silhouette_demo.png successfully.")
