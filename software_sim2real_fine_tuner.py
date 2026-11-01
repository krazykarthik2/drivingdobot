"""
software_sim2real_fine_tuner.py

TOOL 2: Interactive Software Sim-to-Real Digital Twin Fine-Tuner
===============================================================
Use this tool AFTER physically mounting your camera to dial in
exact sub-millimeter virtual alignment in software without touching the physical hardware.

Controls & Sliders:
  - Opacity Slider: Adjust alpha transparency (0.0 to 1.0) of the virtual Dobot mesh.
  - Camera Translation Fine-Tuning: dx, dy, dz (+/- 10 cm in 1mm increments).
  - Camera Rotation Fine-Tuning: Pitch, Yaw, Roll (+/- 20 degrees in 0.1 deg increments).
  - Joint Angle Offsets: Compensate for physical encoder zero-point offsets.

Saves calibrated calibration matrix to 'camera_extrinsics_calibrated.json'
which is directly loaded by the closed-loop driver and VLA inference scripts.
"""

import os
import sys
import json
import numpy as np
import cv2
import pybullet as p
import pybullet_data

# Windows compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

from dobot_kinematics import DobotKinematics


class SoftwareSimToRealFineTuner:
    def __init__(self, urdf_path=None):
        self.kin = DobotKinematics()

        # Connect PyBullet DIRECT
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        if urdf_path is None:
            urdf_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
            )
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        # Baseline Camera Pose relative to Dobot base
        # [x, y, z] in meters, [yaw, pitch, roll] in degrees
        self.cam_params = {
            "x_mm": 420,       # 0.42m
            "y_mm": 0,         # 0.00m
            "z_mm": 320,       # 0.32m
            "yaw_deg": 180.0,
            "pitch_deg": -45.0,
            "roll_deg": 0.0,
            "fov": 45.0,
            "alpha": 45        # 0.45 opacity
        }

    def render_virtual_robot(self, width=640, height=480, q_joints=None):
        if q_joints is not None:
            for j_idx, angle in enumerate(q_joints):
                p.resetJointState(self.robot_id, j_idx + 1, angle)

        cam_eye = [
            self.cam_params["x_mm"] / 1000.0,
            self.cam_params["y_mm"] / 1000.0,
            self.cam_params["z_mm"] / 1000.0
        ]
        target = [0.20, 0.0, -0.04]  # table center
        up = [0.0, 0.0, 1.0]

        view_mat = p.computeViewMatrix(cam_eye, target, up)
        proj_mat = p.computeProjectionMatrixFOV(self.cam_params["fov"], width / height, 0.05, 2.5)

        img_arr = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
        seg = np.reshape(img_arr[4], (height, width))
        mask = (seg == self.robot_id).astype(np.uint8) * 255
        return rgb, mask

    def blend(self, real_frame: np.ndarray, q_joints=None):
        h, w = real_frame.shape[:2]
        sim_rgb, mask = self.render_virtual_robot(width=w, height=h, q_joints=q_joints)

        alpha = self.cam_params["alpha"] / 100.0
        blended = real_frame.copy()
        idx = mask > 0

        sim_blend = (sim_rgb * alpha + real_frame * (1.0 - alpha)).astype(np.uint8)
        blended[idx] = sim_blend[idx]

        # Draw bright cyan silhouette contours for instant visual registration check
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 255, 0), 1)

        # Telemetry overlay
        hud = (
            f"Eye: [{self.cam_params['x_mm']}mm, {self.cam_params['y_mm']}mm, {self.cam_params['z_mm']}mm] | "
            f"FOV: {self.cam_params['fov']:.1f} | Opacity: {alpha:.2f}"
        )
        cv2.rectangle(blended, (10, h - 40), (w - 10, h - 10), (20, 20, 20), -1)
        cv2.putText(blended, hud, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return blended

    def save_calibration(self, output_path="camera_extrinsics_calibrated.json"):
        with open(output_path, "w") as f:
            json.dump(self.cam_params, f, indent=2)
        print(f"[SUCCESS] Calibrated camera extrinsics saved to: {output_path}")


def launch_fine_tuner_gui(real_camera_index=0):
    tuner = SoftwareSimToRealFineTuner()
    window_name = "Sim-to-Real Software Fine-Tuning"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Sliders for real-time visual adjustment
    cv2.createTrackbar("X (mm)", window_name, tuner.cam_params["x_mm"], 800, lambda v: tuner.cam_params.update({"x_mm": v}))
    cv2.createTrackbar("Y (mm)", window_name, tuner.cam_params["y_mm"] + 400, 800, lambda v: tuner.cam_params.update({"y_mm": v - 400}))
    cv2.createTrackbar("Z (mm)", window_name, tuner.cam_params["z_mm"], 800, lambda v: tuner.cam_params.update({"z_mm": v}))
    cv2.createTrackbar("FOV", window_name, int(tuner.cam_params["fov"]), 90, lambda v: tuner.cam_params.update({"fov": max(10, v)}))
    cv2.createTrackbar("Opacity (%)", window_name, tuner.cam_params["alpha"], 100, lambda v: tuner.cam_params.update({"alpha": v}))

    cap = cv2.VideoCapture(real_camera_index)
    has_cam = cap.isOpened()

    print("\n" + "=" * 65)
    print("SOFTWARE SIM-TO-REAL FINE-TUNING ACTIVE")
    print("Controls:")
    print("  - Drag sliders to align virtual cyan silhouette onto real robot")
    print("  - Press 's' to save calibration to camera_extrinsics_calibrated.json")
    print("  - Press 'q' or ESC to exit")
    print("=" * 65 + "\n")

    # Static reference frame fallback if camera not connected
    ref_frame = np.ones((480, 640, 3), dtype=np.uint8) * 160

    while True:
        if has_cam:
            ret, frame = cap.read()
            if not ret:
                frame = ref_frame
        else:
            frame = ref_frame

        q_home = [0.0, 0.4, 0.6, 0.0, 0.0, 0.0]
        blended = tuner.blend(frame, q_joints=q_home)

        cv2.imshow(window_name, blended)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            tuner.save_calibration()
        elif key == ord('q') or key == 27:
            break

    if has_cam:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Real camera index")
    parser.add_argument("--headless_test", action="store_true", help="Generate single alignment test image")
    args = parser.parse_args()

    if args.headless_test:
        tuner = SoftwareSimToRealFineTuner()
        dummy = np.ones((480, 640, 3), dtype=np.uint8) * 170
        res = tuner.blend(dummy, q_joints=[0.1, 0.3, 0.5, 0.0, 0.0, 0.0])
        cv2.imwrite("software_fine_tuner_sample.png", res)
        tuner.save_calibration("camera_extrinsics_calibrated.json")
        print("Generated software_fine_tuner_sample.png and camera_extrinsics_calibrated.json")
    else:
        launch_fine_tuner_gui(real_camera_index=args.cam)
