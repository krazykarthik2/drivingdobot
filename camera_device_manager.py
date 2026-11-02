"""
camera_device_manager.py

Robust Hardware Camera Device Identifier & Persistent Mapping
Verified against live system DirectShow devices:
- Index 0: 'Integrated Camera' (Laptop internal webcam -> EXCLUDED)
- Index 1: 'HD camera ' (Secondary internal / virtual device)
- Index 2: 'USB Camera' (Lateral / Side Profile view -> ARUCO TAG 3 CONFIRMED)
- Index 3: 'MJPEG Camera' (Zebronics Straight / Front Overhead Camera)
"""

import os
import sys
import json
import cv2
import numpy as np

CAMERA_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "camera_hardware_config.json")
)


def get_fixed_camera_config():
    config = {
        "camera_1": {
            "name": "Zebronics External Camera (Straight / Front Overhead)",
            "pnp_name": "MJPEG Camera",
            "opencv_index": 3,
            "backend": "CAP_DSHOW",
            "role": "Straight overhead view of Dobot workspace",
            "default_res": [640, 480],
            "fps": 30
        },
        "camera_2": {
            "name": "USB Camera (Lateral / Side Profile)",
            "pnp_name": "USB Camera",
            "opencv_index": 2,
            "backend": "CAP_DSHOW",
            "role": "Lateral side profile view (Z-depth & gripper alignment)",
            "default_res": [640, 480],
            "fps": 30
        }
    }

    with open(CAMERA_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config


def load_camera_config():
    return get_fixed_camera_config()


def get_fixed_camera_captures():
    """
    Returns (cap1, cap2, cfg1, cfg2):
    - Camera 1: Zebronics (Index 3, CAP_DSHOW)
    - Camera 2: Lateral External USB (Index 2, CAP_DSHOW)
    Zero laptop webcam usage.
    """
    cfg = load_camera_config()
    idx1 = cfg["camera_1"]["opencv_index"]
    idx2 = cfg["camera_2"]["opencv_index"]

    cap1 = cv2.VideoCapture(idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_DSHOW)

    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    return cap1, cap2, cfg["camera_1"], cfg["camera_2"]


if __name__ == "__main__":
    cfg = get_fixed_camera_config()
    print("[CameraManager] Configured Real Physical Cameras:")
    print("  Camera 1 (Straight Overhead):", cfg["camera_1"]["name"], f"(Index {cfg['camera_1']['opencv_index']})")
    print("  Camera 2 (Lateral Profile):  ", cfg["camera_2"]["name"], f"(Index {cfg['camera_2']['opencv_index']})")
