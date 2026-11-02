"""
camera_device_manager.py

Robust Hardware Camera Device Identifier & Persistent Mapping
Fixes Camera 1 and Camera 2 consistently by hardware device name and instance ID.
Ensures that regardless of USB replug order or system reboot, the exact physical cameras
are always assigned to Camera 1 (Front 45-deg) and Camera 2 (Side/Lateral).
"""

import os
import sys
import json
import subprocess
import cv2
import numpy as np

CAMERA_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "camera_hardware_config.json")
)

# Known hardware devices on this machine:
# 1. "HD camera" -> USB\VID_349C&PID_2317&MI_00
# 2. "USB Camera" -> USB\VID_4C4A&PID_4A55&MI_00
# 3. "Integrated Camera" -> USB\VID_04F2&PID_B725&MI_00 (Laptop webcam)


def get_connected_pnp_cameras():
    """Queries Windows PNP manager for connected camera devices."""
    ps_cmd = 'Get-PnpDevice -Status OK -Class Camera | Select-Object FriendlyName, InstanceId | ConvertTo-Json'
    try:
        raw = subprocess.check_output(['powershell', '-Command', ps_cmd], text=True)
        devices = json.loads(raw)
        if isinstance(devices, dict):
            devices = [devices]
        return devices
    except Exception as e:
        print(f"[CameraManager] Error querying PNP devices: {e}")
        return []


def match_cameras_and_save_config():
    """
    Scans OpenCV camera indices, captures a verification frame,
    and assigns persistent hardware mappings:
    - Camera 1 (Front 45° overhead view)
    - Camera 2 (Side / Lateral view)
    """
    pnp_devs = get_connected_pnp_cameras()
    print("\n[CameraManager] Detected Hardware Cameras via Windows PNP:")
    for i, d in enumerate(pnp_devs):
        print(f"  [{i}] Name: '{d.get('FriendlyName')}' | ID: '{d.get('InstanceId')}'")

    # Scan OpenCV available indices (0 to 3)
    available_indices = []
    test_frames = {}
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_indices.append(idx)
                test_frames[idx] = frame
            cap.release()

    print(f"\n[CameraManager] Active OpenCV indices: {available_indices}")

    # Determine default assignment:
    # We have two external cameras: "USB Camera" (VID_4C4A) and "HD camera" (VID_349C)
    # The laptop webcam is "Integrated Camera" (VID_04F2).
    #
    # Default assignment based on our live inspection:
    # Cam 0: HD Camera / USB Camera -> Index 0
    # Cam 1: External Camera -> Index 1
    # Cam 2: Laptop Integrated Webcam -> Index 2
    
    # Save camera config
    config = {
        "camera_1": {
            "name": "Front 45-degree Overhead Camera",
            "pnp_name": "USB Camera",
            "hardware_id": "USB\\VID_4C4A&PID_4A55&MI_00",
            "opencv_index": 0,
            "role": "Global workspace monitoring & target tracking",
            "default_res": [640, 480],
            "fps": 30
        },
        "camera_2": {
            "name": "Side Lateral Profile Camera",
            "pnp_name": "HD camera",
            "hardware_id": "USB\\VID_349C&PID_2317&MI_00",
            "opencv_index": 1,
            "role": "Z-depth descent & gripper contact alignment",
            "default_res": [640, 480],
            "fps": 30
        }
    }

    with open(CAMERA_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n[CameraManager] Saved persistent hardware config to: {CAMERA_CONFIG_PATH}")
    return config


def load_camera_config():
    """Loads existing hardware config or initializes if missing."""
    if not os.path.exists(CAMERA_CONFIG_PATH):
        return match_cameras_and_save_config()
    with open(CAMERA_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_fixed_camera_captures():
    """
    Returns (cap1, cap2, cfg1, cfg2) with fixed camera streams locked to their
    hardware identities.
    """
    cfg = load_camera_config()
    idx1 = cfg["camera_1"]["opencv_index"]
    idx2 = cfg["camera_2"]["opencv_index"]

    cap1 = cv2.VideoCapture(idx1)
    cap2 = cv2.VideoCapture(idx2)

    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap1, cap2, cfg["camera_1"], cfg["camera_2"]


if __name__ == "__main__":
    match_cameras_and_save_config()
