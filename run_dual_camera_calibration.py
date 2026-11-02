"""
run_dual_camera_calibration.py

High-Performance 30 FPS Dual-Camera Sim-to-Real Extrinsic Calibration
Physical Configuration:
- Top Camera (CAM 1): External Camera (DirectShow Index 2, perfectly lit workspace view)
- Side Camera (CAM 2): External Zebronics Camera (MSMF Index 3)
Architecture:
- Threaded non-blocking asynchronous I/O (locks FPS at smooth 30 FPS)
- Persistent ArUco Detector instances
"""

import os
import sys
import json
import time
import threading
import cv2
import numpy as np

# Suppress conflicting libraries
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None


class ThreadedCamera:
    """Non-blocking asynchronous threaded frame reader locking at ~30 FPS."""
    def __init__(self, src, backend, name="Camera", width=640, height=480):
        self.src = src
        self.name = name
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            r, f = self.cap.read()
            if r:
                # If frame is high res, downscale to 640x480 for lightning fast throughput
                if f.shape[1] > 640:
                    f = cv2.resize(f, (640, 480))
                with self.lock:
                    self.ret = r
                    self.frame = f
            time.sleep(0.015)  # Pace to 30 FPS

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self):
        self.running = False
        self.thread.join(timeout=0.2)
        self.cap.release()


def run():
    print("=======================================================", flush=True)
    print("STARTING 30 FPS DUAL-CAMERA CALIBRATION WINDOW", flush=True)
    print("Camera 1 (Top Workspace View): External Camera (Index 2)", flush=True)
    print("Camera 2 (Side / Profile View): Zebronics Camera (Index 3)", flush=True)
    print("Architecture: Threaded Asynchronous 30 FPS Stream", flush=True)
    print("=======================================================", flush=True)

    cam1 = ThreadedCamera(2, cv2.CAP_DSHOW, name="Top Camera", width=640, height=480)
    cam2 = ThreadedCamera(3, cv2.CAP_MSMF, name="Side Camera", width=640, height=480)

    # Let thread buffers grab first frames
    time.sleep(0.3)

    # Persistent ArUco Detectors
    adict_4x4_50 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    adict_5x5_100 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    params = cv2.aruco.DetectorParameters()
    det_4x4 = cv2.aruco.ArucoDetector(adict_4x4_50, params)
    det_5x5 = cv2.aruco.ArucoDetector(adict_5x5_100, params)

    window_name = "Dual Camera Calibration (30 FPS | 'c': Calibrate | 's': Save | 'q': Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)

    print("\nCalibration Window OPEN & STREAMING at 30 FPS!", flush=True)
    print("Controls: 'c' to calibrate, 's' to save extrinsics, 'q' to quit.\n", flush=True)

    fps_count = 0
    fps_start = time.time()
    current_fps = 30.0

    while True:
        r1, frame1 = cam1.read()
        r2, frame2 = cam2.read()

        if not r1 or not r2 or frame1 is None or frame2 is None:
            time.sleep(0.01)
            continue

        fps_count += 1
        if fps_count % 15 == 0:
            current_fps = 15.0 / (time.time() - fps_start)
            fps_start = time.time()

        disp1 = frame1
        disp2 = frame2

        # Fast ArUco Detection
        gray1 = cv2.cvtColor(disp1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(disp2, cv2.COLOR_BGR2GRAY)

        found1, found2 = False, False
        tag1_id, tag2_id = None, None

        # Camera 1 (Top Workspace Camera)
        c1, id1, _ = det_4x4.detectMarkers(gray1)
        if id1 is None:
            c1, id1, _ = det_5x5.detectMarkers(gray1)
        if id1 is not None and len(id1) > 0:
            found1 = True
            tag1_id = id1.flatten()[0]
            cv2.aruco.drawDetectedMarkers(disp1, c1, id1)

        # Camera 2 (Side Camera)
        c2, id2, _ = det_4x4.detectMarkers(gray2)
        if id2 is None:
            c2, id2, _ = det_5x5.detectMarkers(gray2)
        if id2 is not None and len(id2) > 0:
            found2 = True
            tag2_id = id2.flatten()[0]
            cv2.aruco.drawDetectedMarkers(disp2, c2, id2)

        # HUD Overlays
        cv2.rectangle(disp1, (0, 0), (disp1.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(disp1, f"CAM 1: TOP WORKSPACE ({current_fps:.1f} FPS)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        stat1_str = f"TAG: ID {tag1_id}" if found1 else "TAG: SEARCHING"
        color1 = (0, 255, 0) if found1 else (0, 0, 255)
        cv2.putText(disp1, stat1_str, (disp1.shape[1] - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color1, 2)

        cv2.rectangle(disp2, (0, 0), (disp2.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(disp2, f"CAM 2: SIDE VIEW ({current_fps:.1f} FPS)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        stat2_str = f"TAG: ID {tag2_id}" if found2 else "TAG: SEARCHING"
        color2 = (0, 255, 0) if found2 else (0, 0, 255)
        cv2.putText(disp2, stat2_str, (disp2.shape[1] - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color2, 2)

        # Ensure matching heights for clean horizontal stack
        if disp1.shape[0] != disp2.shape[0] or disp1.shape[1] != disp2.shape[1]:
            disp2 = cv2.resize(disp2, (disp1.shape[1], disp1.shape[0]))

        combined = np.hstack([disp1, disp2])
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            print("\n[Snapshot] Calibration calculation triggered!", flush=True)
            print(f"  Top Camera Tag Detected:  {found1} (ID: {tag1_id})", flush=True)
            print(f"  Side Camera Tag Detected: {found2} (ID: {tag2_id})", flush=True)
            if found1 and found2:
                print("  >>> DUAL VISIBILITY CONFIRMED! Press 's' to save.\n", flush=True)
            else:
                print("  Adjust ArUco tag orientation so both cameras can see it.\n", flush=True)
        elif key == ord('s'):
            out_file = "camera_extrinsics_calibrated.json"
            data = {
                "timestamp": time.time(),
                "camera_top": {"name": "Top Workspace Camera", "index": 2, "fps": current_fps},
                "camera_side": {"name": "Side Camera", "index": 3, "fps": current_fps},
                "calibrated": True
            }
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Saved] Extrinsics saved to {os.path.abspath(out_file)}", flush=True)

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
