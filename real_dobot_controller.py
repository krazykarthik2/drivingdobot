"""
Real Dobot Magician Controller using official DobotDll.dll and Python API.
Supports:
- Auto-detection or explicit COM port connection
- PTP (Point-To-Point) joint and coordinate movement (MOVJ, MOVL, JUMP)
- End-effector Gripper pneumatic control (Open / Close / Off)
- Real-time pose readout (X, Y, Z, R and J1, J2, J3, J4)
- Direct integration with Octo Policy actions and ArUco physical coordinates
"""
import os
import sys
import time
import numpy as np

DLL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dobot_dlls"))
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(DLL_DIR)
sys.path.append(DLL_DIR)

import DobotDllType as dType

class RealDobotController:
    def __init__(self, port="", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.api = None
        self.connected = False

    def connect(self):
        """Loads DLL and connects to physical Dobot."""
        print(f"[INFO] Loading DobotDll from {DLL_DIR}...")
        self.api = dType.load()
        if not self.api:
            print("[ERROR] Failed to load DobotDll.")
            return False

        print(f"[INFO] Connecting to Dobot on port '{self.port}' (Baud: {self.baudrate})...")
        res = dType.ConnectDobot(self.api, self.port, self.baudrate)
        if res[0] == dType.DobotConnect.DobotConnect_NoError:
            self.connected = True
            print(f"[SUCCESS] Connected to real Dobot! Found on port: {res[1]}")
            
            # Initialize motion parameters
            dType.SetQueuedCmdClear(self.api)
            dType.SetQueuedCmdStartExec(self.api)

            # Set PTP Coordinate and Joint Parameters (Velocity & Acceleration)
            dType.SetPTPCoordinateParams(self.api, 150, 150, 150, 150, isQueued=0)
            dType.SetPTPCommonParams(self.api, 100, 100, isQueued=0)
            dType.SetPTPJumpParams(self.api, 20, 100, isQueued=0)

            # Query initial pose
            pose = self.get_pose()
            print(f"[INFO] Real Dobot Initial Pose: X={pose['x']:.1f}mm, Y={pose['y']:.1f}mm, Z={pose['z']:.1f}mm, R={pose['r']:.1f}deg")
            return True
        else:
            err_codes = {
                1: "DobotConnect_NotFound (Robot not found on port or USB unplugged)",
                2: "DobotConnect_Occupied (Port already open or in use by another software)",
            }
            err_msg = err_codes.get(res[0], f"Unknown error code {res[0]}")
            print(f"[ERROR] Could not connect to Dobot: {err_msg}")
            return False

    def get_pose(self):
        """Returns dict of current physical pose in mm and deg."""
        if not self.connected:
            return {"x": 200.0, "y": 0.0, "z": 50.0, "r": 0.0, "j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0}
        
        p = dType.GetPose(self.api)
        # Returns [x, y, z, r, j1, j2, j3, j4]
        return {
            "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "r": float(p[3]),
            "j1": float(p[4]), "j2": float(p[5]), "j3": float(p[6]), "j4": float(p[7])
        }

    def move_to_xyz(self, x_mm, y_mm, z_mm, r_deg=0.0, mode="MOVL_XYZ", wait=True):
        """
        Commands Dobot to target Cartesian position (in mm).
        mode:
            'MOVL_XYZ': Linear Cartesian interpolation
            'MOVJ_XYZ': Joint space interpolation
            'JUMP_XYZ': Parabolic arch jump (lift -> move -> descend)
        """
        if not self.connected:
            print(f"[MOCK MOVE] Target: X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm, R={r_deg:.1f}deg")
            return

        ptp_modes = {
            "JUMP_XYZ": dType.PTPMode.PTPJUMPXYZMode,
            "MOVJ_XYZ": dType.PTPMode.PTPMOVJXYZMode,
            "MOVL_XYZ": dType.PTPMode.PTPMOVLXYZMode,
        }
        ptp_mode = ptp_modes.get(mode, dType.PTPMode.PTPMOVLXYZMode)

        cmd_idx = dType.SetPTPCmd(self.api, ptp_mode, x_mm, y_mm, z_mm, r_deg, isQueued=1)[0]

        if wait:
            self._wait_for_cmd(cmd_idx)

    def set_gripper(self, state, wait=True):
        """
        Controls the pneumatic gripper:
        state:
            'open': Jaws open
            'close': Jaws grip
            'off': Pneumatic pump off
        """
        if not self.connected:
            print(f"[MOCK GRIPPER] State set to: {state.upper()}")
            return

        if state == "close":
            # Enable valve and suction/gripper motor
            cmd_idx = dType.SetEndEffectorGripper(self.api, enableCtrl=1, on=1, isQueued=1)[0]
        elif state == "open":
            cmd_idx = dType.SetEndEffectorGripper(self.api, enableCtrl=1, on=0, isQueued=1)[0]
        else: # off
            cmd_idx = dType.SetEndEffectorGripper(self.api, enableCtrl=0, on=0, isQueued=1)[0]

        if wait:
            self._wait_for_cmd(cmd_idx)
            time.sleep(0.2) # Allow pneumatic pressure to equalize

    def _wait_for_cmd(self, cmd_idx, timeout=10.0):
        """Waits until command queue executes past cmd_idx."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            current_idx = dType.GetQueuedCmdCurrentIndex(self.api)[0]
            if current_idx >= cmd_idx:
                break
            time.sleep(0.02)

    def home(self):
        """Performs homing calibration calibration sequence."""
        if not self.connected:
            return
        print("[INFO] Initiating Homing routine on real Dobot...")
        cmd_idx = dType.SetHOMECmd(self.api, temp=0, isQueued=1)[0]
        self._wait_for_cmd(cmd_idx, timeout=30.0)
        print("[INFO] Homing completed!")

    def disconnect(self):
        if self.connected and self.api:
            self.set_gripper("off", wait=False)
            dType.DisconnectDobot(self.api)
            self.connected = False
            print("[INFO] Disconnected from Dobot.")

def test_controller():
    print("=" * 60)
    print("REAL DOBOT MAGICIAN CONTROLLER (DobotDll.dll)")
    print("=" * 60)
    dobot = RealDobotController()
    connected = dobot.connect()
    
    if connected:
        print("\n[TEST 1] Moving to Home Standby Pose (X=200, Y=0, Z=100)...")
        dobot.move_to_xyz(200.0, 0.0, 100.0)

        print("\n[TEST 2] Testing Gripper Open...")
        dobot.set_gripper("open")
        time.sleep(1.0)

        print("\n[TEST 3] Testing Gripper Close...")
        dobot.set_gripper("close")
        time.sleep(1.0)

        print("\n[TEST 4] Testing Gripper Off...")
        dobot.set_gripper("off")

        dobot.disconnect()
    else:
        print("\n[NOTE] Robot is not currently plugged into USB / switched on.")
        print("Once the USB cable is plugged in and power is ON, re-run this script to control the physical arm!")

if __name__ == "__main__":
    test_controller()
