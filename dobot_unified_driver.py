"""
Unified Dobot Magician Controller Supporting Dual Protocols:
1. Native Serial / Protocol Communication (pydobot) - works cross-platform with direct USB serial, queue unstick, & alarm clearing
2. Official Dobot DLL (DobotDll.dll) - official C++ DLL driver

Automatically selects whichever connection is available, and falls back to safe simulation dry-run when hardware is not plugged in.
"""
import os
import sys
import time
import struct
import numpy as np
from serial.tools import list_ports

# Check pydobot availability
try:
    import pydobot
    from pydobot.message import Message
    HAS_PYDOBOT = True
except ImportError:
    HAS_PYDOBOT = False

# DobotDll setup
DLL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dobot_dlls"))
if hasattr(os, "add_dll_directory") and os.path.exists(DLL_DIR):
    os.add_dll_directory(DLL_DIR)
sys.path.append(DLL_DIR)

try:
    import DobotDllType as dType
    HAS_DLL = True
except Exception:
    HAS_DLL = False

class DobotUnifiedDriver:
    def __init__(self, port=None, baudrate=115200, prefer_driver="auto"):
        self.port = port
        self.baudrate = baudrate
        self.prefer_driver = prefer_driver
        self.mode = None # 'pydobot', 'dll', or 'mock'
        self.device = None
        self.api = None
        self.connected = False

    def connect(self):
        """Attempts connection using pydobot first (most reliable direct USB protocol), then DLL."""
        # Detect active ports and rank them by hardware description
        all_ports = list(list_ports.comports())
        print(f"[INFO] Detected system serial ports:")
        for p in all_ports:
            print(f"       -> {p.device}: {p.description}")

        # Dobot/USB-to-Serial chips (CH9102, CH340, FTDI, CP210x, Dobot, Prolific)
        priority_keywords = ["ch9102", "ch340", "ch341", "usb-enhanced-serial", "cp210", "ftdi", "dobot", "usb-serial", "usb serial"]
        
        priority_ports = []
        regular_ports = []
        bluetooth_ports = []

        for p in all_ports:
            desc_lower = (p.description or "").lower()
            hwid_lower = (p.hwid or "").lower()
            
            if any(k in desc_lower or k in hwid_lower for k in priority_keywords):
                priority_ports.append(p.device)
            elif "bluetooth" in desc_lower:
                # Bluetooth serial ports often freeze on Windows when opening, put them last
                bluetooth_ports.append(p.device)
            else:
                regular_ports.append(p.device)

        sorted_ports = priority_ports + regular_ports + bluetooth_ports

        if self.port:
            # If explicit port passed, check it first, followed by others as fallback
            target_ports = [self.port] + [p for p in sorted_ports if p != self.port]
        else:
            target_ports = sorted_ports

        print(f"[INFO] Prioritized port search order: {target_ports}")

        # 1. Try PyDobot Native Serial Protocol (proven in working_python_version.py)
        if HAS_PYDOBOT and self.prefer_driver in ["auto", "pydobot"]:
            for p in target_ports:
                if not p:
                    continue
                try:
                    print(f"[INFO] Trying native serial connection on {p}...")
                    dev = pydobot.Dobot(port=p, verbose=False)
                    
                    # Fix queue and clear alarms (proven snippet from working_python_version.py)
                    self._init_pydobot_queue_and_alarms(dev)

                    pose = dev.pose()
                    self.device = dev
                    self.mode = "pydobot"
                    self.connected = True
                    print(f"[SUCCESS] Connected via native serial on {p}! Current pose: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}")
                    return True
                except Exception as e:
                    # Not a dobot on this port
                    pass

        # 2. Try Official DobotDll.dll
        if HAS_DLL and self.prefer_driver in ["auto", "dll"]:
            try:
                print(f"[INFO] Loading DobotDll.dll...")
                api = dType.load()
                dll_port = self.port if self.port else ""
                res = dType.ConnectDobot(api, dll_port, self.baudrate)
                if res[0] == dType.DobotConnect.DobotConnect_NoError:
                    self.api = api
                    self.mode = "dll"
                    self.connected = True
                    dType.SetQueuedCmdClear(self.api)
                    dType.SetQueuedCmdStartExec(self.api)
                    dType.SetPTPCoordinateParams(self.api, 150, 150, 150, 150, isQueued=0)
                    dType.SetPTPCommonParams(self.api, 100, 100, isQueued=0)
                    pose = dType.GetPose(self.api)
                    print(f"[SUCCESS] Connected via DobotDll on {res[1]}! Pose: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}")
                    return True
            except Exception as e:
                print(f"[INFO] DobotDll check: {e}")

        # 3. Fallback: Offline Mock / Dry-Run Mode
        self.mode = "mock"
        self.connected = False
        print("[NOTICE] Real Dobot hardware not connected or USB unplugged.")
        print("[NOTICE] Operating in SIMULATION / DRY-RUN mode.")
        return False

    def _init_pydobot_queue_and_alarms(self, dev):
        """Clears queue, starts queue execution, and clears hardware alarms (msg id 20, 240, 245)."""
        # Set proper PTP speed & acceleration so the motors physically move smoothly
        try:
            dev.speed(velocity=100.0, acceleration=100.0)
        except Exception:
            pass

        # Clear Queue (245)
        msg_clear = Message()
        msg_clear.id = 245
        msg_clear.ctrl = 0x01
        dev._send_command(msg_clear)
        time.sleep(0.05)

        # Start Queue (240)
        msg_start = Message()
        msg_start.id = 240
        msg_start.ctrl = 0x01
        dev._send_command(msg_start)
        time.sleep(0.05)

        # Clear Alarms (20)
        msg_alarm = Message()
        msg_alarm.id = 20
        msg_alarm.ctrl = 0x01
        msg_alarm.params = bytearray()
        dev._send_command(msg_alarm)
        time.sleep(0.1)

        # Ensure suction and gripper are explicitly OFF (stop sucking immediately)
        msg_suck = Message()
        msg_suck.id = 62 # SET_END_EFFECTOR_SUCTION_CUP
        msg_suck.ctrl = 0x01 # Immediate
        msg_suck.params = bytearray([0x00, 0x00]) # disable enable, disable suck
        dev._send_command(msg_suck)

        msg_grip = Message()
        msg_grip.id = 63 # SET_END_EFFECTOR_GRIPPER
        msg_grip.ctrl = 0x01 # Immediate
        msg_grip.params = bytearray([0x00, 0x00])
        dev._send_command(msg_grip)
        time.sleep(0.05)

    def get_pose(self):
        """Returns physical coordinates [x, y, z, r, j1, j2, j3, j4]."""
        if self.mode == "pydobot" and self.device:
            p = self.device.pose()
            return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "r": float(p[3]),
                    "j1": float(p[4]), "j2": float(p[5]), "j3": float(p[6]), "j4": float(p[7])}
        elif self.mode == "dll" and self.api:
            p = dType.GetPose(self.api)
            return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "r": float(p[3]),
                    "j1": float(p[4]), "j2": float(p[5]), "j3": float(p[6]), "j4": float(p[7])}
        else:
            return {"x": 200.0, "y": 0.0, "z": 100.0, "r": 0.0, "j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0}

    def move_to_xyz(self, x_mm, y_mm, z_mm, r_deg=0.0, wait=True, tolerance_mm=4.0, timeout_sec=4.0):
        """
        Moves robot to target Cartesian position in mm, actively verifying
        that the physical arm tracks and reaches the destination.
        If the arm is lagging or stationary, re-pulses the command until reached.
        """
        if self.mode == "pydobot" and self.device:
            self.device.move_to(x_mm, y_mm, z_mm, r_deg, wait=False)
            if wait:
                t0 = time.time()
                last_reissue = t0
                while time.time() - t0 < timeout_sec:
                    try:
                        p = self.device.pose()
                        err = ((p[0] - x_mm)**2 + (p[1] - y_mm)**2 + (p[2] - z_mm)**2)**0.5
                        if err <= tolerance_mm:
                            return True
                        # If robot hasn't reached within 1.0 second, re-pulse command to ensure queue execution
                        if time.time() - last_reissue > 1.0:
                            self.device.move_to(x_mm, y_mm, z_mm, r_deg, wait=False)
                            last_reissue = time.time()
                    except Exception:
                        pass
                    time.sleep(0.04)
                # Read final pose
                final_p = self.device.pose()
                final_err = ((final_p[0] - x_mm)**2 + (final_p[1] - y_mm)**2 + (final_p[2] - z_mm)**2)**0.5
                return final_err <= tolerance_mm
            return True
        elif self.mode == "dll" and self.api:
            cmd_idx = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x_mm, y_mm, z_mm, r_deg, isQueued=1)[0]
            if wait:
                self._wait_dll_cmd(cmd_idx)
            return True
        else:
            print(f"[DRY-RUN MOVE] X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm, R={r_deg:.1f}deg")
            return True

    def home(self, wait_sec=15.0):
        """Sends the physical robot to its factory HOME calibration position."""
        if self.mode == "pydobot" and self.device:
            print("[INFO] Sending hardware HOME calibration command (id=31)...")
            msg = Message()
            msg.id = 31 # SetHOMECmd
            msg.ctrl = 0x03 # Queued
            msg.params = bytearray() # ZERO BYTES
            self.device._send_command(msg)
            if wait_sec > 0:
                print(f"[INFO] Waiting {wait_sec:.0f}s for physical homing cycle to finish...")
                time.sleep(wait_sec)
                # Re-clear alarms and resume queue
                self._init_pydobot_queue_and_alarms(self.device)
            return self.get_pose()
        elif self.mode == "dll" and self.api:
            print("[INFO] Sending DobotDll SetHOMECmd...")
            dType.SetHOMECmd(self.api, temp=0, isQueued=0)
            if wait_sec > 0:
                time.sleep(wait_sec)
            return self.get_pose()
        else:
            print("[DRY-RUN HOME] Calibrated to default home pose (200, 0, 140)")
            return self.get_pose()

    def set_suction(self, enable=False):
        """Explicitly enables or disables suction pump."""
        if self.mode == "pydobot" and self.device:
            msg = Message()
            msg.id = 62 # SET_END_EFFECTOR_SUCTION_CUP
            msg.ctrl = 0x01 # Immediate write
            msg.params = bytearray([0x01 if enable else 0x00, 0x01 if enable else 0x00])
            self.device._send_command(msg)
        elif self.mode == "dll" and self.api:
            dType.SetEndEffectorSuctionCup(self.api, 1 if enable else 0, 1 if enable else 0, isQueued=0)
        else:
            print(f"[DRY-RUN SUCTION] {'ON' if enable else 'OFF'}")

    def set_gripper(self, state, wait=False):
        """Controls gripper: 'open', 'close', or 'off'."""
        if self.mode == "pydobot" and self.device:
            # Always ensure suction is off when manipulating gripper or in neutral
            self.set_suction(False)
            enable = True if state in ["open", "close"] else False
            grip = True if state == "close" else False
            msg = Message()
            msg.id = 63
            msg.ctrl = 0x01
            msg.params = bytearray([0x01 if enable else 0x00, 0x01 if grip else 0x00])
            self.device._send_command(msg)
            if wait:
                time.sleep(0.15)
        elif self.mode == "dll" and self.api:
            dType.SetEndEffectorSuctionCup(self.api, 0, 0, isQueued=0)
            if state == "close":
                cmd = dType.SetEndEffectorGripper(self.api, 1, 1, isQueued=0)
            elif state == "open":
                cmd = dType.SetEndEffectorGripper(self.api, 1, 0, isQueued=0)
            else:
                cmd = dType.SetEndEffectorGripper(self.api, 0, 0, isQueued=0)
            if wait:
                time.sleep(0.15)
        else:
            print(f"[DRY-RUN GRIPPER] State set to: {state.upper()}")

    def _wait_dll_cmd(self, cmd_idx, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if dType.GetQueuedCmdCurrentIndex(self.api)[0] >= cmd_idx:
                break
            time.sleep(0.02)

    def close(self):
        if self.mode == "pydobot" and self.device:
            try:
                self.device.close()
            except Exception:
                pass
        elif self.mode == "dll" and self.api:
            try:
                dType.DisconnectDobot(self.api)
            except Exception:
                pass
        self.connected = False
        print("[INFO] Robot connection closed.")

if __name__ == "__main__":
    driver = DobotUnifiedDriver()
    driver.connect()
    pose = driver.get_pose()
    print("Initial Pose readout:", pose)
    driver.move_to_xyz(200, 0, 100)
    driver.close()
