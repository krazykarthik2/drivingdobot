"""
Manual Model vs Real Dobot Calibration Tool:
- Interactively calibrate the 3D URDF simulation model to match your physical Dobot by hand.
- Adjust Base Yaw, Pitch, Roll, Height (Z offset), and Scaling in real-time with keyboard.
- Compares real robot joint encoders / Cartesian pose with the 3D URDF Collada model side-by-side.
- Saves the exact calibration to 'calibration_config.json' so all programs use it automatically.

Controls:
  [LEFT] / [RIGHT]   -> Adjust Base Yaw rotation (±1°)
  [UP] / [DOWN]       -> Adjust Base Pitch (±1°)
  [PAGE UP] / [PG DN] -> Adjust Base Roll (±1°)
  [w] / [s]           -> Move Dobot X forward / backward (±5mm)
  [a] / [d]           -> Move Dobot Y left / right (±5mm)
  [e] / [c]           -> Move Dobot Z up / down (±5mm)
  [SPACE]             -> Toggle Suction On / Off
  [h]                 -> Execute Hardware Home calibration
  [ENTER]             -> SAVE calibration to 'calibration_config.json'
  [q] / [ESC]         -> Quit
"""
import os
import sys
import json
import time
import numpy as np
import cv2
import pygame

from dobot_unified_driver import DobotUnifiedDriver
from dobot_urdf_renderer import DobotURDFRenderer
from dobot_magician_ros2.ai_training.env.dobot_env import DobotKinematics

CONFIG_FILE = "calibration_config.json"

def load_config():
    default_cfg = {
        "sim_yaw_offset_deg": 0.0,
        "sim_pitch_offset_deg": 0.0,
        "sim_roll_offset_deg": 0.0,
        "x_scale": 1.0,
        "y_scale": 1.0,
        "z_scale": 1.0,
        "z_table_surface_mm": -35.0,
        "standby_x_mm": 200.0,
        "standby_y_mm": 0.0,
        "standby_z_mm": 140.0
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                default_cfg.update(cfg)
        except Exception:
            pass
    return default_cfg

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"\n[SAVED] Calibration config saved to '{CONFIG_FILE}'!")

def cv2_to_surface(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def main():
    cfg = load_config()
    print("\n" + "=" * 65)
    print("MANUAL CALIBRATION: 3D SIMULATION MODEL VS REAL DOBOT")
    print("=" * 65)
    print("Connecting to physical Dobot...")
    dobot = DobotUnifiedDriver()
    dobot.connect()

    renderer = DobotURDFRenderer(width=720, height=540)
    kin = DobotKinematics()

    pygame.init()
    screen = pygame.display.set_mode((1160, 620))
    pygame.display.set_caption("DOBOT REAL VS SIMULATION MANUAL CALIBRATOR")
    font = pygame.font.SysFont("Consolas", 15)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    cur_p = dobot.get_pose()
    target_x = cur_p["x"]
    target_y = cur_p["y"]
    target_z = cur_p["z"]
    suction = False

    saved_flash_timer = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                # Manual Rotational Offsets
                elif event.key == pygame.K_LEFT:
                    cfg["sim_yaw_offset_deg"] = (cfg["sim_yaw_offset_deg"] - 1.0) % 360.0
                    if cfg["sim_yaw_offset_deg"] > 180.0: cfg["sim_yaw_offset_deg"] -= 360.0
                elif event.key == pygame.K_RIGHT:
                    cfg["sim_yaw_offset_deg"] = (cfg["sim_yaw_offset_deg"] + 1.0) % 360.0
                    if cfg["sim_yaw_offset_deg"] > 180.0: cfg["sim_yaw_offset_deg"] -= 360.0
                elif event.key == pygame.K_UP:
                    cfg["sim_pitch_offset_deg"] += 1.0
                elif event.key == pygame.K_DOWN:
                    cfg["sim_pitch_offset_deg"] -= 1.0
                elif event.key == pygame.K_PAGEUP:
                    cfg["sim_roll_offset_deg"] += 1.0
                elif event.key == pygame.K_PAGEDOWN:
                    cfg["sim_roll_offset_deg"] -= 1.0

                # Real Robot Joint/Cartesian Test Moves
                elif event.key == pygame.K_w:
                    target_x += 10.0
                    dobot.move_to_xyz(target_x, target_y, target_z, wait=False)
                elif event.key == pygame.K_s:
                    target_x -= 10.0
                    dobot.move_to_xyz(target_x, target_y, target_z, wait=False)
                elif event.key == pygame.K_a:
                    target_y += 10.0
                    dobot.move_to_xyz(target_x, target_y, target_z, wait=False)
                elif event.key == pygame.K_d:
                    target_y -= 10.0
                    dobot.move_to_xyz(target_x, target_y, target_z, wait=False)
                elif event.key == pygame.K_e:
                    target_z += 5.0
                    dobot.move_to_xyz(target_x, target_y, target_z, wait=False)
                elif event.key == pygame.K_c:
                    target_z -= 5.0
                    dobot.move_to_xyz(target_x, target_y, target_z, wait=False)
                elif event.key == pygame.K_SPACE:
                    suction = not suction
                    dobot.set_suction(suction)
                elif event.key == pygame.K_h:
                    dobot.home(wait_sec=12.0)
                    cur_p = dobot.get_pose()
                    target_x, target_y, target_z = cur_p["x"], cur_p["y"], cur_p["z"]
                elif event.key == pygame.K_RETURN:
                    save_config(cfg)
                    saved_flash_timer = 45 # 1.5 seconds flash

        # Read actual physical state
        cur_p = dobot.get_pose()

        # Compute Kinematics & URDF transforms
        j1_raw = float(cur_p.get("j1", 0.0))
        j2_raw = float(cur_p.get("j2", 0.0))
        j3_raw = float(cur_p.get("j3", 0.0))
        j4_raw = float(cur_p.get("j4", 0.0))

        if abs(j2_raw) > 0.01 or abs(j3_raw) > 0.01:
            j1_rad = np.radians(j1_raw)
            j2_rad = np.radians(j2_raw)
            j3_rad = np.radians(j3_raw - j2_raw)
            j4_rad = np.radians(j4_raw)
        else:
            j_calc = kin.inverse(cur_p["x"]/1000.0, cur_p["y"]/1000.0, cur_p["z"]/1000.0)
            j1_rad, j2_rad, j3_rad, j4_rad = j_calc[0], j_calc[1], j_calc[2], 0.0

        # Apply manual base yaw offset
        j1_rad += np.radians(cfg["sim_yaw_offset_deg"])
        t_m = renderer.compute_urdf_transforms(j1_rad, j2_rad, j3_rad, j4_rad, gripper_closed=False)
        scene = renderer.build_scene(t_m, end_effector="suction")
        _, _, sim_frame = renderer.render_views(scene, t_m["ee_pos"])

        # Render to Screen
        screen.fill((20, 22, 28))

        # Main 3D View (720x540)
        sim_surf = cv2_to_surface(sim_frame)
        screen.blit(sim_surf, (20, 20))
        pygame.draw.rect(screen, (70, 130, 255), (19, 19, 722, 542), 2)

        # Right Panel: Live Manual Calibration Sliders & Telemetry
        px = 760
        screen.blit(font_bold.render("REAL VS SIM CALIBRATION", True, (0, 240, 180)), (px, 20))
        
        lines = [
            f"Base Yaw Offset:   {cfg['sim_yaw_offset_deg']:+6.1f}°  [LEFT/RIGHT]",
            f"Base Pitch Offset: {cfg['sim_pitch_offset_deg']:+6.1f}°  [UP/DOWN]",
            f"Base Roll Offset:  {cfg['sim_roll_offset_deg']:+6.1f}°  [PGUP/PGDN]",
            "-" * 42,
            "REAL DOBOT HARDWARE READOUT:",
            f"  X:  {cur_p['x']:+6.1f} mm     J1 (Yaw):  {j1_raw:+6.1f}°",
            f"  Y:  {cur_p['y']:+6.1f} mm     J2 (Rear): {j2_raw:+6.1f}°",
            f"  Z:  {cur_p['z']:+6.1f} mm     J3 (Fore): {j3_raw:+6.1f}°",
            f"  Sucker: {'ON (SUCKING)' if suction else 'OFF'}",
            "-" * 42,
            "TEST ROBOT JOGGING (HAND CALIB):",
            "  [w / s] -> Jog X (+/- 10 mm)",
            "  [a / d] -> Jog Y (+/- 10 mm)",
            "  [e / c] -> Jog Z (+/- 5 mm)",
            "  [SPACE] -> Toggle Suction",
            "  [h]     -> Hardware Home",
            "-" * 42,
            "SAVE CALIBRATION:",
            "  [ENTER] -> SAVE TO FILE (Permanent)",
            "  [q/ESC] -> Exit"
        ]

        py = 55
        for line in lines:
            col = (255, 215, 0) if "Offset" in line else ((200, 220, 240) if "REAL" in line or "TEST" in line else (170, 180, 200))
            screen.blit(font.render(line, True, col), (px, py))
            py += 24

        if saved_flash_timer > 0:
            saved_flash_timer -= 1
            pygame.draw.rect(screen, (30, 180, 80), (px, 510, 370, 45), border_radius=6)
            screen.blit(font_bold.render("CONFIG SAVED SUCCESSFULLY!", True, (255, 255, 255)), (px + 30, 522))

        pygame.display.flip()
        clock.tick(30)

    dobot.set_suction(False)
    renderer.close()
    dobot.close()
    pygame.quit()
    print("[INFO] Calibration tool exited.")

if __name__ == "__main__":
    main()
