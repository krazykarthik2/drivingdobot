"""
Live camera viewer with on-the-fly camera switching, MJPEG support, and ArUco auto-calibration:
- Supports MJPEG format (cv2.CAP_PROP_FOURCC = 'MJPG') and YUYV/RGB formats automatically
- Connects to USB Webcams, external MJPEG cameras, or network/phone IP MJPEG streams
- Safe multi-camera switching: [c] cycles through available cameras (or switches back and forth)
- Pygame GUI window showing both live camera feed and rectified Octo observation
- Hotkey [c] -> SWITCH CAMERA (Camera 0 <-> Camera 1 / MJPEG stream)
- Hotkey [s] -> SAVE current calibrated crop to disk
- Hotkey [r] -> RESET calibration
- Hotkey [q] or [ESC] -> QUIT
"""
import sys
import time
import argparse
import numpy as np
import cv2
import pygame

from aruco_calibrator import ArucoAutoCalibrator

def open_camera(cam_src, force_mjpeg=True):
    """
    Opens camera device (index int or stream URL str) with MJPEG fourcc support.
    """
    print(f"[INFO] Opening Camera source '{cam_src}' (MJPEG enabled: {force_mjpeg})...")
    
    # Check if source is a URL/stream or hardware index
    if isinstance(cam_src, str) and (cam_src.startswith("http://") or cam_src.startswith("rtsp://") or cam_src.endswith(".mjpg")):
        cap = cv2.VideoCapture(cam_src)
        return cap

    # Hardware camera index
    idx = int(cam_src) if isinstance(cam_src, (int, str)) and str(cam_src).isdigit() else 0
    
    # Try DirectShow first on Windows
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)

    if cap.isOpened():
        if force_mjpeg:
            # Set MJPG codec format
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Test frame read
        ret, frame = cap.read()
        if not ret:
            # Fallback to default pixel format if MJPEG isn't supported on this specific device
            print("[INFO] Retrying camera with default pixel format...")
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap

def cv2_to_pygame_surface(cv_img):
    """Converts a BGR or RGB OpenCV image to a Pygame Surface."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def main():
    parser = argparse.ArgumentParser(description="Live ArUco Camera with MJPEG & Multi-Camera Support")
    parser.add_argument("--camera_id", default="0", help="Initial camera index (0, 1, ...) or MJPEG stream URL")
    parser.add_argument("--secondary_cam", default="1", help="Second camera index or MJPEG URL for toggling")
    parser.add_argument("--out_size", type=int, default=256, help="Output crop size for Octo (default: 256)")
    parser.add_argument("--no_mjpeg", action="store_true", help="Disable MJPEG fourcc request")
    args = parser.parse_args()

    use_mjpeg = not args.no_mjpeg
    cam_sources = [args.camera_id, args.secondary_cam]
    current_src_idx = 0
    active_src = cam_sources[current_src_idx]

    cap = open_camera(active_src, force_mjpeg=use_mjpeg)
    if not cap.isOpened():
        print(f"[WARN] Could not open camera '{active_src}'. Trying secondary camera '{cam_sources[1]}'...")
        current_src_idx = 1
        active_src = cam_sources[current_src_idx]
        cap = open_camera(active_src, force_mjpeg=use_mjpeg)
        if not cap.isOpened():
            print("[ERROR] Neither camera could be opened. Please check USB plug or stream URL.")
            return

    calibrator = ArucoAutoCalibrator(output_size=(args.out_size, args.out_size), marker_size_mm=80.0)

    # Initialize Pygame GUI window
    pygame.init()
    win_w, win_h = 960, 520
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Dobot ArUco Auto-Calibrator (Source: {active_src})")
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    print("\n" + "=" * 60)
    print("LIVE ARUCO CALIBRATION & MJPEG CAMERA SWITCHER")
    print("=" * 60)
    print(f"Primary Source:   {cam_sources[0]}")
    print(f"Secondary Source: {cam_sources[1]}")
    print(f"MJPEG FourCC:     {'ENABLED (*MJPG)' if use_mjpeg else 'DISABLED'}")
    print("-" * 60)
    print("Controls (In Pygame Window):")
    print("  [c] -> SWITCH CAMERA (Toggle Primary <-> Secondary MJPEG Cam)")
    print("  [s] -> SAVE current calibrated crop to disk")
    print("  [r] -> RESET calibration")
    print("  [q] / [ESC] -> QUIT")
    print("=" * 60 + "\n")

    running = True
    frame_count = 0
    t0 = time.time()
    save_notification_timer = 0
    last_crop = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key == pygame.K_c:
                    # Switch Camera Source
                    next_src_idx = (current_src_idx + 1) % len(cam_sources)
                    next_src = cam_sources[next_src_idx]
                    print(f"\n[INFO] Switching to Camera: '{next_src}'...")
                    cap.release()
                    new_cap = open_camera(next_src, force_mjpeg=use_mjpeg)
                    if new_cap.isOpened():
                        cap = new_cap
                        current_src_idx = next_src_idx
                        active_src = next_src
                        pygame.display.set_caption(f"Dobot ArUco Auto-Calibrator (Source: {active_src})")
                        calibrator.is_calibrated = False
                        frame_count = 0
                        t0 = time.time()
                    else:
                        print(f"[WARN] Failed to open '{next_src}', reverting to '{active_src}'...")
                        cap = open_camera(active_src, force_mjpeg=use_mjpeg)
                elif event.key == pygame.K_s:
                    if last_crop is not None:
                        cv2.imwrite("saved_calibrated_obs.png", last_crop)
                        print("[SAVED] Saved frame to 'saved_calibrated_obs.png'!")
                        save_notification_timer = time.time() + 2.0
                elif event.key == pygame.K_r:
                    calibrator.is_calibrated = False
                    print("[INFO] Calibration reset. Re-detecting markers...")

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=True)
        last_crop = calibrated_crop

        frame_count += 1
        fps = frame_count / (time.time() - t0 + 1e-5)

        # Draw UI
        screen.fill((25, 27, 34))

        # 1. Left View: Camera Debug (640x480)
        cam_surf = cv2_to_pygame_surface(debug_view)
        screen.blit(cam_surf, (15, 20))

        # 2. Right View: Calibrated Octo Crop (270x270)
        crop_display = cv2.resize(calibrated_crop, (270, 270))
        crop_surf = cv2_to_pygame_surface(crop_display)
        screen.blit(crop_surf, (670, 20))
        pygame.draw.rect(screen, (80, 200, 120), (669, 19, 272, 272), 2)

        # Information Panel below crop
        lbl_title = font_bold.render("CALIBRATED OCTO (256x256)", True, (240, 240, 240))
        screen.blit(lbl_title, (670, 305))

        status_txt = "STATUS: CALIBRATED (READY)" if calibrator.is_calibrated else "SEARCHING FOR MARKERS (0, 1, 2, 3)"
        status_col = (50, 240, 100) if calibrator.is_calibrated else (255, 100, 100)
        lbl_status = font.render(status_txt, True, status_col)
        screen.blit(lbl_status, (670, 335))

        lbl_cam = font.render(f"CAMERA: '{active_src}' (MJPEG) [Press 'c' to switch]", True, (0, 220, 255))
        screen.blit(lbl_cam, (670, 365))

        lbl_fps = font.render(f"FPS: {fps:.1f}", True, (180, 190, 210))
        screen.blit(lbl_fps, (670, 395))

        lbl_help = font.render("[c] Switch Cam | [s] Save | [r] Reset | [q] Quit", True, (140, 150, 170))
        screen.blit(lbl_help, (670, 425))

        if time.time() < save_notification_timer:
            lbl_saved = font_bold.render("FRAME SAVED TO DISK!", True, (255, 220, 0))
            screen.blit(lbl_saved, (670, 455))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    print("[INFO] Exited cleanly.")

if __name__ == "__main__":
    main()
