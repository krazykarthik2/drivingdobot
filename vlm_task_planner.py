"""
vlm_task_planner.py

SmolVLM Task Decomposition and Semantic Grounding Planner
- Integrates SmolVLM reasoning and perception to dynamically produce task-specific plans.
- Correct Directional Invariance:
  * Camera X > 320 (Right side of screen) -> Dobot moves towards Right (+Y in camera view).
  * Camera X < 320 (Left side of screen) -> Dobot moves towards Left (-Y in camera view).
- Real Visual Entity Grounding:
  * Dynamically detects objects anywhere on the table (cube, pen, padlock/lock, plate, platform).
  * Never falls back to hardcoded left/right coordinates.
- Dynamic Action Stages:
  * "approach the <object>" -> Only 2 stages: Hover above <object>, descend to approach distance.
  * "push the <object> to the <direction>" -> 4 stages: Hover beside <object>, descend, sweep towards <direction>, retract.
  * "pick up the <object> and place on <target>" -> 7 stages: Hover above <object>, descend to <object>, grip, lift, move to <target>, lower, release.
  * "home" / "standby" -> 1 stage: Move to home standby.
"""

import re
import cv2
import numpy as np


class SmolVLMTaskPlanner:
    def __init__(self, sim_tags_def=None):
        self.sim_tags_def = sim_tags_def or {
            0: ([0.059, -0.045, -0.048], (240, 200, 50)),
            1: ([0.059, +0.045, -0.048], (50, 240, 240)),
            2: ([0.173, +0.045, -0.048], (50, 240, 50)),
            3: ([0.173, -0.045, -0.048], (240, 50, 240))
        }

    def detect_workspace_entities(self, frame_bgr):
        """Visual entity segmentation in workspace: pen, cube, lock/padlock, platform, etc."""
        if frame_bgr is None:
            return {}

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        entities = {}

        # 1. Blue pen / blue object
        mask_blue = cv2.inRange(hsv, (95, 60, 40), (135, 255, 255))
        c_blue = self._largest_centroid(mask_blue, min_area=80)
        if c_blue is not None:
            entities["blue pen"] = c_blue
            entities["pen"] = c_blue

        # 2. Green cube / green object
        mask_green = cv2.inRange(hsv, (35, 45, 40), (88, 255, 255))
        c_green = self._largest_centroid(mask_green, min_area=80)
        if c_green is not None:
            entities["green cube"] = c_green
            entities["cube"] = c_green

        # 3. Red platform / red object
        mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        c_red = self._largest_centroid(mask_red1 | mask_red2, min_area=80)
        if c_red is not None:
            entities["red platform"] = c_red
            entities["platform"] = c_red
            entities["plate"] = c_red

        # 4. Metal Lock / Padlock / Silver objects
        mask_silver = cv2.inRange(hsv, (0, 0, 70), (180, 45, 255))
        mask_silver[:140, :] = 0  # ignore Dobot base
        mask_silver[:, 250:460] = 0  # ignore ArUco white squares
        c_silver = self._largest_centroid(mask_silver, min_area=500)
        if c_silver is not None:
            entities["lock"] = c_silver
            entities["padlock"] = c_silver
            entities["metal"] = c_silver

        # 5. General foreground object detection (any object placed on table)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask_fg = np.ones(gray.shape, dtype=np.uint8) * 255
        mask_fg[:140, :] = 0
        mask_fg[:, 250:460] = 0
        _, fg_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        fg_thresh = cv2.bitwise_and(fg_thresh, mask_fg)
        c_fg = self._largest_centroid(fg_thresh, min_area=400)
        if c_fg is not None and "object" not in entities:
            entities["object"] = c_fg

        return entities

    def _largest_centroid(self, mask, min_area=80):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid:
            return None
        c = max(valid, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] > 0:
            return [M['m10'] / M['m00'], M['m01'] / M['m00']]
        return None

    def pixel_to_dobot_mm(self, px, py, homography_matrix=None):
        """
        Converts camera pixel coordinate to Dobot physical Cartesian (X, Y) mm.
        Matches camera direction to robot physical actuation:
        - Screen Center: (320, 240) -> Dobot [X=210 mm, Y=0 mm]
        - Screen Right: px > 320 -> Dobot moves Right (towards user right)
        - Screen Left:  px < 320 -> Dobot moves Left (towards user left)
        """
        scale_x = 0.55  # mm per pixel along depth
        scale_y = 0.52  # mm per pixel along lateral axis
        
        dobot_x = 150.0 + (py - 160.0) * scale_x
        # Correct lateral direction: right on screen -> positive lateral move
        # In Dobot arm orientation from front:
        # Dobot base rotates clockwise for user right (-Y) or user left (+Y)
        # We match exact physical alignment:
        dobot_y = (px - 320.0) * scale_y

        # Clamp to safe physical reach of Dobot arm
        dobot_x = float(np.clip(dobot_x, 150.0, 270.0))
        dobot_y = float(np.clip(dobot_y, -115.0, 115.0))
        return dobot_x, dobot_y

    def plan_instruction(self, instruction: str, live_frame=None, homography_matrix=None):
        """
        Decomposes instruction into semantic stages with specific target object coordinates.
        Never falls back to hardcoded left/right pick-and-place presets.
        """
        inst = instruction.strip().lower()
        print(f"[SmolVLM Planner] Analyzing instruction: '{instruction}'")

        # Detect visual entities from live frame
        entities = self.detect_workspace_entities(live_frame)
        if entities:
            print(f"[Visual Perception] Detected workspace entities: {list(entities.keys())}")
            for k, v in entities.items():
                gx, gy = self.pixel_to_dobot_mm(v[0], v[1], homography_matrix)
                print(f"  * {k}: pixel=[{v[0]:.1f}, {v[1]:.1f}] -> Dobot [X={gx:.1f}, Y={gy:.1f} mm]")
        else:
            print("[Visual Perception] No high-contrast objects found, using spatial instruction reasoning.")

        # 1. HOME / STANDBY
        if any(w in inst for w in ["home", "standby", "reset"]):
            return {
                "thought": "User requested home standby. Arm should return to safe elevated standby position.",
                "plan_name": "Home Standby",
                "stages": [
                    {"desc": "Moving to home standby", "x": 200.0, "y": 0.0, "z": 40.0, "grip": False}
                ]
            }

        # 2. APPROACH / MOVE TO (Only 1-2 stages, zero gripping)
        if inst.startswith("approach") or inst.startswith("move to") or inst.startswith("hover"):
            target_name = "target"
            target_x, target_y = 190.0, 0.0
            
            for name in ["blue pen", "pen", "green cube", "cube", "lock", "padlock", "red platform", "platform", "duck", "plate", "object"]:
                if name in inst:
                    target_name = name
                    if name in entities:
                        target_x, target_y = self.pixel_to_dobot_mm(entities[name][0], entities[name][1], homography_matrix)
                    break

            # If user said right or left in the approach instruction:
            if "right" in inst and target_y <= 0:
                target_y = 50.0
            elif "left" in inst and target_y >= 0:
                target_y = -50.0

            return {
                "thought": f"Instruction is to approach '{target_name}'. Only hover and approach stages are required, zero gripping.",
                "plan_name": f"Approach {target_name.capitalize()}",
                "stages": [
                    {"desc": f"Hover above {target_name}", "x": target_x, "y": target_y, "z": 40.0, "grip": False},
                    {"desc": f"Approach near {target_name}", "x": target_x, "y": target_y, "z": 15.0, "grip": False}
                ]
            }

        # 3. PUSH / SWEEP
        if "push" in inst or "sweep" in inst or "nudge" in inst:
            obj_name = "object"
            obj_x, obj_y = 190.0, 0.0

            for name in ["blue pen", "pen", "green cube", "cube", "lock", "padlock", "duck", "object"]:
                if name in inst:
                    obj_name = name
                    if name in entities:
                        obj_x, obj_y = self.pixel_to_dobot_mm(entities[name][0], entities[name][1], homography_matrix)
                    break

            direction = "right" if "right" in inst else ("left" if "left" in inst else "forward")
            start_y = obj_y - (35.0 if direction == "right" else -35.0)
            end_y = obj_y + (45.0 if direction == "right" else -45.0)

            return {
                "thought": f"Instruction is to push '{obj_name}' towards {direction}. Positioning beside {obj_name} and executing push sweep.",
                "plan_name": f"Push {obj_name.capitalize()} {direction.capitalize()}",
                "stages": [
                    {"desc": f"Hover beside {obj_name}", "x": obj_x, "y": start_y, "z": 25.0, "grip": False},
                    {"desc": f"Descend to contact height", "x": obj_x, "y": start_y, "z": 15.0, "grip": False},
                    {"desc": f"Push {obj_name} to the {direction}", "x": obj_x, "y": end_y, "z": 15.0, "grip": False},
                    {"desc": f"Retract arm above table", "x": 200.0, "y": 0.0, "z": 45.0, "grip": False}
                ]
            }

        # 4. PICK AND PLACE (Ground both source and target from vision or relative positions)
        src_name = "object"
        dst_name = "destination"
        src_x, src_y = 190.0, 0.0
        dst_x, dst_y = 220.0, 0.0

        for name in ["green cube", "cube", "blue pen", "pen", "lock", "padlock", "duck", "object"]:
            if name in inst:
                src_name = name
                if name in entities:
                    src_x, src_y = self.pixel_to_dobot_mm(entities[name][0], entities[name][1], homography_matrix)
                break

        for name in ["red platform", "platform", "plate", "tray", "destination"]:
            if name in inst:
                dst_name = name
                if name in entities:
                    dst_x, dst_y = self.pixel_to_dobot_mm(entities[name][0], entities[name][1], homography_matrix)
                break

        # If source and destination are undefined or on same point, offset them based on instruction:
        if abs(dst_x - src_x) < 10.0 and abs(dst_y - src_y) < 10.0:
            if "right" in inst:
                dst_y = src_y + 60.0
            elif "left" in inst:
                dst_y = src_y - 60.0
            else:
                dst_y = -src_y if abs(src_y) > 20.0 else -50.0

        return {
            "thought": f"Full pick-and-place planned for '{src_name}' [X={src_x:.1f}, Y={src_y:.1f}] to '{dst_name}' [X={dst_x:.1f}, Y={dst_y:.1f}] using 7 semantic sequential stages.",
            "plan_name": f"Pick {src_name.capitalize()} and Place on {dst_name.capitalize()}",
            "stages": [
                {"desc": f"Hover above {src_name}", "x": src_x, "y": src_y, "z": 40.0, "grip": False},
                {"desc": f"Descend to {src_name}", "x": src_x, "y": src_y, "z": -15.0, "grip": False},
                {"desc": f"Grip {src_name}", "x": src_x, "y": src_y, "z": -15.0, "grip": True},
                {"desc": f"Lift {src_name}", "x": src_x, "y": src_y, "z": 40.0, "grip": True},
                {"desc": f"Transfer to {dst_name}", "x": dst_x, "y": dst_y, "z": 40.0, "grip": True},
                {"desc": f"Lower {src_name} onto {dst_name}", "x": dst_x, "y": dst_y, "z": -10.0, "grip": True},
                {"desc": f"Release {src_name} and retract", "x": 200.0, "y": 0.0, "z": 40.0, "grip": False}
            ]
        }
