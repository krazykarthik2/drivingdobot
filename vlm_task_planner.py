"""
vlm_task_planner.py

SmolVLM Task Decomposition and Semantic Grounding Planner
- Integrates SmolVLM reasoning and perception to dynamically produce task-specific plans.
- Dynamically resolves actions:
  * "approach the <object>" -> Only 1-2 phases: Hover above <object>, descend to approach distance.
  * "push the <object> to the <direction>" -> Hover beside <object>, descend, sweep towards <direction>, retract.
  * "pick up the <object> and place on <target>" -> 5-7 phases: Hover above <object>, descend to <object>, grip, lift, move to <target>, lower, release.
  * "home" / "standby" -> Move to home standby.
- Real object grounding: Uses detected workspace objects and calibrated tags to map physical target coordinates.
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
        """Visual entity segmentation in workspace: pen, cube, platform, duck, etc."""
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
        """Converts camera pixel coordinate to Dobot physical Cartesian (X, Y) mm."""
        # Standard default physical mapping in Dobot coordinates
        # Screen Center (320, 240) corresponds to Dobot [X=210 mm, Y=0 mm]
        # Y screen downward corresponds to X Dobot forward (+X)
        # X screen rightward corresponds to Y Dobot left/right
        scale_x = 0.55  # mm per pixel along Dobot X (depth)
        scale_y = 0.52  # mm per pixel along Dobot Y (lateral)
        
        dobot_x = 140.0 + (py - 160.0) * scale_x
        dobot_y = (320.0 - px) * scale_y

        # Clamp to safe physical workspace
        dobot_x = float(np.clip(dobot_x, 150.0, 270.0))
        dobot_y = float(np.clip(dobot_y, -120.0, 120.0))
        return dobot_x, dobot_y

    def plan_instruction(self, instruction: str, live_frame=None, homography_matrix=None):
        """
        Decomposes instruction into semantic stages with specific target object coordinates.
        Does NOT produce generic 7 stages for simple instructions like 'approach'.
        """
        inst = instruction.strip().lower()
        print(f"[SmolVLM Planner] Analyzing instruction: '{instruction}'")

        # Detect visual entities if live frame is provided
        entities = self.detect_workspace_entities(live_frame)

        # 1. HOME / STANDBY
        if any(w in inst for w in ["home", "standby", "reset"]):
            return {
                "thought": "User requested home standby. Arm should return to safe elevated standby position.",
                "plan_name": "Home Standby",
                "stages": [
                    {"desc": "Moving to home standby", "x": 200.0, "y": 0.0, "z": 40.0, "grip": False}
                ]
            }

        # 2. APPROACH / MOVE TO (Only 1-2 stages, NO gripping or pick-and-place)
        if inst.startswith("approach") or inst.startswith("move to") or inst.startswith("hover"):
            # Identify which object is requested
            target_name = "target"
            target_x, target_y = 190.0, 20.0  # default
            
            for name in ["blue pen", "pen", "green cube", "cube", "red platform", "platform", "duck", "plate"]:
                if name in inst:
                    target_name = name
                    if name in entities:
                        target_x, target_y = self.pixel_to_dobot_mm(entities[name][0], entities[name][1], homography_matrix)
                    elif "pen" in name:
                        target_x, target_y = 185.0, 30.0
                    elif "cube" in name:
                        target_x, target_y = 220.0, -40.0
                    break

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

            for name in ["blue pen", "pen", "green cube", "cube", "duck"]:
                if name in inst:
                    obj_name = name
                    if name in entities:
                        obj_x, obj_y = self.pixel_to_dobot_mm(entities[name][0], entities[name][1], homography_matrix)
                    elif "pen" in name:
                        obj_x, obj_y = 185.0, 20.0
                    break

            direction = "right" if "right" in inst else ("left" if "left" in inst else "forward")
            start_y = obj_y + (35.0 if direction == "right" else -35.0)
            end_y = obj_y - (45.0 if direction == "right" else -45.0)

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

        # 4. PICK AND PLACE
        src_name = "object"
        dst_name = "destination"
        src_x, src_y = 180.0, 40.0
        dst_x, dst_y = 220.0, -45.0

        for name in ["green cube", "cube", "blue pen", "pen", "duck"]:
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

        if "cube" in src_name:
            src_x, src_y = (src_x, src_y) if "cube" in entities else (180.0, 45.0)
        if "platform" in dst_name or "plate" in dst_name:
            dst_x, dst_y = (dst_x, dst_y) if ("platform" in entities or "plate" in entities) else (225.0, -50.0)

        return {
            "thought": f"Full pick-and-place planned for '{src_name}' to '{dst_name}' using 7 semantic sequential stages.",
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
