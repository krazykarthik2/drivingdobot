import os
import inspect
from robosuite.environments.manipulation import pick_place

with open(pick_place.__file__, encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "_to_robot0_eef_pos" in line or "_to_eef_pos" in line or "eef_pos - " in line or " - eef_pos" in line:
            print(f"Line {i+1}: {line.strip()}")
            for j in range(max(0, i-2), min(len(lines), i+3)):
                print(f"  {lines[j].rstrip()}")
            break
