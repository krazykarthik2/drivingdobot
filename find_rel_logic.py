import os
import inspect
import robosuite
from robosuite.environments.manipulation import manipulation_env

print("robosuite path:", robosuite.__file__)
# Inspect how robosuite computes eef relative vector
for f in os.listdir(os.path.join(os.path.dirname(robosuite.__file__), "environments", "manipulation")):
    if f.endswith(".py"):
        full = os.path.join(os.path.dirname(robosuite.__file__), "environments", "manipulation", f)
        with open(full, encoding="utf-8", errors="ignore") as fp:
            c = fp.read()
            if "_to_robot0_eef_pos" in c or "_to_eef_pos" in c:
                print(f"Found relative key logic in: {f}")
