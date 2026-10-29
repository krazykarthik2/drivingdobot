import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(0)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
plate_id = env.env.obj_body_id["plate_1"]

bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
plate_pos = env.env.sim.data.body_xpos[plate_id].copy()

# Move at Z=0.920, but STOP before hitting plate:
# Plate rim starts at plate_center - 0.08m
push_stop_x = plate_pos[0] - 0.05
push_target = np.array([push_stop_x, plate_pos[1], 0.920])

start_pos = np.array([bowl_pos[0] - 0.06, bowl_pos[1], 0.920])

print("Moving to start position...")
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((start_pos - curr) * 10.0, -0.6, 0.6)
    action[6] = 1.0
    obs, _, _, _ = env.step(action)

print("Pushing bowl into plate bowl basin...")
for step in range(40):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((push_target - curr) * 10.0, -0.4, 0.4)
    action[6] = 1.0
    obs, _, _, _ = env.step(action)
    if env.check_success():
        print(f"*** PHYSICAL SUCCESS ACHIEVED AT PUSH STEP {step}! ***")
        break

# Retract upward
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0.4, 0, 0, 0, -1.0]))

for _ in range(40):
    obs, _, _, _ = env.step(np.zeros(7))
    if env.check_success():
        print("*** SUCCESS ACHIEVED DURING SETTLING! ***")
        break

print("Task 1 Check Success:", env.check_success())
bowl_end = env.env.sim.data.body_xpos[bowl_id]
plate_end = env.env.sim.data.body_xpos[plate_id]
print("Final Bowl:", bowl_end)
print("Final Plate:", plate_end)
print("Final XY delta:", np.linalg.norm(bowl_end[:2] - plate_end[:2]))

env.close()
