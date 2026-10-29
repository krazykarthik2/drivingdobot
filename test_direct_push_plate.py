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

for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

b_id = env.env.obj_body_id["akita_black_bowl_1"]
p_id = env.env.obj_body_id["plate_1"]

b_pos = env.env.sim.data.body_xpos[b_id].copy()
p_pos = env.env.sim.data.body_xpos[p_id].copy()

# Compute unit vector from bowl to plate
push_dir = p_pos[:2] - b_pos[:2]
dist = np.linalg.norm(push_dir)
push_unit = push_dir / dist

# Pre-push stance: 0.08m behind bowl along negative push direction
# EEF height at table contact: Z = 0.935m
standoff_pos = np.array([b_pos[0] - push_unit[0] * 0.08, b_pos[1] - push_unit[1] * 0.08, 0.935])

print("Standoff pos:", standoff_pos)
print("Plate center target:", p_pos)

# 1. Approach standoff position above
high_standoff = standoff_pos + np.array([0, 0, 0.15])
for _ in range(30):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((high_standoff - curr) * 8.0, -0.6, 0.6)
    action[6] = 1.0 # closed gripper as push blade
    obs, _, _, _ = env.step(action)

# 2. Descend to push height
for _ in range(30):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((standoff_pos - curr) * 8.0, -0.5, 0.5)
    action[6] = 1.0
    obs, _, _, _ = env.step(action)

# 3. Controlled linear push through plate center
# Target past plate center by +0.01m to ensure full entry into dish
target_end = np.array([p_pos[0] + push_unit[0] * 0.01, p_pos[1] + push_unit[1] * 0.01, 0.935])

print("Executing linear push toward plate...")
for step in range(60):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((target_end - curr) * 8.0, -0.35, 0.35)
    action[6] = 1.0
    obs, _, _, _ = env.step(action)
    
    b_curr = env.env.sim.data.body_xpos[b_id]
    p_curr = env.env.sim.data.body_xpos[p_id]
    xy_dist = np.linalg.norm(b_curr[:2] - p_curr[:2])
    if xy_dist < 0.03:
        print(f"Goal distance satisfied (<0.03m) at step {step}: xy_dist = {xy_dist:.4f}m!")
        break

# 4. Retract upwards to leave bowl settling on plate
for _ in range(25):
    action = np.zeros(7)
    action[2] = 0.5
    action[6] = 1.0
    obs, _, _, _ = env.step(action)

for _ in range(25):
    obs, _, _, _ = env.step(np.zeros(7))

b_final = env.env.sim.data.body_xpos[b_id]
p_final = env.env.sim.data.body_xpos[p_id]
final_dist = np.linalg.norm(b_final[:2] - p_final[:2])
contact = env.env.check_contact(env.env.get_object("plate_1"), env.env.get_object("akita_black_bowl_1"))
succ = env.check_success()

print("=" * 50)
print(f"Final Bowl Pos: {b_final}")
print(f"Final Plate Pos: {p_final}")
print(f"Final XY Distance: {final_dist:.4f} m (Threshold: < 0.0300 m)")
print(f"Direct Contact: {contact}")
print(f"LIBERO SUCCESS STATUS: {succ}")
print("=" * 50)

env.close()
