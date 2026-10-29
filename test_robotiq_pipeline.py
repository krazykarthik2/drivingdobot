import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(0)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    gripper_types="Robotiq140Gripper",
    camera_widths=128,
    camera_heights=128,
    render_gpu_device_id=-1,
)
obs = env.reset()

for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
plate_id = env.env.obj_body_id["plate_1"]

bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
plate_pos = env.env.sim.data.body_xpos[plate_id].copy()

print(f"Settled Bowl pos: {bowl_pos}")
print(f"Settled Plate pos: {plate_pos}")

# 1. Approach above bowl
approach_pos = np.array([bowl_pos[0], bowl_pos[1], 1.15])
for _ in range(25):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((approach_pos - curr) * 6.0, -0.6, 0.6)
    action[6] = -1.0 # open gripper
    obs, _, _, _ = env.step(action)

# 2. Descend over bowl
# EEF for Robotiq 140 is slightly different offset, let's descend to 0.950
grasp_z = 0.940
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    target = np.array([bowl_pos[0], bowl_pos[1], grasp_z])
    action = np.zeros(7)
    action[:3] = np.clip((target - curr) * 6.0, -0.5, 0.5)
    action[6] = -1.0
    obs, _, _, _ = env.step(action)

print("EEF before clamp:", obs["robot0_eef_pos"])

# 3. Clamp
for _ in range(30):
    action = np.zeros(7)
    action[6] = 1.0
    obs, _, _, _ = env.step(action)

# 4. Lift
for _ in range(35):
    action = np.zeros(7)
    action[2] = 0.6
    action[6] = 1.0
    obs, _, _, _ = env.step(action)

b_lift = env.env.sim.data.body_xpos[bowl_id].copy()
delta_z = b_lift[2] - bowl_pos[2]
print(f"Bowl pos after lift: {b_lift} (Delta Z: {delta_z:+.4f} m)")

if delta_z > 0.05:
    print("*** CLEAN LIFT SUCCESS WITH ROBOTIQ 140! ***")
    # 5. Transfer to plate
    for step in range(50):
        curr = obs["robot0_eef_pos"]
        target = np.array([plate_pos[0], plate_pos[1], 1.15])
        action = np.zeros(7)
        action[:3] = np.clip((target - curr) * 6.0, -0.6, 0.6)
        action[6] = 1.0
        obs, _, _, _ = env.step(action)

    # 6. Descend onto plate
    for step in range(35):
        curr = obs["robot0_eef_pos"]
        target = np.array([plate_pos[0], plate_pos[1], 0.960])
        action = np.zeros(7)
        action[:3] = np.clip((target - curr) * 6.0, -0.5, 0.5)
        action[6] = 1.0
        obs, _, _, _ = env.step(action)

    # 7. Release
    for step in range(25):
        action = np.zeros(7)
        action[6] = -1.0
        obs, _, _, _ = env.step(action)

    # 8. Settle
    for _ in range(20):
        obs, _, _, _ = env.step(np.zeros(7))

    succ = env.check_success()
    print(f"Final Success: {succ}")
    final_b = env.env.sim.data.body_xpos[bowl_id]
    final_p = env.env.sim.data.body_xpos[plate_id]
    print(f"Final Bowl: {final_b}, Final Plate: {final_p}")
    print(f"Final XY Delta: {np.linalg.norm(final_b[:2] - final_p[:2]):.4f} m")

env.close()
